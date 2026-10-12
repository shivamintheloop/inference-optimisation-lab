"""Speculative decoding on MLX (Apple's own array library), for comparison with llama.cpp.

Why: on llama.cpp b11146 Metal, the 8B's multi-token check costs 1.4-6 normal
steps (analysis/speculative-decoding.md), which kills the speedup. MLX has its
own quantized matmul kernels, so the check cost may differ.

micro():  time one target pass over N new tokens after 1024 tokens of context
          (the "check" cost), plus target and draft decode steps.
run_ab(): the frozen prompts, in 4-prompt blocks with a cool-down before each,
          variants rotated per block (same design as rag/bench_ab.py). Models
          stay loaded in this process, so switching variants is free.

Variants: "baseline" | "draft:<k>". Temperature 0 (greedy), no prompt cache
reuse between requests.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from rag.bench_ab import block_drift, paired_ratio, rotated
from rag.bench_llm import PROMPTS, file_hash, load_prompts, same_output, summarize
from rag.config import CONFIG

TARGET = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
DRAFT = "mlx-community/Llama-3.2-1B-Instruct-4bit"


def _log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _timed_pass(model, tokens, cache) -> float:
    import mlx.core as mx
    t0 = time.perf_counter()
    mx.eval(model(tokens, cache=cache))
    return (time.perf_counter() - t0) * 1000


def micro(target: str = TARGET, draft: str = DRAFT, depth: int = 1024,
          sizes=(1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 17, 24, 32), reps: int = 5) -> dict:
    """ms for one pass over N tokens on top of `depth` cached tokens (median of reps)."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

    out: dict = {"target": target, "draft": draft, "depth": depth}
    for name, repo in (("target", target), ("draft", draft)):
        model, _ = load(repo)
        cache = make_prompt_cache(model)
        ctx = mx.random.randint(0, 1000, (1, depth))
        mx.eval(model(ctx, cache=cache))                     # fill the cache once
        for n in (1, 4, 16):                                 # warm up kernels
            _timed_pass(model, mx.random.randint(0, 1000, (1, n)), cache)
            trim_prompt_cache(cache, n)
        res = {}
        for n in (sizes if name == "target" else (1,)):
            times = []
            for _ in range(reps):
                times.append(_timed_pass(model, mx.random.randint(0, 1000, (1, n)), cache))
                trim_prompt_cache(cache, n)                  # back to `depth` tokens
            res[n] = round(sorted(times)[reps // 2], 2)
        out[name] = res
        del model, cache
        mx.clear_cache()
    return out


def _generate(model, tokenizer, messages, draft_model=None, k: int = 0,
              max_tokens: int | None = None) -> tuple[str, dict]:
    from mlx_lm import stream_generate
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    kw = {"draft_model": draft_model, "num_draft_tokens": k} if draft_model is not None else {}
    t0 = time.perf_counter()
    first = None
    parts, n, from_draft, last = [], 0, 0, None
    for r in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens or CONFIG.max_output_tokens, **kw):
        if first is None:
            first = time.perf_counter()
        parts.append(r.text)
        n += 1
        from_draft += bool(r.from_draft)
        last = r
    end = time.perf_counter()
    m = {"ttft_ms": round((first - t0) * 1000, 1), "e2e_ms": round((end - t0) * 1000, 1),
         "output_tokens": n, "prompt_tokens": last.prompt_tokens if last else None,
         "done_reason": last.finish_reason if last else None}
    if n > 1:
        m["tpot_ms"] = round((end - first) * 1000 / (n - 1), 2)
        m["decode_tok_s"] = round(1000 / m["tpot_ms"], 1)
    if draft_model is not None:
        m["from_draft"] = from_draft                         # tokens that came from kept guesses
        m["from_draft_share"] = round(from_draft / n, 3) if n else None
    return "".join(parts), m


def run_ab(variants: list[str], target: str = TARGET, draft: str = DRAFT,
           prompts_path: Path = PROMPTS, limit: int | None = None, block_size: int = 4,
           pause_s: float = 8.0, idle_s: float = 120.0) -> dict:
    import mlx_lm
    from mlx_lm import load

    ks = {}
    for v in variants:
        if v == "baseline":
            ks[v] = 0
        elif v.startswith("draft:") and v[6:].isdigit():
            ks[v] = int(v[6:])
        else:
            raise ValueError(f"bad MLX variant {v!r}: use baseline or draft:<k>")
    model, tokenizer = load(target)
    draft_model = load(draft)[0] if any(ks.values()) else None
    prompts = load_prompts(prompts_path)[:limit]
    blocks = [prompts[i:i + block_size] for i in range(0, len(prompts), block_size)]
    for v in variants:                                       # compile kernels for each path
        _generate(model, tokenizer, prompts[0]["messages"], draft_model if ks[v] else None, ks[v], 16)

    rows: list[dict] = []
    started = time.time()
    for b, block in enumerate(blocks):
        for v in rotated(variants, b):
            time.sleep(idle_s)
            for q in block:
                text, m = _generate(model, tokenizer, q["messages"], draft_model if ks[v] else None, ks[v])
                rows.append({"variant": v, "pass": 0, "block": b, "id": q["id"], "text": text, **m})
                time.sleep(pause_s)
            tpots = " ".join(f"{r['tpot_ms']:.1f}" for r in rows if r["variant"] == v and r["block"] == b)
            _log(f"block {b} {v:<9} TPOT per request: {tpots} ms")

    base_rows = [r for r in rows if r["variant"] == variants[0]]
    summaries = {}
    for v in variants:
        vr = [r for r in rows if r["variant"] == v]
        s = summarize(vr)
        s["block_drift"] = block_drift(vr)
        if ks[v]:
            s["from_draft_share"] = round(sum(r["from_draft"] for r in vr) / sum(r["output_tokens"] for r in vr), 3)
        if v != variants[0]:
            s["baseline"] = variants[0]
            s["same_as_baseline"] = same_output(vr, base_rows)
            s["speedup_tpot"] = paired_ratio(vr, base_rows, "tpot_ms")
            s["speedup_answer"] = paired_ratio(vr, base_rows, "e2e_ms")
        summaries[v] = s
    result = {"time": datetime.now().isoformat(timespec="seconds"),
              "minutes": round((time.time() - started) / 60, 1), "engine": f"mlx-lm {mlx_lm.__version__}",
              "target": target, "draft": draft, "variants": variants, "prompts": prompts_path.name,
              "prompts_hash": file_hash(prompts_path), "n_prompts": len(prompts),
              "block_size": block_size, "pause_s": pause_s, "idle_s": idle_s,
              "summaries": summaries, "rows": rows}
    out = CONFIG.results_dir / f"mlx-ab-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(result, indent=2))
    result["path"] = str(out)
    return result


def as_bench_result(res: dict, variant: str) -> dict:
    """One variant shaped like a bench_llm result, for record_inference()."""
    k = variant.split(":")[1] if ":" in variant else None
    cmd = f"{res['target']}" + (f" + {res['draft']} --spec-draft-n-max {k}" if k else "")
    return {"label": f"MLX {variant} (A/B)", "path": res["path"],
            "server": {"engine": "mlx", "build": res["engine"].split()[-1], "cmd": cmd},
            "summary": res["summaries"][variant]}
