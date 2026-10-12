"""Interleaved A/B benchmark for llama-server variants (inference experiments).

Why not bench_llm? This laptop slows down under sustained GPU load (decode
85 -> 26 tok/s over two minutes of back-to-back requests), so running one
variant for 4 minutes and then the next compares a cool machine with a hot
one. And this llama-server build can't switch drafting on/off per request,
so variants need a server restart.

So the run is split into short blocks:

    for each pass:
      for each block of `block_size` prompts:
        for each variant (order rotated every block):
          restart llama-server in that mode -> warm-up request
          -> cool-down check: a short probe must reach 90% of this variant's
             best probe speed so far (else wait and probe again)
          -> run the block's prompts, with a pause between requests

Every variant answers every prompt once per pass, and the same prompts run
close together in time across variants. Speedup is paired: for each prompt,
baseline TPOT / variant TPOT, then the median of those ratios.

Lesson from the first run (results/ab-aborted-20261004-1436.json): once the
laptop is heat-soaked, 15 s of idle doesn't cool it, and a ~1.5 s probe
still gets full burst speed while longer requests slow 2-3x. Hence the
defaults: 2 min idle per block, 8 s pauses, 4-prompt blocks, a tiny warm-up,
and each block's first-vs-last TPOT drift saved, so heat shows in the data.
"""
from __future__ import annotations

import json
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

from rag.bench_llm import PROMPTS, file_hash, load_prompts, summarize, same_output
from rag.config import CONFIG, PROJECT_ROOT

SCRIPT = PROJECT_ROOT / "scripts" / "llama-server.sh"
SERVER_LOG = Path("/tmp/llama-server-ab.log")
PROBE = [{"role": "user", "content": "Count from 1 to 60, separated by commas. Output only the numbers."}]


# Short names for models in variant strings (Ollama names; see llama-server.sh).
TARGETS = {"q4km": "llama3.1:8b", "q40": "llama3.1:8b-instruct-q4_0",
           "q8": "llama3.1:8b-instruct-q8_0", "f16": "llama3.1:8b-instruct-fp16"}
DRAFTS = {"1b": "llama3.2:1b", "1bq4": "llama3.2:1b-instruct-q4_K_M"}


def script_args(variant: str) -> tuple[list[str], dict[str, str]]:
    """Variant string -> (llama-server.sh arguments, environment).

        baseline | draft:<k>[:<p_min>] | ngram:<m>:<n>
        optionally followed by @t=<target>,d=<draft> (short names above or Ollama names)
    """
    mode, _, models = variant.partition("@")
    env = {}
    for kv in filter(None, models.split(",")):
        key, _, name = kv.partition("=")
        if key == "t":
            env["TARGET"] = TARGETS.get(name, name)
        elif key == "d":
            env["DRAFT"] = DRAFTS.get(name, name)
        else:
            raise ValueError(f"bad model setting {kv!r} in {variant!r}: use t=<target> or d=<draft>")
    parts = mode.split(":")
    if parts[0] == "baseline" and len(parts) == 1:
        return ["baseline"], env
    if parts[0] == "draft" and len(parts) in (2, 3):
        return ["draft", *parts[1:]], env
    if parts[0] == "ngram" and len(parts) == 3:
        return ["ngram", parts[1], parts[2]], env
    raise ValueError(f"bad variant {variant!r}: use baseline, draft:<k>[:<p_min>] or ngram:<m>:<n>, "
                     "optionally @t=<target>,d=<draft>")


def rotated(variants: list[str], block: int) -> list[str]:
    i = block % len(variants)
    return variants[i:] + variants[:i]


def paired_ratio(rows: list[dict], base_rows: list[dict], key: str) -> float | None:
    """Median over (pass, prompt) pairs of baseline[key] / variant[key]."""
    base = {(r["pass"], r["id"]): r.get(key) for r in base_rows}
    ratios = sorted(base[(r["pass"], r["id"])] / r[key] for r in rows
                    if r.get(key) and base.get((r["pass"], r["id"])))
    return round(ratios[len(ratios) // 2], 2) if ratios else None


# --- server control ----------------------------------------------------------

def _health() -> bool:
    from rag import llamacpp_client
    try:
        return llamacpp_client._http().get("/health", timeout=2).status_code == 200
    except Exception:                             # noqa: BLE001 - still starting
        return False


def stop_server() -> None:
    subprocess.run(["pkill", "-x", "llama-server"], capture_output=True)
    for _ in range(50):
        if subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True).returncode:
            return
        time.sleep(0.2)
    subprocess.run(["pkill", "-9", "-x", "llama-server"], capture_output=True)


def start_server(variant: str, timeout_s: float = 300) -> None:
    import os
    stop_server()
    args, env = script_args(variant)
    with SERVER_LOG.open("a") as log:
        log.write(f"\n===== {datetime.now():%H:%M:%S} start {variant}\n")
        subprocess.Popen([str(SCRIPT), *args], stdout=log, stderr=log, env={**os.environ, **env},
                         cwd=PROJECT_ROOT, start_new_session=True)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _health():
            return
        time.sleep(0.5)
    raise RuntimeError(f"llama-server ({variant}) not healthy after {timeout_s:.0f} s; see {SERVER_LOG}")


def probe(max_tokens: int = 120) -> float:
    """Decode speed of a short, fixed request (little heat; only compared
    within one variant, since drafting changes it). max_tokens=16 doubles as
    the warm-up after a server start (the first request is always slow)."""
    from rag import llamacpp_client
    r = llamacpp_client._http().post("/v1/chat/completions", json={
        "messages": PROBE, "temperature": 0, "max_tokens": max_tokens, "cache_prompt": False})
    return round(r.json()["timings"]["predicted_per_second"], 1)


def cool_down(best: float | None, idle_s: float, ratio: float = 0.9,
              retry_s: float = 20, max_wait_s: float = 240) -> tuple[float, float]:
    """Idle, then probe until the speed is back near this variant's best.
    Returns (probe tok/s, seconds waited)."""
    t0 = time.time()
    time.sleep(idle_s)
    speed = probe()
    while best and speed < ratio * best and time.time() - t0 < max_wait_s:
        time.sleep(retry_s)
        speed = probe()
    return speed, round(time.time() - t0, 1)


# --- the run -----------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def block_drift(rows: list[dict]) -> list[float]:
    """Per block: last request's TPOT / first request's TPOT (1.0 = no slowdown)."""
    blocks: dict[tuple, list[dict]] = {}
    for r in rows:
        blocks.setdefault((r["pass"], r["block"]), []).append(r)
    return sorted(round(b[-1]["tpot_ms"] / b[0]["tpot_ms"], 2) for b in blocks.values()
                  if len(b) > 1 and b[0].get("tpot_ms") and b[-1].get("tpot_ms"))


def _save(result: dict, name: str) -> Path:
    out = CONFIG.results_dir / f"{name}-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(result, indent=2))
    return out


def run_ab(variants: list[str], passes: int = 2, block_size: int = 4,
           pause_s: float = 8.0, idle_s: float = 120.0, prompts_path: Path = PROMPTS,
           limit: int | None = None) -> dict:
    from rag.bench_llm import _server_info
    from rag.generate import generate
    from rag.trace import config_hash, trace

    for v in variants:
        script_args(v)                            # fail fast on a typo
    if CONFIG.llm_backend != "llama_server":
        raise RuntimeError("run with RAG_LLM_BACKEND=llama_server")

    prompts = load_prompts(prompts_path)[:limit]
    blocks = [prompts[i:i + block_size] for i in range(0, len(prompts), block_size)]
    snapshot = CONFIG.snapshot()
    rows: list[dict] = []
    probes: list[dict] = []
    best: dict[str, float] = {}
    servers: dict[str, dict] = {}                 # command, build, memory per variant
    started = time.time()

    def on_sigterm(*_):                           # `kill` saves partial rows like Ctrl-C does
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_sigterm)
    settings = {"variants": variants, "passes": passes, "block_size": block_size,
                "pause_s": pause_s, "idle_s": idle_s, "prompts": prompts_path.name,
                "prompts_hash": file_hash(prompts_path), "n_prompts": len(prompts)}
    try:
        for p in range(passes):
            for b, block in enumerate(blocks):
                for v in rotated(variants, p * len(blocks) + b):
                    start_server(v)
                    probe(max_tokens=16)                      # warm-up: first request after start is slow
                    servers.setdefault(v, _server_info())
                    speed, waited = cool_down(best.get(v), idle_s)
                    best[v] = max(best.get(v, 0), speed)
                    probes.append({"pass": p, "block": b, "variant": v,
                                   "probe_tok_s": speed, "waited_s": waited})
                    _log(f"pass {p} block {b} {v:<12} probe {speed:5.1f} tok/s "
                         f"(waited {waited:.0f} s)")
                    for q in block:
                        with trace("bench", config=snapshot) as t:
                            t.set("label", f"ab:{v}")
                            t.set("pass", p)
                            t.set("question_id", q["id"])
                            text, m = generate(q["messages"])
                        rows.append({"variant": v, "pass": p, "block": b, "id": q["id"],
                                     "probe_tok_s": speed, "text": text, **m})
                        time.sleep(pause_s)
                    done = [r for r in rows if r["variant"] == v and r["block"] == b and r["pass"] == p]
                    tpots = " ".join(f"{r['tpot_ms']:.1f}" for r in done)
                    _log(f"    TPOT per request: {tpots} ms")
    except BaseException:
        out = _save({**settings, "partial": True, "servers": servers, "probes": probes, "rows": rows},
                    "ab-partial")
        _log(f"stopped early; {len(rows)} rows saved to {out}")
        raise
    finally:
        stop_server()

    summaries = {}
    base_rows = [r for r in rows if r["variant"] == variants[0]]
    for v in variants:
        vr = [r for r in rows if r["variant"] == v]
        s = summarize(vr)
        s["probe_tok_s"] = sorted(x["probe_tok_s"] for x in probes if x["variant"] == v)
        s["block_drift"] = block_drift(vr)
        if v != variants[0]:
            s["baseline"] = variants[0]
            s["same_as_baseline"] = same_output(vr, base_rows)
            s["speedup_tpot"] = paired_ratio(vr, base_rows, "tpot_ms")
            s["speedup_answer"] = paired_ratio(vr, base_rows, "e2e_ms")
        summaries[v] = s

    result = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "minutes": round((time.time() - started) / 60, 1), **settings,
        "config_hash": config_hash(snapshot),
        "servers": servers, "summaries": summaries, "probes": probes, "rows": rows,
    }
    result["path"] = str(_save(result, "ab"))
    return result


def as_bench_result(ab: dict, variant: str) -> dict:
    """One variant of an A/B run, shaped like a bench_llm result, so
    record_inference() can write its row."""
    return {"label": f"{variant} (A/B)", "path": ab["path"],
            "server": ab["servers"][variant], "summary": ab["summaries"][variant]}
