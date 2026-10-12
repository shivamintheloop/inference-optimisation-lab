"""LLM-only benchmark on a frozen set of prompts (inference experiments).

export_prompts(): run retrieval + context once for every answerable eval
question and save the exact messages the LLM would get. Every variant then
sees identical input, and the benchmark process never loads torch or the
re-ranker, so nothing else competes for the GPU (PLAN.md principle 8).

run_bench(): one warm-up, then every prompt in order, one at a time, for N
passes, through the engine picked by RAG_LLM_BACKEND. Saves every answer and
its timings to results/llm-<label>-<time>.json; record_inference() appends a
row to results/INFERENCE.md.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rag.config import PROJECT_ROOT

PROMPTS = PROJECT_ROOT / "bench" / "prompts-v1.jsonl"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def export_prompts(out: Path = PROMPTS) -> tuple[Path, int, list[str]]:
    """Returns (path, prompts written, ids skipped because retrieval said 'not found')."""
    from rag import docs_db
    from rag.context import build_context
    from rag.evaluate import load_questions
    from rag.generate import build_messages
    from rag.rerank import is_not_found
    from rag.retrieve import search

    conn = docs_db.connect()
    rows, skipped = [], []
    for q in load_questions():
        if q["style"] == "unanswerable":
            continue                              # no LLM call for these in the pipeline
        hits = search(conn, q["question"], "rerank")
        if is_not_found(hits):
            skipped.append(q["id"])
            continue
        sources, context = build_context(hits)
        rows.append({"id": q["id"], "style": q["style"], "question": q["question"],
                     "sources": [s.doc_id for s in sources],
                     "messages": build_messages(q["question"], context)})
    conn.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return out, len(rows), skipped


EXTRACT_PROMPTS = PROJECT_ROOT / "bench" / "prompts-extract-v1.jsonl"
EXTRACT_SYSTEM = """You are a study assistant. Answer the question by quoting the sentences from the \
numbered sources that answer it, word for word.

Rules:
- Copy each relevant sentence exactly as written. Do not reword, shorten or summarize.
- Put each quote on its own line starting with "- ", followed by its source tag, like [S1].
- Quote at most 6 sentences, most relevant first.
- If no sentence answers the question, reply exactly: "I couldn't find this in your notes.\""""


def export_extract_prompts(src: Path = PROMPTS, out: Path = EXTRACT_PROMPTS) -> tuple[Path, int]:
    """Same questions and sources as prompts-v1, but the system prompt asks for
    word-for-word quotes (an "show me the evidence" RAG mode). Answers that copy
    the prompt are the case where drafts should be accepted far more often."""
    rows = load_prompts(src)
    for r in rows:
        r["messages"] = [{"role": "system", "content": EXTRACT_SYSTEM}, *r["messages"][1:]]
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return out, len(rows)


def load_prompts(path: Path = PROMPTS) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- benchmark -------------------------------------------------------------

INFERENCE_LOG = PROJECT_ROOT / "results" / "INFERENCE.md"


def _p50(values: list) -> float | None:
    values = sorted(v for v in values if v is not None)
    return values[len(values) // 2] if values else None


def summarize(rows: list[dict]) -> dict:
    """One number per metric: the median of each pass's p50, plus the min-max
    of those per-pass p50s (how much the run-to-run noise was)."""
    passes = sorted({r["pass"] for r in rows})
    out: dict = {"passes": len(passes), "prompts": len(rows) // max(len(passes), 1)}
    for key in ("ttft_ms", "tpot_ms", "decode_tok_s", "e2e_ms", "output_tokens"):
        per_pass = [_p50([r.get(key) for r in rows if r["pass"] == p]) for p in passes]
        per_pass = [v for v in per_pass if v is not None]
        if per_pass:
            out[key] = _p50(per_pass)
            out[key + "_range"] = [min(per_pass), max(per_pass)]
    drafted = sum(r.get("draft_tokens", 0) for r in rows)
    if drafted:                                   # pooled over every request
        accepted = sum(r.get("draft_accepted", 0) for r in rows)
        steps = sum(r.get("draft_steps", 0) for r in rows)
        out["acceptance_rate"] = round(accepted / drafted, 3)
        out["tokens_per_step"] = round((accepted + steps) / steps, 2) if steps else None
    first = {r["id"]: r["text"] for r in rows if r["pass"] == passes[0]}
    out["stable_across_passes"] = sum(
        all(r["text"] == first[r["id"]] for r in rows if r["id"] == i) for i in first)
    return out


def same_output(rows: list[dict], baseline_rows: list[dict]) -> int:
    """How many prompts got exactly the baseline's answer (first pass of each).
    At temperature 0 speculative decoding should not change a single token."""
    first = min(r["pass"] for r in rows)
    base = {r["id"]: r["text"] for r in baseline_rows if r["pass"] == min(b["pass"] for b in baseline_rows)}
    return sum(r["text"] == base.get(r["id"]) for r in rows if r["pass"] == first)


def _server_info() -> dict:
    """How the server was started, its build and its memory use (llama-server only)."""
    import subprocess

    from rag.config import CONFIG
    if CONFIG.llm_backend != "llama_server":
        return {"engine": CONFIG.llm_backend}
    from rag import llamacpp_client
    info: dict = {"engine": "llama_server"}
    cmd = PROJECT_ROOT / "data" / "llama-server.cmd"
    info["cmd"] = cmd.read_text().strip() if cmd.exists() else None
    try:
        info["build"] = llamacpp_client._http().get("/props").json().get("build_info")
    except Exception:                             # noqa: BLE001 - info only
        info["build"] = None
    pid = subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True, text=True).stdout.split()
    if pid:
        rss = subprocess.run(["ps", "-o", "rss=", "-p", pid[0]], capture_output=True, text=True).stdout
        info["rss_mb"] = round(int(rss.strip()) / 1024) if rss.strip() else None
    return info


def run_bench(label: str, passes: int = 3, baseline: Path | None = None) -> dict:
    from datetime import datetime

    from rag.config import CONFIG
    from rag.generate import generate
    from rag.trace import config_hash, trace

    prompts = load_prompts()
    snapshot = CONFIG.snapshot()
    generate(prompts[0]["messages"])              # warm-up, not counted
    rows: list[dict] = []
    for p in range(passes):
        for q in prompts:
            with trace("bench", config=snapshot) as t:
                t.set("label", label)
                t.set("pass", p)
                t.set("question_id", q["id"])
                text, m = generate(q["messages"])
            rows.append({"pass": p, "id": q["id"], "text": text, **m})

    summary = summarize(rows)
    result = {
        "label": label,
        "time": datetime.now().isoformat(timespec="seconds"),
        "server": _server_info(),
        "config_hash": config_hash(snapshot),
        "prompts_hash": file_hash(PROMPTS),
        "summary": summary,
        "rows": rows,
    }
    if baseline:
        base = json.loads(Path(baseline).read_text())
        summary["baseline"] = Path(baseline).name
        summary["same_as_baseline"] = same_output(rows, base["rows"])
        b = base["summary"]
        if b.get("tpot_ms") and summary.get("tpot_ms"):
            summary["speedup_tpot"] = round(b["tpot_ms"] / summary["tpot_ms"], 2)
        if b.get("e2e_ms") and summary.get("e2e_ms"):
            summary["speedup_answer"] = round(b["e2e_ms"] / summary["e2e_ms"], 2)

    slug = "".join(c if c.isalnum() else "-" for c in label.lower()).strip("-")
    out = CONFIG.results_dir / f"llm-{slug}-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    result["path"] = str(out)
    return result


def _fmt(v, spec: str = ".0f", suffix: str = "") -> str:
    return "–" if v is None else f"{v:{spec}}{suffix}"


def inference_row(res: dict, note: str = "") -> str:
    s, srv = res["summary"], res["server"]
    k = "–"
    cmd = srv.get("cmd") or ""
    for flag in ("--spec-draft-n-max", "--spec-ngram-simple-size-m"):
        if flag in cmd:
            k = cmd.split(flag)[1].split()[0]
    lo, hi = s.get("tpot_ms_range", [None, None])
    same = f"{s['same_as_baseline']}/{s['prompts']}" if "same_as_baseline" in s else "–"
    return ("| " + " | ".join([
        res["label"], f"{srv.get('engine')} {srv.get('build') or ''}".strip(), k,
        _fmt(s.get("ttft_ms"), ".0f", " ms"),
        f"{_fmt(s.get('tpot_ms'), '.1f')} ms ({_fmt(lo, '.1f')}–{_fmt(hi, '.1f')})",
        _fmt(s.get("decode_tok_s"), ".0f"),
        _fmt(s.get("speedup_tpot"), ".2f", "×"),
        _fmt(s.get("e2e_ms") and s["e2e_ms"] / 1000, ".2f", " s"),
        _fmt(s.get("acceptance_rate"), ".2f"), _fmt(s.get("tokens_per_step"), ".2f"),
        same, f"{s['stable_across_passes']}/{s['prompts']}",
        _fmt(srv.get("rss_mb"), ".0f", " MB"),
        f"`{Path(res['path']).name}`", note,
    ]) + " |\n")


def record_inference(res: dict, note: str = "", log: Path = INFERENCE_LOG) -> Path:
    """Append one row to results/INFERENCE.md (created with a legend if new)."""
    if not log.exists():
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            "# Inference experiments\n\n"
            "Appended by `python -m rag bench-llm --record`; never edited by hand. "
            "Every row runs the same frozen prompts (`bench/prompts-v1.jsonl`, 24 answerable eval "
            "questions) one at a time, 3 passes, temperature 0, prefix cache off.\n\n"
            "- **TTFT / TPOT**: time to first token / time per output token; median of the per-pass "
            "p50s, with the min–max of those p50s (run-to-run noise)\n"
            "- **Decode tok/s**: the engine's own decode speed\n"
            "- **Speedup**: baseline TPOT ÷ this TPOT\n"
            "- **Answer time**: p50 time for the whole LLM answer\n"
            "- **Acceptance**: share of drafted tokens the big model kept\n"
            "- **Tokens/check**: tokens produced per big-model pass (kept guesses + 1 of its own)\n"
            "- **Same as baseline**: answers identical to the baseline run (should be all, "
            "at temperature 0)\n"
            "- **Stable**: answers identical across this run's own passes\n"
            "- **Server RSS**: llama-server memory\n\n"
            "| Variant | Engine | k | TTFT p50 | TPOT p50 (range) | Decode tok/s | Speedup | "
            "Answer time p50 | Acceptance | Tokens/check | Same as baseline | Stable | "
            "Server RSS | Raw | Note |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    with log.open("a") as f:
        f.write(inference_row(res, note))
    return log
