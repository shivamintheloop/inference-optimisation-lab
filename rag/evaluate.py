"""Retrieval eval: does the right file come back near the top?

eval/questions.jsonl holds questions with the file(s) that answer them.
Each question runs through every retrieval mode, and we score the ranked
chunks (the same list the LLM's context will be built from):

    hit@1   the top chunk comes from an expected file
    hit@5   at least one of the top 5 chunks does
    MRR@10  1 / rank of the first good chunk in the top 10 (0 if none);
            rewards putting the right chunk higher, not just "somewhere"

Unanswerable questions have no expected file. In rerank mode we compare the
top re-rank score of answerable vs unanswerable questions and pick the
"not found" threshold that separates them best (threshold_analysis).
Every question x mode is one trace (kind "eval"), and the summary is saved
to results/eval-<time>.json together with the config hash.

run_answer_eval runs the full pipeline (retrieve -> LLM) once per question
and scores refusals, citations and LLM timing (results/answers-<time>.json).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from rag import docs_db
from rag.config import CONFIG, PROJECT_ROOT
from rag.retrieve import MODES, search
from rag.trace import config_hash, trace

QUESTIONS = PROJECT_ROOT / "eval" / "questions.jsonl"
RESULTS_LOG = CONFIG.results_dir / "RESULTS.md"


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def questions_hash(path: Path = QUESTIONS) -> str:
    """Changes whenever a question or its expected files change, so results
    scored against different labels are never compared by mistake."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def first_good_rank(doc_ids: list[str], expected: list[str], depth: int = 10) -> int | None:
    for rank, doc_id in enumerate(doc_ids[:depth], 1):
        if doc_id in expected:
            return rank
    return None


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    ranks = [r["rank"] for r in rows]
    return {
        "n": n,
        "hit@1": round(sum(1 for x in ranks if x == 1) / n, 3),
        "hit@5": round(sum(1 for x in ranks if x and x <= 5) / n, 3),
        "mrr@10": round(sum(1 / x for x in ranks if x) / n, 3),
    }


def threshold_analysis(rows: list[dict]) -> dict:
    """Pick the not-found threshold from re-rank top scores.

    A threshold t refuses a question when its top score < t. We want to keep
    answerable questions and refuse unanswerable ones; the best t gets the
    most of both right. On a tie we take the lowest t: wrongly refusing a
    real question is worse than one extra 'I'm not sure' answer."""
    ans = sorted(r["top_score"] for r in rows if r["style"] != "unanswerable")
    una = sorted(r["top_score"] for r in rows if r["style"] == "unanswerable")
    if not ans or not una:
        return {}
    cuts = sorted(set(ans + una))
    candidates = [cuts[0] - 0.01] + [(a + b) / 2 for a, b in zip(cuts, cuts[1:])]
    best_t, best_correct = None, -1
    for t in candidates:
        correct = sum(s >= t for s in ans) + sum(s < t for s in una)
        if correct > best_correct:
            best_t, best_correct = t, correct
    return {
        "answerable_min": round(ans[0], 3), "answerable_p10": round(ans[len(ans) // 10], 3),
        "unanswerable_max": round(una[-1], 3),
        "separable": una[-1] < ans[0],
        "suggested_threshold": round(best_t, 2),
        "kept_answerable": sum(s >= best_t for s in ans), "n_answerable": len(ans),
        "refused_unanswerable": sum(s < best_t for s in una), "n_unanswerable": len(una),
        "at_current": {
            "threshold": CONFIG.not_found_threshold,
            "kept_answerable": sum(s >= CONFIG.not_found_threshold for s in ans),
            "refused_unanswerable": sum(s < CONFIG.not_found_threshold for s in una),
        },
    }


def run_eval(modes: tuple[str, ...] = MODES) -> dict:
    questions = load_questions()
    conn = docs_db.connect()
    per_question: list[dict] = []
    snapshot = CONFIG.snapshot()

    for q in questions:
        for mode in modes:
            with trace("eval", config=snapshot) as t:
                t.set("question_id", q["id"])
                t.set("mode", mode)
                t.set("question", q["question"])
                hits = search(conn, q["question"], mode)
                doc_ids = [h.doc_id for h in hits]
                rank = first_good_rank(doc_ids, q["expected"]) if q["expected"] else None
                t.set("rank", rank)
            per_question.append({
                "id": q["id"], "style": q["style"], "mode": mode, "rank": rank,
                "top_doc": doc_ids[0] if doc_ids else None,
                "top_score": round(hits[0].score, 4) if hits else None,
                "ms": round(t.duration_ms, 1),
            })
    conn.close()

    summary: dict = {}
    for mode in modes:
        rows = [r for r in per_question if r["mode"] == mode and r["style"] != "unanswerable"]
        summary[mode] = {"all": summarize(rows)}
        for style in ("keyword", "paraphrase"):
            summary[mode][style] = summarize([r for r in rows if r["style"] == style])
        ms = sorted(r["ms"] for r in per_question if r["mode"] == mode)
        summary[mode]["p50_ms"] = ms[len(ms) // 2]

    if "rerank" in modes:
        summary["rerank"]["threshold"] = threshold_analysis(
            [r for r in per_question if r["mode"] == "rerank"])

    result = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "config_hash": config_hash(snapshot),
        "questions_hash": questions_hash(),
        "summary": summary,
        "per_question": per_question,
    }
    CONFIG.results_dir.mkdir(parents=True, exist_ok=True)
    out = CONFIG.results_dir / f"eval-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(result, indent=2))
    result["path"] = str(out)
    return result


def _p(values: list[float], q: float) -> float | None:
    values = sorted(v for v in values if v is not None)
    return values[min(len(values) - 1, int(q * len(values)))] if values else None


def run_answer_eval() -> dict:
    """Full pipeline per question (retrieve -> LLM). Scores what the retrieval
    eval cannot see: refusals, citations, and LLM timing.

        refused right   unanswerable question got "not found" (threshold or LLM)
        refused wrong   answerable question got "not found"
        cited           answer has at least one valid [S#]
        cites expected  a cited source is one of the expected files
        invalid         [S#] tags pointing to sources that don't exist
        cut off         answer hit max_output_tokens (done_reason "length")
    """
    from rag.generate import answer, generate

    questions = load_questions()
    conn = docs_db.connect()
    snapshot = CONFIG.snapshot()
    generate([{"role": "user", "content": "Say OK."}])  # warm-up: load the model, not timed

    rows: list[dict] = []
    for q in questions:
        with trace("eval_answer", config=snapshot) as t:
            t.set("question_id", q["id"])
            t.set("question", q["question"])
            a = answer(conn, q["question"])
            t.set("answer", a.text)
        cited_docs = [a.sources[n - 1].doc_id for n in a.cited]
        m = a.metrics
        rows.append({
            "id": q["id"], "style": q["style"], "answer": a.text,
            "not_found": a.not_found, "llm_refused": a.llm_refused,
            "cited": a.cited, "invalid": a.invalid_citations, "cited_docs": cited_docs,
            "cites_expected": any(d in q["expected"] for d in cited_docs) if q["expected"] else None,
            "ms": round(t.duration_ms, 1),
            **{k: m.get(k) for k in ("ttft_ms", "tpot_ms", "output_tokens", "prompt_tokens",
                                     "load_ms", "done_reason")},
        })
    conn.close()

    ans = [r for r in rows if r["style"] != "unanswerable"]
    una = [r for r in rows if r["style"] == "unanswerable"]
    refused = lambda r: r["not_found"] or r["llm_refused"]  # noqa: E731
    llm = [r for r in rows if not r["not_found"]]            # rows that called the LLM
    summary = {
        "n_answerable": len(ans), "n_unanswerable": len(una),
        "refused_right": sum(refused(r) for r in una),
        "refused_wrong": sum(refused(r) for r in ans),
        "cited": round(sum(bool(r["cited"]) for r in ans if not refused(r)) / len(ans), 3),
        "cites_expected": round(sum(bool(r["cites_expected"]) for r in ans) / len(ans), 3),
        "invalid_citations": sum(len(r["invalid"]) for r in rows),
        "cut_off": sum(r["done_reason"] == "length" for r in rows),
        "llm_calls": len(llm),
        "ttft_p50_ms": _p([r["ttft_ms"] for r in llm], 0.5),
        "ttft_p95_ms": _p([r["ttft_ms"] for r in llm], 0.95),
        "tpot_p50_ms": _p([r["tpot_ms"] for r in llm], 0.5),
        "prompt_tokens_p50": _p([r["prompt_tokens"] for r in llm], 0.5),
        "output_tokens_p50": _p([r["output_tokens"] for r in llm], 0.5),
        "e2e_p50_ms": _p([r["ms"] for r in llm], 0.5),
        "e2e_p95_ms": _p([r["ms"] for r in llm], 0.95),
    }
    result = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "config_hash": config_hash(snapshot),
        "questions_hash": questions_hash(),
        "summary": summary,
        "per_question": rows,
    }
    CONFIG.results_dir.mkdir(parents=True, exist_ok=True)
    out = CONFIG.results_dir / f"answers-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(result, indent=2))
    result["path"] = str(out)
    return result


def answers_table(gen: dict) -> list[str]:
    g = gen["summary"]
    return [
        "",
        f"Answers (full pipeline, `{CONFIG.llm_model}`) · raw: `{Path(gen['path']).name}`",
        "",
        "| Refused right | Refused wrong | Cited | Cites expected file | Invalid cites | Cut off "
        "| TTFT p50 / p95 | TPOT p50 | Prompt tokens p50 | Output tokens p50 | Time per question p50 / p95 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        f"| {g['refused_right']}/{g['n_unanswerable']} | {g['refused_wrong']}/{g['n_answerable']} "
        f"| {g['cited']:.2f} | {g['cites_expected']:.2f} | {g['invalid_citations']} | {g['cut_off']} "
        f"| {g['ttft_p50_ms']:.0f} / {g['ttft_p95_ms']:.0f} ms | {g['tpot_p50_ms']:.1f} ms "
        f"| {g['prompt_tokens_p50']:.0f} | {g['output_tokens_p50']:.0f} "
        f"| {g['e2e_p50_ms'] / 1000:.1f} / {g['e2e_p95_ms'] / 1000:.1f} s |",
    ]


def results_table(result: dict, label: str, note: str = "") -> str:
    """One RESULTS.md entry: the phase label, run identity, and the mode table."""
    s = result["summary"]
    src = Path(result["path"]).name if result.get("path") else "?"
    lines = [
        f"## {label}",
        "",
        f"Run {result['time']} · config `{result['config_hash']}` · "
        f"questions `{result.get('questions_hash', 'n/a')}` · raw: `{src}`",
        "",
        "| Mode | hit@1 | hit@5 | MRR@10 | Jargon hit@5 | Everyday-words hit@5 | Time per question |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode, m in s.items():
        a, k, p = m["all"], m["keyword"], m["paraphrase"]
        lines.append(f"| {mode} | {a['hit@1']:.2f} | {a['hit@5']:.2f} | {a['mrr@10']:.3f} | "
                     f"{k['hit@5']:.2f} | {p['hit@5']:.2f} | {m['p50_ms']:.0f} ms |")
    th = s.get("rerank", {}).get("threshold")
    if th:
        lines += ["", f"Not found (re-rank top score): answerable ≥ {th['answerable_min']}, "
                      f"unanswerable ≤ {th['unanswerable_max']}; threshold "
                      f"{th['at_current']['threshold']} keeps {th['at_current']['kept_answerable']}/"
                      f"{th['n_answerable']} and refuses {th['at_current']['refused_unanswerable']}/"
                      f"{th['n_unanswerable']}."]
    if result.get("answers"):
        lines += answers_table(result["answers"])
    if note:
        lines += ["", note]
    return "\n".join(lines) + "\n"


def record_results(result: dict, label: str, note: str = "", log: Path = RESULTS_LOG) -> Path:
    """Append one entry to results/RESULTS.md (created with a header if new)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    if not log.exists():
        log.write_text(
            "# Retrieval results by phase\n\n"
            "Appended by `python -m rag eval --record \"<label>\"`; never edited by hand.\n"
            "Compare rows only when the `questions` hash matches: a different hash means "
            "the eval labels changed in between.\n\n"
            "- **hit@1 / hit@5**: share of questions whose expected file is the top chunk / "
            "in the top 5 chunks\n"
            "- **MRR@10**: average of 1 / rank of the first correct chunk (0 if not in the top 10)\n"
            "- **Jargon / Everyday-words**: questions that use the notes' exact terms vs. "
            "the same idea in plain words\n"
            "- **Time per question**: p50 of the whole retrieval step\n"
            "- **Answers table** (from Phase 4, `--answers`): full pipeline per question. "
            "Refused right/wrong = 'not found' on unanswerable/answerable questions; "
            "Cited = answers with a valid [S#]; TTFT = time to first token; "
            "TPOT = time per output token\n\n")
    with log.open("a") as f:
        f.write("\n" + results_table(result, label, note))
    return log
