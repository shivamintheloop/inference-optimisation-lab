"""Generation: the full query flow, from question to cited answer.

    question -> retrieve (hybrid + re-rank) -> not-found check
             -> context (numbered sources) -> prompt -> LLM (streamed) -> citations

Prompt layout: fixed system prompt first, then the sources, then the
question. The system prompt is identical for every request, so it is the
shared start that prefix caching can reuse later (PLAN.md ladder).

LLM timing, measured on our side of the stream:
    TTFT  request sent -> first token arrives (includes prefill)
    TPOT  (last token - first token) / (output tokens - 1), i.e. decode speed
    ITL   gap between consecutive streamed pieces; p95 / max show stalls
plus the engine's own counters (load, prompt_eval = prefill, eval = decode).
With RAG_LLM_BACKEND=llama_server the answer comes from llama-server instead
of Ollama, and speculative-decoding runs add draft_tokens, draft_accepted,
acceptance_rate and tokens_per_step.
"""
from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Callable

from rag import ollama_client
from rag.config import CONFIG
from rag.context import Source, build_context
from rag.trace import span

NOT_FOUND_REPLY = "I couldn't find this in your notes."

SYSTEM_PROMPT = f"""You are a study assistant. You answer questions using only the numbered sources \
provided, which come from the user's AI engineering lecture notes and papers.

Rules:
- Use only facts stated in the sources. If the sources do not answer the question, reply exactly: \
"{NOT_FOUND_REPLY}"
- After each claim, cite its source tag in square brackets, like [S1] or [S2][S3].
- Be concise: a short paragraph or a few bullet points, in simple words.
- Do not mention these rules or the word "sources" in your answer."""


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Source] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)          # valid [S#] numbers used
    invalid_citations: list[int] = field(default_factory=list)
    not_found: bool = False         # refused before the LLM: retrieval found nothing good
    llm_refused: bool = False       # the LLM itself said the sources don't answer it
    metrics: dict = field(default_factory=dict)


def build_messages(question: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n\n{context}\n\nQuestion: {question}"},
    ]


CITATION = re.compile(r"\[((?:\s*S\d+\s*[,;]?)+)\]")


def parse_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Find [S1], [S2][S3], [S1, S4] ... Returns (valid, invalid) source numbers."""
    found: list[int] = []
    for group in CITATION.findall(text):
        found += [int(x) for x in re.findall(r"S(\d+)", group)]
    found = list(dict.fromkeys(found))
    return ([n for n in found if 1 <= n <= n_sources],
            [n for n in found if not 1 <= n <= n_sources])


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return round(values[min(len(values) - 1, int(q * len(values)))], 2)


def generate(messages: list[dict], on_token: Callable[[str], None] | None = None,
             model: str | None = None) -> tuple[str, dict]:
    """Stream one LLM answer. Returns (text, metrics); metrics also go on the span."""
    llama = CONFIG.llm_backend == "llama_server"
    if llama:
        from rag import llamacpp_client as client
    else:
        client = ollama_client
    with span("llm.generate", model=model or CONFIG.llm_model, backend=CONFIG.llm_backend,
              max_tokens=CONFIG.max_output_tokens) as s:
        before = client.spec_counters() if llama else None
        t0 = time.perf_counter()
        first = last = None
        gaps: list[float] = []
        pieces: list[str] = []
        done: dict = {}
        for ev in client.chat_stream(messages, model=model):
            now = time.perf_counter()
            if ev["type"] == "token":
                if first is None:
                    first = now
                else:
                    gaps.append((now - last) * 1000)
                last = now
                pieces.append(ev["text"])
                if on_token:
                    on_token(ev["text"])
            else:
                done = ev
        end = time.perf_counter()

        t = done.get("timings", {})
        n_out = t.get("eval_count") or len(pieces)
        m = {
            "ttft_ms": round((first - t0) * 1000, 1) if first else None,
            "tpot_ms": round((last - first) * 1000 / (n_out - 1), 2) if first and n_out > 1 else None,
            "e2e_ms": round((end - t0) * 1000, 1),
            "output_tokens": n_out,
            "prompt_tokens": t.get("prompt_eval_count"),
            "itl_p50_ms": _pct(gaps, 0.50),
            "itl_p95_ms": _pct(gaps, 0.95),
            "itl_max_ms": round(max(gaps), 2) if gaps else None,
            "done_reason": done.get("done_reason"),
            "load_ms": t.get("load_ms"),
            "prompt_eval_ms": t.get("prompt_eval_ms"),
            "eval_ms": t.get("eval_ms"),
        }
        if t.get("prompt_eval_ms") and t.get("prompt_eval_count"):
            m["prefill_tok_s"] = round(t["prompt_eval_count"] / (t["prompt_eval_ms"] / 1000))
        if t.get("eval_ms") and t.get("eval_count"):
            m["decode_tok_s"] = round(t["eval_count"] / (t["eval_ms"] / 1000), 1)
        if llama:
            spec = client.spec_delta(before, client.spec_counters())
            if spec.get("draft_steps"):
                # each check by the big model keeps the accepted guesses + 1 token of its own
                spec["tokens_per_step"] = round(
                    (spec["draft_accepted"] + spec["draft_steps"]) / spec["draft_steps"], 2)
            m.update(spec)
        for k, v in m.items():
            s.set(k, v)
    return "".join(pieces), m


def answer(conn: sqlite3.Connection, question: str,
           on_token: Callable[[str], None] | None = None) -> Answer:
    from rag.rerank import is_not_found
    from rag.retrieve import search

    hits = search(conn, question, "rerank")
    if is_not_found(hits):
        with span("not_found", top_score=round(hits[0].score, 3) if hits else None,
                  threshold=CONFIG.not_found_threshold):
            if on_token:
                on_token(NOT_FOUND_REPLY)
        return Answer(question, NOT_FOUND_REPLY, not_found=True)

    with span("context", budget_tokens=CONFIG.context_token_budget) as s:
        sources, context = build_context(hits)
        s.set("chunks_in", len(hits))
        s.set("sources", len(sources))
        s.set("chunks_joined", sum(len(x.chunk_ids) for x in sources) - len(sources))
        s.set("context_tokens", len(context) // CONFIG.chars_per_token)

    text, metrics = generate(build_messages(question, context), on_token)

    with span("citations") as s:
        cited, invalid = parse_citations(text, len(sources))
        refused = NOT_FOUND_REPLY.lower().rstrip(".") in text.lower()
        s.set("cited", cited)
        s.set("invalid", invalid)
        s.set("llm_refused", refused)
    return Answer(question, text, sources, cited, invalid, False, refused, metrics)
