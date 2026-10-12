"""Re-rank: read the question and each candidate chunk together.

Vector search compares two vectors made separately (question alone, chunk
alone). A cross-encoder reads "question + chunk" as one input and outputs a
relevance score, so it can see that a formula card about `1/(k + rank)` is
the answer to "What is RRF?". It is too slow to run on every chunk, so it
only re-orders the top `rerank_candidates` from hybrid search.

The top score also drives the "not found" check: if even the best chunk
scores below `not_found_threshold`, the notes probably don't answer the
question, and we say so instead of letting the LLM guess.
"""
from __future__ import annotations

from functools import lru_cache

from rag.config import CONFIG
from rag.retrieve import Hit, _summary
from rag.trace import span


def _device() -> str:
    if CONFIG.rerank_device != "auto":
        return CONFIG.rerank_device
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()           # keep "Loading weights" bars out of answers

    try:                                        # cached copy: no network round trip
        return CrossEncoder(CONFIG.rerank_model, device=_device(), local_files_only=True)
    except Exception:                           # first run: download from Hugging Face
        return CrossEncoder(CONFIG.rerank_model, device=_device())


def pair_text(hit: Hit) -> str:
    """What the cross-encoder reads for a chunk: its own heading + its text."""
    p = hit.payload or {}
    heading = p.get("section", "").split(" > ")[-1]
    return f"{heading}\n{p.get('text', '')}"


def rerank(query: str, hits: list[Hit], top_n: int = CONFIG.rerank_top_n,
           candidates: int = CONFIG.rerank_candidates) -> list[Hit]:
    pool = [h for h in hits[:candidates] if h.payload]
    with span("rerank", model=CONFIG.rerank_model, candidates=len(pool)) as s:
        if not pool:
            s.set("top_score", None)
            return []
        if _model.cache_info().currsize == 0:
            with span("rerank.load_model", device=_device()):
                _model()
        scores = _model().predict([(query, pair_text(h)) for h in pool], batch_size=32)
        ranked = []
        for prev_rank, (h, score) in enumerate(zip(pool, scores), 1):
            ranks = {**h.ranks, "hybrid": prev_rank}
            ranked.append(Hit(h.chunk_id, float(score), ranks, h.payload))
        ranked.sort(key=lambda h: h.score, reverse=True)
        s.set("top", _summary(ranked[:top_n]))
        s.set("top_score", round(ranked[0].score, 4))
    return ranked[:top_n]


def is_not_found(hits: list[Hit], threshold: float = CONFIG.not_found_threshold) -> bool:
    return not hits or hits[0].score < threshold
