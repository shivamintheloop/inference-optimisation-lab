"""Retrieval: vector search + BM25 keyword search, merged with RRF.

vector  embed the question (with the "search_query: " prefix) and ask Qdrant
        for the nearest chunks by cosine similarity. Good at meaning: it finds
        "how long can I be off work" when the text says "leave policy".
bm25    SQLite FTS5 keyword search. Good at exact words that embeddings blur:
        names, acronyms, codes ("RRF", "HNSW", "Jev").
hybrid  run both, then merge with Reciprocal Rank Fusion:
            score(chunk) = sum over lists of 1 / (rrf_k + rank in that list)
        Only ranks matter, so the two lists' very different score scales
        (cosine 0..1 vs BM25 0..~30) never need to be compared.
rerank  hybrid, then a cross-encoder re-scores the top candidates
        (rag/rerank.py) and keeps the best rerank_top_n.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

from rag import ollama_client, vector_store
from rag.config import CONFIG
from rag.trace import span

MODES = ("vector", "bm25", "hybrid", "rerank")

# Words that match almost every chunk; they add noise to an OR keyword query.
STOPWORDS = set("""
a an and are as at be but by can do does for from how i in is it its me my of on or
should so than that the their them then there these they this to was we what when
where which who why will with would you your about into vs versus
""".split())


@dataclass
class Hit:
    chunk_id: str
    score: float                    # cosine, BM25 or RRF, depending on the mode
    ranks: dict[str, int] = field(default_factory=dict)   # 1-based rank in each list
    payload: dict | None = None     # metadata + text, from Qdrant

    @property
    def doc_id(self) -> str:
        return self.payload["doc_id"] if self.payload else "?"


def _summary(hits: list[Hit]) -> list[list]:
    """Compact form for traces: [chunk_id prefix, score]."""
    return [[h.chunk_id[:8], round(h.score, 4)] for h in hits]


# ---------------------------------------------------------------- vector

def vector_search(query: str, top_k: int = CONFIG.vector_top_k) -> list[Hit]:
    with span("embed.query", model=CONFIG.embed_model) as s:
        vectors, timings = ollama_client.embed([CONFIG.embed_query_prefix + query])
        s.set("load_ms", timings.get("load_ms"))
    with span("retrieve.vector", top_k=top_k) as s:
        points = vector_store.client().query_points(
            CONFIG.qdrant_collection, query=vectors[0], limit=top_k, with_payload=True).points
        hits = [Hit(str(p.id), p.score, {"vector": i}, p.payload)
                for i, p in enumerate(points, 1)]
        s.set("hits", _summary(hits))
    return hits


# ---------------------------------------------------------------- keyword

def fts_query(query: str) -> str | None:
    """Turn a question into an FTS5 query: quoted words joined by OR.
    Quoting stops FTS5 from reading words like AND/NEAR or '-' as operators."""
    words = [w for w in re.findall(r"\w+", query.lower()) if w not in STOPWORDS]
    words = list(dict.fromkeys(words))          # drop repeats, keep order
    return " OR ".join(f'"{w}"' for w in words) or None


def bm25_search(conn: sqlite3.Connection, query: str,
                top_k: int = CONFIG.bm25_top_k) -> list[Hit]:
    with span("retrieve.bm25", top_k=top_k) as s:
        q = fts_query(query)
        s.set("fts_query", q)
        if q is None:
            return []
        # bm25() is lower-is-better (negative); flip it so higher is better.
        rows = conn.execute(
            "SELECT chunk_id, -bm25(chunks_fts) AS score FROM chunks_fts "
            "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?", (q, top_k)).fetchall()
        hits = [Hit(r["chunk_id"], r["score"], {"bm25": i}) for i, r in enumerate(rows, 1)]
        s.set("hits", _summary(hits))
    return hits


# ---------------------------------------------------------------- merge

def rrf_merge(lists: dict[str, list[Hit]], k: int = CONFIG.rrf_k) -> list[Hit]:
    merged: dict[str, Hit] = {}
    for name, hits in lists.items():
        for rank, h in enumerate(hits, 1):
            m = merged.setdefault(h.chunk_id, Hit(h.chunk_id, 0.0, {}, h.payload))
            m.score += 1.0 / (k + rank)
            m.ranks[name] = rank
            m.payload = m.payload or h.payload
    return sorted(merged.values(), key=lambda h: h.score, reverse=True)


def fill_payloads(hits: list[Hit]) -> None:
    """BM25 hits only carry an ID; fetch their metadata + text from Qdrant."""
    missing = [h.chunk_id for h in hits if h.payload is None]
    if not missing:
        return
    with span("retrieve.fetch_payloads", count=len(missing)):
        records = vector_store.client().retrieve(CONFIG.qdrant_collection, ids=missing,
                                                 with_payload=True)
        by_id = {str(r.id): r.payload for r in records}
        for h in hits:
            if h.payload is None:
                h.payload = by_id.get(h.chunk_id)


def search(conn: sqlite3.Connection, query: str, mode: str = "hybrid") -> list[Hit]:
    """Run one retrieval mode and return hits with payloads, best first."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    with span("retrieve", mode=mode) as s:
        if mode == "vector":
            hits = vector_search(query)
        elif mode == "bm25":
            hits = bm25_search(conn, query)
        else:                                   # hybrid and rerank
            v = vector_search(query)
            b = bm25_search(conn, query)
            with span("retrieve.merge", rrf_k=CONFIG.rrf_k) as m:
                hits = rrf_merge({"vector": v, "bm25": b})
                both = {h.chunk_id for h in v} & {h.chunk_id for h in b}
                m.set("merged", len(hits))
                m.set("in_both", len(both))     # how often the two methods agree
                m.set("hits", _summary(hits[:CONFIG.vector_top_k]))
        fill_payloads(hits)
        if mode == "rerank":
            from rag.rerank import rerank       # lazy: loads torch only when needed
            hits = rerank(query, hits)
        s.set("returned", len(hits))
    return hits
