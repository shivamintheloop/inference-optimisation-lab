"""Context builder: turn the re-ranked chunks into numbered sources for the prompt.

1. Group chunks by document and sort them by position in the document.
2. Join neighbours from the same section whose text overlaps. Chunks are
   exact slices of the document (char_start / char_end), so the 50-token
   overlap is cut out instead of being sent to the LLM twice.
3. Order the sources by their best re-rank position and number them
   [S1], [S2], ... so the LLM can cite them.
4. Stop adding sources when the token budget is used up. A source is never
   cut in half; it is either in or out.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rag.config import CONFIG
from rag.retrieve import Hit


@dataclass
class Source:
    n: int                          # the number in [S<n>]
    doc_id: str
    title: str
    section: str
    page: int | None
    source_path: str
    text: str
    char_start: int
    char_end: int
    best_rank: int                  # best re-rank position among its chunks (1 = top)
    chunk_ids: list[str] = field(default_factory=list)

    @property
    def heading(self) -> str:
        """Section path without the document title (the title is shown separately)."""
        return " > ".join(self.section.split(" > ")[1:]) or self.title

    def label(self) -> str:
        where = f"page {self.page}" if self.page else self.heading
        return f"{self.title} | {where}"


def merge_hits(hits: list[Hit]) -> list[Source]:
    """Join overlapping neighbours; returns sources sorted by best rank, unnumbered."""
    by_doc: dict[str, list[tuple[int, Hit]]] = {}
    for rank, h in enumerate(hits, 1):
        if h.payload:
            by_doc.setdefault(h.payload["doc_id"], []).append((rank, h))

    sources: list[Source] = []
    for doc_hits in by_doc.values():
        doc_hits.sort(key=lambda rh: rh[1].payload["char_start"])
        for rank, h in doc_hits:
            p = h.payload
            last = sources[-1] if sources and sources[-1].doc_id == p["doc_id"] else None
            if last and last.section == p["section"] and p["char_start"] <= last.char_end:
                if p["char_end"] > last.char_end:      # keep only the new part
                    last.text += p["text"][last.char_end - p["char_start"]:]
                    last.char_end = p["char_end"]
                last.best_rank = min(last.best_rank, rank)
                last.chunk_ids.append(h.chunk_id)
            else:
                sources.append(Source(
                    n=0, doc_id=p["doc_id"], title=p["title"], section=p["section"],
                    page=p.get("page"), source_path=p["source_path"], text=p["text"],
                    char_start=p["char_start"], char_end=p["char_end"], best_rank=rank,
                    chunk_ids=[h.chunk_id]))
    sources.sort(key=lambda s: s.best_rank)
    return sources


def format_source(s: Source) -> str:
    return f"[S{s.n}] {s.label()}\n{s.text}"


def build_context(hits: list[Hit], budget_tokens: int = CONFIG.context_token_budget,
                  chars_per_token: int = CONFIG.chars_per_token) -> tuple[list[Source], str]:
    kept: list[Source] = []
    used = 0
    for s in merge_hits(hits):
        s.n = len(kept) + 1
        cost = len(format_source(s)) // chars_per_token
        if kept and used + cost > budget_tokens:
            continue                    # does not fit; a smaller later source still might
        kept.append(s)
        used += cost
    for i, s in enumerate(kept, 1):     # renumber after any skips
        s.n = i
    return kept, "\n\n".join(format_source(s) for s in kept)
