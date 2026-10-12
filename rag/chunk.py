"""Split a parsed document into overlapping chunks.

Sizes are in estimated tokens (chars / chars_per_token). Each section is
chunked on its own, so a chunk never mixes two concept cards or two pages.

A section that fits is one chunk. A longer one is cut at the best boundary
that fits: paragraph break, then line break, then sentence end. The next chunk
starts about `overlap_tokens` earlier, snapped to a boundary, so an idea that
gets cut at an edge still appears whole in one of the two chunks.

Chunk text is always an exact slice of ParsedDoc.text[char_start:char_end].
The context builder uses those offsets later to join neighbouring chunks
without repeating the overlap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rag.config import CONFIG
from rag.parse import ParsedDoc


@dataclass
class Chunk:
    index: int
    char_start: int
    char_end: int
    text: str
    heading_path: list[str]
    page: int | None

    @property
    def section(self) -> str:
        return " > ".join(self.heading_path)

    def embed_text(self) -> str:
        """Heading path + text: gives a short chunk the context of where it sits."""
        return f"{self.section}\n\n{self.text}"


PARA = re.compile(r"\n[ \t]*\n\s*")
LINE = re.compile(r"\n")
SENTENCE = re.compile(r"[.!?][\"')\]]*\s+")
FENCE_BLOCK = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.S | re.M)


def _boundaries(text: str) -> dict[int, int]:
    """Positions where a chunk may start/end -> rank (3 paragraph, 2 line, 1 sentence).
    Inside code fences only line breaks count, so code is not cut mid-line."""
    fences = [(m.start(), m.end()) for m in FENCE_BLOCK.finditer(text)]

    def in_fence(p: int) -> bool:
        return any(a < p < b for a, b in fences)

    out: dict[int, int] = {}
    for rank, pattern in ((1, SENTENCE), (2, LINE), (3, PARA)):
        for m in pattern.finditer(text):
            p = m.end()
            if 0 < p < len(text) and (rank == 2 or not in_fence(p)):
                out[p] = max(out.get(p, 0), rank)
    return out


def split_text(text: str, max_chars: int, overlap_chars: int) -> list[tuple[int, int]]:
    """Return (start, end) spans covering `text`, each at most ~max_chars."""
    n = len(text)
    if n <= max_chars:
        return [(0, n)]

    bounds = _boundaries(text)
    positions = sorted(bounds)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        if n - start <= max_chars:
            spans.append((start, n))
            break

        limit = start + max_chars
        window = [p for p in positions if start + max_chars // 2 < p <= limit]
        if window:
            end = max(window, key=lambda p: (bounds[p], p))
        else:                                   # no boundary: cut at a space
            ws = text.rfind(" ", start + max_chars // 2, limit)
            end = ws + 1 if ws != -1 else limit

        # Avoid a tiny last chunk: absorb a short tail into this one.
        if n - end < max_chars // 4 and n - start <= int(max_chars * 1.25):
            end = n
        spans.append((start, end))
        if end >= n:
            break

        nxt = [p for p in positions if end - overlap_chars <= p < end and p > start]
        start = nxt[0] if nxt else end
    return spans


def _strip(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def chunk_document(doc: ParsedDoc,
                   chunk_tokens: int = CONFIG.chunk_tokens,
                   overlap_tokens: int = CONFIG.chunk_overlap_tokens,
                   chars_per_token: int = CONFIG.chars_per_token) -> list[Chunk]:
    max_chars = chunk_tokens * chars_per_token
    overlap_chars = overlap_tokens * chars_per_token
    chunks: list[Chunk] = []
    for sec in doc.sections:
        body = doc.section_text(sec)
        for a, b in split_text(body, max_chars, overlap_chars):
            a, b = _strip(body, a, b)
            if b <= a:
                continue
            start, end = sec.char_start + a, sec.char_start + b
            chunks.append(Chunk(index=len(chunks), char_start=start, char_end=end,
                                text=doc.text[start:end], heading_path=sec.heading_path,
                                page=sec.page))
    return chunks
