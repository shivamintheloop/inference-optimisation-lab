"""Turn a file into clean text split into sections.

Markdown -> one section per heading, labelled with its heading path
            ("Title > Section > Concept"). The Contents link list, horizontal
            rules, YAML frontmatter and a quote-only intro under the title are
            dropped. Code fences stay whole ('#' inside them is not a heading).
PDF      -> one section per page, labelled with the page number. Lines that
            repeat at the top or bottom of most pages (headers, footers) and
            bare page numbers are removed.

Every section is a slice of ParsedDoc.text, so chunk offsets
(char_start / char_end) all point into one string per document.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


class PermanentError(Exception):
    """A problem with the file itself; retrying will not help."""


@dataclass
class Section:
    heading_path: list[str]
    page: int | None
    char_start: int
    char_end: int


@dataclass
class ParsedDoc:
    title: str
    text: str
    sections: list[Section]
    pages: int | None = None
    empty_pages: list[int] = field(default_factory=list)

    def section_text(self, s: Section) -> str:
        return self.text[s.char_start:s.char_end]


SECTION_SEP = "\n\n"
MIN_SECTION_CHARS = 20


def _assemble(title: str, parts: list[tuple[list[str], int | None, str]]) -> ParsedDoc:
    """Join section bodies into one text and record where each one sits."""
    buf: list[str] = []
    sections: list[Section] = []
    pos = 0
    for path, page, body in parts:
        body = body.strip()
        if len(body) < MIN_SECTION_CHARS:
            continue
        if buf:
            buf.append(SECTION_SEP)
            pos += len(SECTION_SEP)
        sections.append(Section(path, page, pos, pos + len(body)))
        buf.append(body)
        pos += len(body)
    return ParsedDoc(title=title, text="".join(buf), sections=sections)


# ---------------------------------------------------------------- markdown

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
RULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
SKIP_SECTIONS = {"contents", "table of contents"}


def parse_markdown(raw: str, fallback_title: str) -> ParsedDoc:
    raw = FRONTMATTER.sub("", raw, count=1)
    title: str | None = None
    stack: list[tuple[int, str]] = []          # headings below the title
    parts: list[tuple[list[str], int | None, str]] = []
    body: list[str] = []
    in_fence = False

    def flush() -> None:
        if body:
            path = [title or fallback_title] + [h for _, h in stack]
            parts.append((path, None, "\n".join(body)))
            body.clear()

    for line in raw.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            body.append(line)
            continue
        if not in_fence:
            m = HEADING.match(line)
            if m:
                flush()
                level, name = len(m.group(1)), m.group(2).strip()
                if level == 1 and title is None:
                    title = name
                    stack.clear()
                    continue
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, name))
                continue
            if RULE.match(line):
                continue
        body.append(line)
    flush()

    kept = []
    for path, page, text in parts:
        if path[-1].strip().lower() in SKIP_SECTIONS:
            continue
        lines = [l for l in text.splitlines() if l.strip()]
        if len(path) == 1 and lines and all(l.lstrip().startswith(">") for l in lines):
            continue                           # boilerplate note under the title
        kept.append((path, page, text))
    return _assemble(title or fallback_title, kept)


# ---------------------------------------------------------------- pdf

PAGE_NUMBER = re.compile(r"^\s*(page\s*)?\d+(\s*(of|/)\s*\d+)?\s*$", re.I)
EDGE_LINES = 3                                 # how many lines count as "top" / "bottom"


def _norm(line: str) -> str:
    """Compare header/footer lines ignoring spacing, case and numbers."""
    return re.sub(r"\d+", "#", " ".join(line.split()).lower())


def _edge_indexes(lines: list[str]) -> set[int]:
    """Indexes of the top and bottom lines. Short pages get a smaller edge
    zone, otherwise every line would count as a possible header."""
    nonempty = [i for i, l in enumerate(lines) if l.strip()]
    k = min(EDGE_LINES, len(nonempty) // 4)
    if k == 0:
        return set()
    return set(nonempty[:k] + nonempty[-k:])


def remove_repeated_lines(pages: list[list[str]]) -> list[list[str]]:
    """Drop headers and footers: lines near the top or bottom of a page that
    repeat on more than half of the pages, plus bare page numbers."""
    n = len(pages)
    repeated: set[str] = set()
    if n >= 3:
        counts: Counter[str] = Counter()
        for lines in pages:
            counts.update({_norm(lines[i]) for i in _edge_indexes(lines)})
        repeated = {k for k, c in counts.items() if k and c > n / 2}

    out = []
    for lines in pages:
        edges = _edge_indexes(lines)
        out.append([l for i, l in enumerate(lines)
                    if not (i in edges and (_norm(l) in repeated or PAGE_NUMBER.match(l)))])
    return out


def _join_lines(lines: list[str]) -> str:
    """PDF text has a line break at the end of every printed line. Re-join
    words split with a hyphen, turn single line breaks into spaces, and keep
    blank lines as paragraph breaks."""
    text = "\n".join(lines)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?<!\n)[ \t]*\n[ \t]*(?!\n)", " ", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def parse_pdf(path: Path) -> ParsedDoc:
    import logging

    from pypdf import PdfReader

    logging.getLogger("pypdf").setLevel(logging.ERROR)   # we report problems ourselves
    try:
        reader = PdfReader(path)
        raw_pages = [(p.extract_text() or "") for p in reader.pages]
    except Exception as e:
        raise PermanentError(f"unreadable PDF: {type(e).__name__}: {e}") from e

    n = len(raw_pages)
    if n == 0:
        raise PermanentError("PDF has no pages")
    empty = [i for i, t in enumerate(raw_pages, 1) if len(t.strip()) < MIN_SECTION_CHARS]
    if len(empty) > n / 2:
        raise PermanentError(f"needs OCR: {len(empty)} of {n} pages have no text layer")

    try:
        meta_title = (reader.metadata.title or "").strip() if reader.metadata else ""
    except Exception:
        meta_title = ""
    title = meta_title if len(meta_title) >= 3 else path.stem

    pages = remove_repeated_lines([t.splitlines() for t in raw_pages])
    parts = [([title, f"page {i}"], i, _join_lines(lines)) for i, lines in enumerate(pages, 1)]
    doc = _assemble(title, parts)
    doc.pages = n
    doc.empty_pages = empty
    return doc


# ---------------------------------------------------------------- entry point

def parse_file(path: Path) -> ParsedDoc:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return parse_markdown(path.read_text(errors="replace"), fallback_title=path.stem)
    if suffix == ".pdf":
        return parse_pdf(path)
    raise PermanentError(f"unsupported file type: {suffix}")
