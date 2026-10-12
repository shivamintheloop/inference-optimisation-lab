from rag.chunk import chunk_document, split_text
from rag.parse import parse_markdown

MAX, OVERLAP = 400, 60


def long_text(paragraphs=12):
    return "\n\n".join(
        f"Paragraph {i}. " + " ".join(f"Sentence {i}.{j} has some words in it." for j in range(4))
        for i in range(paragraphs))


def test_short_text_is_one_chunk():
    assert split_text("short text", MAX, OVERLAP) == [(0, 10)]


def test_spans_fit_cover_everything_and_overlap():
    text = long_text()
    spans = split_text(text, MAX, OVERLAP)
    assert len(spans) > 1
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    for a, b in spans[:-1]:
        assert b - a <= MAX
    for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
        assert a1 < a2 <= b1                     # no gap, always moves forward
        assert b1 - a2 <= OVERLAP                # overlap never larger than asked


def test_prefers_paragraph_breaks():
    text = long_text()
    spans = split_text(text, MAX, OVERLAP)
    for _, b in spans[:-1]:
        assert text[b - 2:b] == "\n\n"


def test_no_tiny_last_chunk():
    text = "word " * 95                          # 475 chars: just over MAX
    spans = split_text(text, MAX, OVERLAP)
    assert spans == [(0, len(text))]


def test_chunks_are_exact_slices_with_heading_paths():
    md = "# Title\n\n## Big Section\n\n" + long_text() + "\n\n## Small\n\nA short card about RRF merging."
    doc = parse_markdown(md, fallback_title="t")
    chunks = chunk_document(doc, chunk_tokens=100, overlap_tokens=15, chars_per_token=4)
    assert len(chunks) > 2
    assert [c.index for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert c.text == doc.text[c.char_start:c.char_end]
        assert c.text == c.text.strip()
    assert chunks[-1].heading_path == ["Title", "Small"]
    assert chunks[-1].embed_text().startswith("Title > Small\n\n")
    # a chunk never spans two sections
    assert all(c.heading_path == ["Title", "Big Section"] for c in chunks[:-1])
