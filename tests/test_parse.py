from rag.parse import parse_markdown, remove_repeated_lines, _join_lines

MD = """---
title: x
---

# Design a RAG System - Flash Notes

> Quick-revision concept cards. For the full explanations, see the notes.

## Contents

- [Chunking](#chunking)
- [Retrieval](#retrieval)

---

## Chunking

### Fixed-Size Chunks

- Split text every 400 tokens with 50 tokens of overlap.

```python
# keep only k most likely
top_k = 5
```

### Semantic Chunks

- Split where the topic changes, using embedding similarity.

## Retrieval

Hybrid search runs vector and keyword search in parallel.
"""


def test_markdown_sections_have_heading_paths():
    doc = parse_markdown(MD, fallback_title="fallback")
    assert doc.title == "Design a RAG System - Flash Notes"
    paths = [s.heading_path for s in doc.sections]
    assert paths == [
        [doc.title, "Chunking", "Fixed-Size Chunks"],
        [doc.title, "Chunking", "Semantic Chunks"],
        [doc.title, "Retrieval"],
    ]


def test_markdown_drops_contents_intro_rules_and_frontmatter():
    doc = parse_markdown(MD, fallback_title="fallback")
    assert "Quick-revision" not in doc.text
    assert "(#chunking)" not in doc.text
    assert "title: x" not in doc.text
    assert "\n---\n" not in doc.text


def test_hash_inside_code_fence_is_not_a_heading():
    doc = parse_markdown(MD, fallback_title="fallback")
    first = doc.section_text(doc.sections[0])
    assert "# keep only k most likely" in first
    assert all("keep only" not in p for s in doc.sections for p in s.heading_path)


def test_section_offsets_point_into_doc_text():
    doc = parse_markdown(MD, fallback_title="fallback")
    for s in doc.sections:
        body = doc.section_text(s)
        assert body == body.strip() and body


def test_repeated_headers_footers_and_page_numbers_removed():
    pages = [
        ["ACME Corp Handbook", f"Body text of page {i} talks about leave.", "More text.",
         f"Page {i} of 4"]
        for i in range(1, 5)
    ]
    cleaned = remove_repeated_lines(pages)
    for i, lines in enumerate(cleaned, 1):
        assert lines == [f"Body text of page {i} talks about leave.", "More text."]


def test_unique_first_lines_are_kept():
    pages = [[f"Chapter {name}", "text"] for name in ("one", "two", "three")]
    # "chapter one/two/three" differ after normalising, so they are not headers
    assert remove_repeated_lines(pages) == pages


def test_join_lines_rejoins_hyphenated_words_and_keeps_paragraphs():
    text = _join_lines(["The attention mecha-", "nism is fast.", "", "New paragraph here."])
    assert text == "The attention mechanism is fast.\n\nNew paragraph here."
