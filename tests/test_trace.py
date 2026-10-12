import json
import time

import pytest

from rag.trace import config_hash, span, trace


def read_lines(dir_):
    files = list(dir_.glob("*.jsonl"))
    assert len(files) == 1
    return [json.loads(line) for line in files[0].read_text().splitlines()]


def test_nested_spans_link_to_parents(tmp_path):
    with trace("query", out_dir=tmp_path):
        with span("outer"):
            with span("inner", top_k=30) as s:
                s.set("hits", 7)
        with span("sibling"):
            pass

    (rec,) = read_lines(tmp_path)
    by_name = {s["name"]: s for s in rec["spans"]}
    assert by_name["outer"]["parent_id"] is None
    assert by_name["inner"]["parent_id"] == by_name["outer"]["span_id"]
    assert by_name["sibling"]["parent_id"] is None
    assert by_name["inner"]["attrs"] == {"top_k": 30, "hits": 7}


def test_child_time_fits_inside_parent(tmp_path):
    with trace("query", out_dir=tmp_path):
        with span("outer"):
            time.sleep(0.01)
            with span("inner"):
                time.sleep(0.02)

    (rec,) = read_lines(tmp_path)
    outer, inner = rec["spans"]
    assert inner["duration_ms"] >= 20
    assert outer["duration_ms"] >= inner["duration_ms"] + 10
    assert inner["start_ms"] >= outer["start_ms"]
    assert rec["duration_ms"] >= outer["duration_ms"]


def test_error_is_recorded_and_trace_still_written(tmp_path):
    with pytest.raises(ValueError):
        with trace("query", out_dir=tmp_path):
            with span("boom"):
                raise ValueError("bad pdf")

    (rec,) = read_lines(tmp_path)
    assert rec["status"] == "error"
    assert rec["spans"][0]["status"] == "error"
    assert "bad pdf" in rec["spans"][0]["error"]


def test_span_outside_trace_is_a_noop(tmp_path):
    with span("orphan") as s:
        s.set("x", 1)
    assert not list(tmp_path.glob("*.jsonl"))


def test_one_line_per_trace_with_config_hash(tmp_path):
    cfg = {"chunk_tokens": 400, "llm_model": "llama3.1:8b"}
    for _ in range(2):
        with trace("query", config=cfg, out_dir=tmp_path):
            pass

    recs = read_lines(tmp_path)
    assert len(recs) == 2
    assert recs[0]["trace_id"] != recs[1]["trace_id"]
    assert recs[0]["config_hash"] == recs[1]["config_hash"] == config_hash(cfg)
    assert config_hash({**cfg, "chunk_tokens": 500}) != config_hash(cfg)
