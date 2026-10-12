import json

import pytest

from rag import llamacpp_client as lc


def sse(*objs):
    return [f"data: {json.dumps(o)}" for o in objs] + ["", "data: [DONE]"]


def test_parse_sse_tokens_finish_and_timings():
    lines = sse(
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "KV "}}]},
        {"choices": [{"delta": {"content": "cache [S1]."}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "timings": {"prompt_n": 1000, "prompt_ms": 812.5, "predicted_n": 3,
                     "predicted_ms": 30.0, "cache_n": 0}},
    )
    events = list(lc.parse_sse(iter(lines)))
    assert [e["text"] for e in events if e["type"] == "token"] == ["KV ", "cache [S1]."]
    done = events[-1]
    assert done["type"] == "done" and done["done_reason"] == "stop"
    assert done["timings"] == {"prompt_eval_count": 1000, "eval_count": 3,
                               "prompt_eval_ms": 812.5, "eval_ms": 30.0,
                               "cache_n": 0, "load_ms": 0.0}


def test_parse_sse_error_chunk():
    with pytest.raises(lc.LlamaServerError):
        list(lc.parse_sse(iter(sse({"error": {"message": "context too long"}}))))


METRICS = """# HELP llamacpp:spec_decode_num_draft_tokens_total Speculative: Total draft tokens generated
# TYPE llamacpp:spec_decode_num_draft_tokens_total counter
llamacpp:spec_decode_num_draft_tokens_total 120
llamacpp:spec_decode_num_accepted_tokens_total 90
llamacpp:spec_decode_num_drafts_total 30
llamacpp:tokens_predicted_total 500
"""


def test_parse_metrics_and_delta():
    after = lc.parse_metrics(METRICS)
    assert after == {"draft_tokens": 120, "draft_accepted": 90, "draft_steps": 30}
    before = {"draft_tokens": 20, "draft_accepted": 10, "draft_steps": 5}
    d = lc.spec_delta(before, after)
    assert d == {"draft_tokens": 100, "draft_accepted": 80, "draft_steps": 25,
                 "acceptance_rate": 0.8}


def test_delta_without_speculation_or_metrics():
    same = {"draft_tokens": 7, "draft_accepted": 3, "draft_steps": 2}
    assert lc.spec_delta(same, same) == {}        # baseline server: counters don't move
    assert lc.spec_delta(None, same) == {}        # server started without --metrics
