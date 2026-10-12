import pytest

from rag import bench_ab as ab


def test_script_args():
    assert ab.script_args("baseline") == (["baseline"], {})
    assert ab.script_args("draft:8") == (["draft", "8"], {})
    assert ab.script_args("draft:3:0.8") == (["draft", "3", "0.8"], {})
    assert ab.script_args("ngram:16:4") == (["ngram", "16", "4"], {})
    assert ab.script_args("draft:3@t=q8,d=1bq4") == (
        ["draft", "3"], {"TARGET": "llama3.1:8b-instruct-q8_0", "DRAFT": "llama3.2:1b-instruct-q4_K_M"})
    assert ab.script_args("baseline@t=gemma3:27b")[1] == {"TARGET": "gemma3:27b"}
    for bad in ("draft", "ngram:16", "fast", "baseline:2", "draft:3@x=1"):
        with pytest.raises(ValueError):
            ab.script_args(bad)


def test_rotation_moves_every_variant_through_every_slot():
    v = ["baseline", "draft:3", "draft:8"]
    orders = [ab.rotated(v, b) for b in range(3)]
    assert orders[0] == v and orders[1] == ["draft:3", "draft:8", "baseline"]
    assert {o[0] for o in orders} == set(v)       # each variant goes first once


def test_paired_ratio_matches_same_pass_and_prompt():
    base = [{"pass": 0, "id": "q1", "tpot_ms": 12.0}, {"pass": 0, "id": "q2", "tpot_ms": 20.0},
            {"pass": 1, "id": "q1", "tpot_ms": 30.0}]
    new = [{"pass": 0, "id": "q1", "tpot_ms": 6.0},   # 2.0x
           {"pass": 0, "id": "q2", "tpot_ms": 25.0},  # 0.8x
           {"pass": 1, "id": "q1", "tpot_ms": 20.0}]  # 1.5x (paired with the hot pass-1 baseline)
    assert ab.paired_ratio(new, base, "tpot_ms") == 1.5
    assert ab.paired_ratio([], base, "tpot_ms") is None


def test_block_drift_is_last_over_first_per_block():
    rows = [{"pass": 0, "block": 0, "tpot_ms": 10.0}, {"pass": 0, "block": 0, "tpot_ms": 30.0},
            {"pass": 0, "block": 1, "tpot_ms": 12.0}, {"pass": 0, "block": 1, "tpot_ms": 12.0},
            {"pass": 1, "block": 0, "tpot_ms": 9.0}]               # one-request block: skipped
    assert ab.block_drift(rows) == [1.0, 3.0]
