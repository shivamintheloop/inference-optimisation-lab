from rag import bench_llm as b


def row(p, qid, text, tpot, **extra):
    return {"pass": p, "id": qid, "text": text, "ttft_ms": 100.0 + p, "tpot_ms": tpot,
            "decode_tok_s": 1000 / tpot, "e2e_ms": 1000.0, "output_tokens": 50, **extra}


def test_summarize_median_of_pass_p50s_and_pooled_acceptance():
    rows = [row(0, "q1", "a", 10.0, draft_tokens=10, draft_accepted=8, draft_steps=4),
            row(0, "q2", "b", 12.0, draft_tokens=10, draft_accepted=2, draft_steps=4),
            row(1, "q1", "a", 11.0, draft_tokens=10, draft_accepted=8, draft_steps=4),
            row(1, "q2", "B", 13.0, draft_tokens=10, draft_accepted=2, draft_steps=4)]
    s = b.summarize(rows)
    assert s["passes"] == 2 and s["prompts"] == 2
    assert s["tpot_ms"] == 13.0 and s["tpot_ms_range"] == [12.0, 13.0]   # per-pass p50s: 12, 13
    assert s["acceptance_rate"] == 0.5                  # 20 kept of 40 drafted
    assert s["tokens_per_step"] == 2.25                 # (20 kept + 16 checks) / 16 checks
    assert s["stable_across_passes"] == 1               # q2 changed between passes


def test_summarize_without_drafts_has_no_acceptance():
    s = b.summarize([row(0, "q1", "a", 10.0)])
    assert "acceptance_rate" not in s and s["stable_across_passes"] == 1


def test_same_output_compares_first_passes():
    base = [row(0, "q1", "a", 10), row(0, "q2", "b", 10), row(1, "q1", "x", 10)]
    new = [row(0, "q1", "a", 5), row(0, "q2", "different", 5)]
    assert b.same_output(new, base) == 1


def test_inference_row_reads_k_from_server_cmd():
    res = {"label": "draft 1B k=8", "path": "/x/results/llm-draft.json",
           "server": {"engine": "llama_server", "build": "b11146",
                      "cmd": "mode=draft k=8 llama-server -m x --spec-draft-n-max 8", "rss_mb": 7000},
           "summary": {**b.summarize([row(0, "q1", "a", 8.0, draft_tokens=8, draft_accepted=6,
                                          draft_steps=2)]),
                       "same_as_baseline": 1, "speedup_tpot": 1.45}}
    line = b.inference_row(res, "note")
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    assert cells[0] == "draft 1B k=8" and cells[2] == "8"
    assert cells[6] == "1.45×" and cells[8] == "0.75" and cells[10] == "1/1"
    assert cells[13] == "`llm-draft.json`" and cells[14] == "note"
