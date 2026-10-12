from rag import generate as gen
from rag import ollama_client
from rag.context import build_context, merge_hits
from rag.retrieve import Hit
from rag.trace import trace

DOC = "A" * 100 + "B" * 100 + "C" * 100       # chunks are exact slices of this text


def hit(cid, start, end, doc="notes/x.md", section="X > Card"):
    return Hit(cid, 1.0, {}, {
        "doc_id": doc, "title": "X", "section": section, "page": None,
        "source_path": f"/notes/{doc}", "char_start": start, "char_end": end,
        "text": DOC[start:end]})


def test_overlapping_neighbours_join_without_repeating_text():
    # re-rank order: the later chunk ranked first; overlap is 150..180
    sources = merge_hits([hit("b", 150, 300), hit("a", 0, 180)])
    assert len(sources) == 1
    s = sources[0]
    assert s.text == DOC                        # no duplicated overlap
    assert (s.char_start, s.char_end, s.best_rank) == (0, 300, 1)
    assert s.chunk_ids == ["a", "b"]


def test_gaps_other_sections_and_docs_stay_separate():
    sources = merge_hits([
        hit("a", 0, 100),
        hit("b", 150, 250),                     # gap after a
        hit("c", 100, 150, section="X > Other"),  # touches a but another section
        hit("d", 0, 100, doc="notes/y.md"),
    ])
    assert [s.chunk_ids for s in sources] == [["a"], ["b"], ["c"], ["d"]]


def test_budget_skips_what_does_not_fit_and_renumbers():
    hits = [hit("a", 0, 100), hit("big", 0, 300, doc="notes/big.md"), hit("c", 0, 40, doc="notes/c.md")]
    # each source costs (len(header) + len(text)) // 4 tokens; budget fits a and c only
    sources, context = build_context(hits, budget_tokens=50, chars_per_token=4)
    assert [s.chunk_ids for s in sources] == [["a"], ["c"]]
    assert [s.n for s in sources] == [1, 2]
    assert context.startswith("[S1] X | Card\n") and "\n\n[S2] X | Card\n" in context


def test_parse_citations():
    text = "KV cache stores keys [S1]. It grows with context [S2][S3], see [S1, S4] and [S9]."
    assert gen.parse_citations(text, n_sources=4) == ([1, 2, 3, 4], [9])
    assert gen.parse_citations("no tags here [1]", 3) == ([], [])


def test_prompt_order_keeps_fixed_system_prompt_first():
    m1 = gen.build_messages("q1", "[S1] ctx one")
    m2 = gen.build_messages("q2", "[S1] ctx two")
    assert m1[0] == m2[0] and m1[0]["role"] == "system"   # identical prefix for caching
    assert m1[1]["content"].index("ctx one") < m1[1]["content"].index("Question: q1")


def fake_stream(messages, model=None, max_tokens=None):
    for piece in ["KV ", "cache ", "[S1]."]:
        yield {"type": "token", "text": piece}
    yield {"type": "done", "done_reason": "stop", "timings": {
        "prompt_eval_count": 1000, "prompt_eval_ms": 500.0,
        "eval_count": 3, "eval_ms": 30.0, "load_ms": 0.0}}


def test_answer_flow_records_llm_metrics(monkeypatch, tmp_path):
    import rag.retrieve
    monkeypatch.setattr(rag.retrieve, "search", lambda conn, q, mode: [hit("a", 0, 100)])
    monkeypatch.setattr(ollama_client, "chat_stream", fake_stream)
    streamed = []
    with trace("query", out_dir=tmp_path) as t:
        a = gen.answer(None, "what is a kv cache?", on_token=streamed.append)
    assert a.text == "KV cache [S1]." == "".join(streamed)
    assert a.cited == [1] and not a.not_found and not a.llm_refused
    m = a.metrics
    assert m["output_tokens"] == 3 and m["prompt_tokens"] == 1000
    assert m["prefill_tok_s"] == 2000 and m["decode_tok_s"] == 100.0
    assert m["ttft_ms"] is not None and m["tpot_ms"] is not None
    assert [s.name for s in t.spans] == ["context", "llm.generate", "citations"]


def test_not_found_skips_the_llm(monkeypatch, tmp_path):
    import rag.retrieve

    def boom(*a, **k):
        raise AssertionError("LLM must not be called")

    weak = hit("a", 0, 100)
    weak.score = -11.0
    monkeypatch.setattr(rag.retrieve, "search", lambda conn, q, mode: [weak])
    monkeypatch.setattr(ollama_client, "chat_stream", boom)
    with trace("query", out_dir=tmp_path):
        a = gen.answer(None, "best pizza in Naples?")
    assert a.not_found and a.text == gen.NOT_FOUND_REPLY and a.sources == []
