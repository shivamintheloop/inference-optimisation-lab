import sqlite3

import pytest

from rag import docs_db
from rag.evaluate import first_good_rank, summarize
from rag.retrieve import Hit, bm25_search, fts_query, rrf_merge


def test_rrf_adds_reciprocal_ranks():
    vec = [Hit("a", 0.9), Hit("b", 0.8), Hit("c", 0.7)]
    kw = [Hit("c", 12.0), Hit("a", 9.0)]
    merged = rrf_merge({"vector": vec, "bm25": kw}, k=60)
    by_id = {h.chunk_id: h for h in merged}
    assert by_id["a"].score == pytest.approx(1 / 61 + 1 / 62)
    assert by_id["c"].score == pytest.approx(1 / 63 + 1 / 61)
    assert by_id["b"].score == pytest.approx(1 / 62)
    assert by_id["a"].ranks == {"vector": 1, "bm25": 2}
    # found by both lists beats found by one, whatever the raw scores were
    assert [h.chunk_id for h in merged] == ["a", "c", "b"]


def test_fts_query_quotes_words_and_drops_stopwords():
    assert fts_query("What is the KV-cache?") == '"kv" OR "cache"'
    assert fts_query("what is it?") is None
    # FTS5 operators are quoted, so they are treated as plain words
    assert fts_query("NEAR AND NOT") == '"near" OR "not"'


@pytest.fixture
def conn(tmp_path):
    c = docs_db.connect(tmp_path / "t.db")
    docs_db.replace_fts(c, "notes/a.md", [("c1", "Caching keys and values speeds up decoding"),
                                          ("c2", "HNSW builds a layered graph for search")])
    docs_db.replace_fts(c, "notes/b.md", [("c3", "Reciprocal rank fusion merges lists")])
    yield c
    c.close()


def test_bm25_finds_stemmed_keywords(conn):
    hits = bm25_search(conn, "How does the KV cache work?", top_k=5)
    assert [h.chunk_id for h in hits] == ["c1"]      # "cache" matches "Caching" (porter)
    assert hits[0].score > 0 and hits[0].ranks == {"bm25": 1}


def test_replace_fts_swaps_one_documents_rows(conn):
    docs_db.replace_fts(conn, "notes/a.md", [("c9", "only one chunk now")])
    assert docs_db.fts_count(conn, "notes/a.md") == 1
    assert docs_db.fts_count(conn, "notes/b.md") == 1


def test_eval_scoring():
    assert first_good_rank(["x", "y", "good", "good"], ["good"]) == 3
    assert first_good_rank(["x"] * 12 + ["good"], ["good"]) is None   # beyond depth 10
    s = summarize([{"rank": 1}, {"rank": 4}, {"rank": None}, {"rank": 8}])
    assert s == {"n": 4, "hit@1": 0.25, "hit@5": 0.5,
                 "mrr@10": round((1 + 1 / 4 + 1 / 8) / 4, 3)}


def test_results_table_and_log(tmp_path):
    from rag.evaluate import record_results, results_table

    m = {"all": {"hit@1": 0.5, "hit@5": 0.75, "mrr@10": 0.6}, "keyword": {"hit@5": 1.0},
         "paraphrase": {"hit@5": 0.5}, "p50_ms": 21.4}
    res = {"time": "2026-10-01T20:00:00", "config_hash": "abcd1234", "questions_hash": "q1q2q3q4",
           "summary": {"vector": m}, "path": "/x/eval-1.json"}
    table = results_table(res, "Phase 9: test")
    assert "## Phase 9: test" in table
    assert "| vector | 0.50 | 0.75 | 0.600 | 1.00 | 0.50 | 21 ms |" in table
    log = tmp_path / "RESULTS.md"
    record_results(res, "Phase 9: test", log=log)
    record_results(res, "Phase 10: test", log=log)
    text = log.read_text()
    assert text.startswith("# Retrieval results by phase")
    assert text.count("## Phase") == 2
