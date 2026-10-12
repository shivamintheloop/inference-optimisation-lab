from rag import rerank as rr
from rag.evaluate import threshold_analysis
from rag.retrieve import Hit


class FakeCrossEncoder:
    """Scores a chunk by how many question words it contains."""

    def predict(self, pairs, batch_size=32):
        return [sum(w in text.lower() for w in q.lower().split()) for q, text in pairs]


def hit(cid, text, heading="Card"):
    return Hit(cid, 0.0, {"hybrid_input": 1}, {"section": f"Doc > {heading}", "text": text})


def test_rerank_reorders_and_keeps_top_n(monkeypatch):
    monkeypatch.setattr(rr, "_model", lambda: FakeCrossEncoder())
    rr._model.cache_info = lambda: type("I", (), {"currsize": 1})()
    hits = [hit("a", "nothing useful"), hit("b", "rrf merges ranked lists"),
            hit("c", "ranked lists"), hit("d", "no payload")]
    hits[3].payload = None                      # unfetched hits are skipped
    out = rr.rerank("rrf merges ranked lists", hits, top_n=2, candidates=10)
    assert [h.chunk_id for h in out] == ["b", "c"]
    assert out[0].ranks["hybrid"] == 2          # where it was before re-ranking
    assert out[0].score == 4


def test_pair_text_uses_own_heading():
    h = hit("a", "body text", heading="Reciprocal Rank Fusion (RRF)")
    assert rr.pair_text(h) == "Reciprocal Rank Fusion (RRF)\nbody text"


def test_not_found():
    assert rr.is_not_found([], threshold=0.0)
    assert rr.is_not_found([Hit("a", -3.0)], threshold=0.0)
    assert not rr.is_not_found([Hit("a", 2.5)], threshold=0.0)


def test_threshold_picks_best_separation():
    rows = ([{"style": "keyword", "top_score": s} for s in (5.0, 3.0, 1.0)]
            + [{"style": "unanswerable", "top_score": s} for s in (-4.0, -2.0)])
    th = threshold_analysis(rows)
    assert th["separable"] and th["suggested_threshold"] == -0.5   # midpoint of -2 and 1
    assert th["kept_answerable"] == 3 and th["refused_unanswerable"] == 2
