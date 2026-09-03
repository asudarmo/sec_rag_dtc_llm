from rag import rerank as rr


class _FakeCrossEncoder:
    def predict(self, pairs):
        # Deterministic stand-in score: longer passage = "more relevant" — lets the
        # test assert on exact ordering without downloading the real model.
        return [len(text) for _, text in pairs]


def test_rerank_sorts_by_score_desc_and_truncates(monkeypatch):
    monkeypatch.setattr(rr, "_get_model", lambda: _FakeCrossEncoder())
    hits = [
        {"text": "short"},
        {"text": "a much longer piece of text"},
        {"text": "mid length text"},
    ]

    result = rr.rerank("query", hits, k=2)

    assert len(result) == 2
    assert result[0]["text"] == "a much longer piece of text"
    assert result[1]["text"] == "mid length text"
    assert result[0]["rerank_score"] > result[1]["rerank_score"]


def test_rerank_empty_input_returns_empty(monkeypatch):
    monkeypatch.setattr(rr, "_get_model", lambda: _FakeCrossEncoder())
    assert rr.rerank("query", [], k=5) == []
