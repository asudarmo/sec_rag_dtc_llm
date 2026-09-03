from rag import hybrid

# RRF math itself is tested in tests/test_rrf.py (rag/rrf.py); these tests cover
# hybrid.py's own job — wiring dense+sparse into that fusion for a single query.


def test_retrieve_boosts_docs_ranked_in_both_lists(monkeypatch):
    dense_hits = [
        {"id": "d1", "text": "t1", "metadata": {}, "distance": 0.1},
        {"id": "d2", "text": "t2", "metadata": {}, "distance": 0.2},
        {"id": "d3", "text": "t3", "metadata": {}, "distance": 0.3},
    ]
    sparse_hits = [
        {"id": "d3", "text": "t3", "metadata": {}, "score": 9.0},
        {"id": "d2", "text": "t2", "metadata": {}, "score": 8.0},
        {"id": "d4", "text": "t4", "metadata": {}, "score": 7.0},
    ]
    monkeypatch.setattr(hybrid.retrieve_dense, "retrieve", lambda query, k, filter_dict=None: dense_hits)
    monkeypatch.setattr(hybrid.retrieve_sparse, "retrieve", lambda query, k, filter_dict=None: sparse_hits)

    results = hybrid.retrieve("irrelevant query", k=4)
    result_ids = [h["id"] for h in results]

    # d2 and d3 each appear in both lists (near the top of one, mid of the other), so
    # RRF should rank them above d1/d4, which each only appear in a single list.
    assert set(result_ids[:2]) == {"d2", "d3"}
    assert set(result_ids) == {"d1", "d2", "d3", "d4"}
