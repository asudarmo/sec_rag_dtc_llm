from eval.retrieval_eval import evaluate_config


def test_hit_rate_and_mrr_hand_computed(monkeypatch):
    import eval.retrieval_eval as re_module

    # q1's chunk found at rank 2, q2's chunk found at rank 1, q3's chunk not found at all.
    fake_results = {
        "q1": [{"id": "other"}, {"id": "target-1"}],
        "q2": [{"id": "target-2"}, {"id": "other"}],
        "q3": [{"id": "other"}, {"id": "another"}],
    }
    monkeypatch.setattr(
        re_module.pipeline, "retrieve",
        lambda query, k, method, use_rerank: fake_results[query],
    )

    ground_truth = [
        {"question": "q1", "chunk_id": "target-1"},
        {"question": "q2", "chunk_id": "target-2"},
        {"question": "q3", "chunk_id": "target-3"},
    ]
    result = evaluate_config(ground_truth, {"name": "test", "method": "dense", "use_rerank": False}, k=2)

    assert result["hit_rate"] == round(2 / 3, 4)
    # reciprocal ranks: 1/2, 1/1, 0 -> mean = (0.5 + 1.0 + 0.0) / 3
    assert result["mrr"] == round((0.5 + 1.0 + 0.0) / 3, 4)
