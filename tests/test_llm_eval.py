from eval.llm_eval import summarize


def test_summarize_computes_per_config_averages():
    raw = [
        {"config": "dense_no_rewrite", "question": "q1", "score": 1.0, "covers_both_sides": True},
        {"config": "dense_no_rewrite", "question": "q2", "score": 0.0, "covers_both_sides": False},
        {"config": "dense_with_rewrite", "question": "q1", "score": 1.0, "covers_both_sides": True},
        {"config": "dense_with_rewrite", "question": "q2", "score": 1.0, "covers_both_sides": True},
    ]

    summary = summarize(raw)
    by_config = {s["config"]: s for s in summary}

    assert by_config["dense_no_rewrite"]["avg_score"] == 0.5
    assert by_config["dense_no_rewrite"]["pct_covers_both_sides"] == 0.5
    assert by_config["dense_with_rewrite"]["avg_score"] == 1.0
    assert by_config["dense_with_rewrite"]["pct_covers_both_sides"] == 1.0
    assert by_config["dense_no_rewrite"]["n_questions"] == 2


def test_summarize_skips_configs_with_no_completed_items():
    raw = [{"config": "hybrid_no_rewrite", "question": "q1", "score": 1.0, "covers_both_sides": True}]

    summary = summarize(raw)

    assert len(summary) == 1
    assert summary[0]["config"] == "hybrid_no_rewrite"
