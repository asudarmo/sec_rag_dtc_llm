from eval.agentic_eval import summarize


def test_summarize_computes_per_config_averages():
    raw = [
        {"config": "traditional", "question": "q1", "faithfulness": 1.0, "context_precision": 0.8, "relevance": "RELEVANT", "latency_s": 5.0, "n_gemini_calls": 2},
        {"config": "traditional", "question": "q2", "faithfulness": 0.5, "context_precision": 0.4, "relevance": "PARTLY_RELEVANT", "latency_s": 3.0, "n_gemini_calls": 2},
        {"config": "agentic", "question": "q1", "faithfulness": 1.0, "context_precision": 0.9, "relevance": "RELEVANT", "latency_s": 12.0, "n_gemini_calls": 3},
        {"config": "agentic", "question": "q2", "faithfulness": 1.0, "context_precision": 1.0, "relevance": "RELEVANT", "latency_s": 8.0, "n_gemini_calls": 2},
    ]

    summary = summarize(raw)
    by_config = {s["config"]: s for s in summary}

    assert by_config["traditional"]["avg_faithfulness"] == 0.75
    assert by_config["traditional"]["avg_context_precision"] == 0.6
    assert by_config["traditional"]["pct_relevant"] == 0.5
    assert by_config["traditional"]["avg_latency_s"] == 4.0
    assert by_config["traditional"]["avg_gemini_calls"] == 2.0

    assert by_config["agentic"]["pct_relevant"] == 1.0
    assert by_config["agentic"]["avg_gemini_calls"] == 2.5


def test_summarize_handles_missing_faithfulness():
    raw = [
        {"config": "agentic", "question": "q1", "faithfulness": None, "context_precision": 0.5, "relevance": "NON_RELEVANT", "latency_s": 1.0, "n_gemini_calls": 1},
    ]

    summary = summarize(raw)

    assert summary[0]["avg_faithfulness"] is None
    assert summary[0]["avg_context_precision"] == 0.5
