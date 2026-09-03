from rag import judge


def test_judge_answer_quality_computes_faithfulness_as_fraction_supported(monkeypatch):
    monkeypatch.setattr(judge, "call_json", lambda prompt: {
        "claims": [{"claim": "a", "supported": True}, {"claim": "b", "supported": False}],
        "passage_relevance": [True],
        "relevance": "RELEVANT",
        "reasoning": "test",
    })

    result = judge.judge_answer_quality("q", [{"text": "p1", "metadata": {"ticker": "GOOGL", "sector": "tech", "filing_date": "x"}}], "a")

    assert result["faithfulness"] == 0.5


def test_judge_answer_quality_faithfulness_none_when_no_claims(monkeypatch):
    monkeypatch.setattr(judge, "call_json", lambda prompt: {
        "claims": [],
        "passage_relevance": [False],
        "relevance": "NON_RELEVANT",
        "reasoning": "insufficient information",
    })

    result = judge.judge_answer_quality("q", [{"text": "p1", "metadata": {"ticker": "GOOGL", "sector": "tech", "filing_date": "x"}}], "no answer")

    assert result["faithfulness"] is None


def test_judge_answer_quality_context_precision_hand_computed(monkeypatch):
    """RAGAS-style rank-aware precision: mean of precision@i over relevant positions.
    Relevance pattern [True, False, True] -> precision@1=1/1, precision@3=2/3 ->
    mean = (1 + 2/3) / 2 = 0.833.
    """
    monkeypatch.setattr(judge, "call_json", lambda prompt: {
        "claims": [{"claim": "a", "supported": True}],
        "passage_relevance": [True, False, True],
        "relevance": "RELEVANT",
        "reasoning": "test",
    })

    result = judge.judge_answer_quality(
        "q", [{"text": t, "metadata": {"ticker": "X", "sector": "y", "filing_date": "z"}} for t in ("p1", "p2", "p3")], "a"
    )

    assert result["context_precision"] == round((1 / 1 + 2 / 3) / 2, 3)


def test_judge_answer_quality_context_precision_zero_when_nothing_relevant(monkeypatch):
    monkeypatch.setattr(judge, "call_json", lambda prompt: {
        "claims": [{"claim": "a", "supported": True}],
        "passage_relevance": [False, False],
        "relevance": "PARTLY_RELEVANT",
        "reasoning": "test",
    })

    result = judge.judge_answer_quality("q", [{"text": "p1", "metadata": {"ticker": "X", "sector": "y", "filing_date": "z"}}], "a")

    assert result["context_precision"] == 0.0


def test_judge_answer_quality_passes_through_relevance_and_reasoning(monkeypatch):
    monkeypatch.setattr(judge, "call_json", lambda prompt: {
        "claims": [{"claim": "a", "supported": True}],
        "passage_relevance": [True],
        "relevance": "PARTLY_RELEVANT",
        "reasoning": "answers half the question",
    })

    result = judge.judge_answer_quality("q", [{"text": "p1", "metadata": {"ticker": "X", "sector": "y", "filing_date": "z"}}], "a")

    assert result["relevance"] == "PARTLY_RELEVANT"
    assert result["reasoning"] == "answers half the question"
