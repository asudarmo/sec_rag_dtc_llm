from eval.generate_ground_truth import sample_chunks


def test_sample_chunks_is_stratified_by_ticker():
    chunks = (
        [{"id": f"GOOGL_{i}", "ticker": "GOOGL"} for i in range(100)]
        + [{"id": f"JPM_{i}", "ticker": "JPM"} for i in range(300)]  # 3x more, like the real corpus
    )

    sampled = sample_chunks(chunks, per_ticker=25, seed=42)

    tickers = [c["ticker"] for c in sampled]
    assert tickers.count("GOOGL") == 25
    assert tickers.count("JPM") == 25  # NOT 3x GOOGL's count, despite the corpus imbalance
    assert len(sampled) == 50


def test_sample_chunks_is_deterministic_for_a_given_seed():
    chunks = [{"id": f"GOOGL_{i}", "ticker": "GOOGL"} for i in range(50)]

    first = sample_chunks(chunks, per_ticker=10, seed=42)
    second = sample_chunks(chunks, per_ticker=10, seed=42)

    assert [c["id"] for c in first] == [c["id"] for c in second]


def test_sample_chunks_caps_at_available_count():
    chunks = [{"id": f"NVDA_{i}", "ticker": "NVDA"} for i in range(5)]

    sampled = sample_chunks(chunks, per_ticker=25, seed=42)

    assert len(sampled) == 5  # doesn't error or duplicate when fewer than per_ticker exist
