from ingestion.ingest import chunk_text


def test_chunk_ids_are_sequential_and_scoped_to_ticker_accession():
    text = "word " * 500  # long enough to split into multiple chunks at CHUNK_SIZE=1000
    chunks = chunk_text(text, ticker="GOOGL", accession="0001-25-000123", filing_date="20250101", source="data/raw/x.txt")

    assert len(chunks) > 1  # confirms splitting actually happened, not a single-chunk edge case
    for i, c in enumerate(chunks):
        assert c["id"] == f"GOOGL_0001-25-000123_{i}"


def test_chunk_ids_are_unique_within_one_filing():
    text = "word " * 500
    chunks = chunk_text(text, ticker="MSFT", accession="acc-1", filing_date="20250101", source="src")
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_ids_dont_collide_across_tickers_or_accessions_at_the_same_index():
    # Same index (0 for a short single-chunk text) but different ticker/accession
    # must still produce different ids -> retrieval/dedup (e.g. rag/agent.py's
    # all_hits keyed on "id") relies on global uniqueness across the whole corpus,
    # not just within one filing.
    text = "short text, no splitting needed"
    same_ticker_diff_accession = chunk_text(text, ticker="GOOGL", accession="acc-2", filing_date="d", source="s")
    diff_ticker_same_accession = chunk_text(text, ticker="MSFT", accession="acc-1", filing_date="d", source="s")
    baseline = chunk_text(text, ticker="GOOGL", accession="acc-1", filing_date="d", source="s")

    assert baseline[0]["id"] != same_ticker_diff_accession[0]["id"]
    assert baseline[0]["id"] != diff_ticker_same_accession[0]["id"]


def test_chunk_metadata_fields_propagate_to_every_chunk():
    text = "word " * 500
    chunks = chunk_text(text, ticker="NVDA", accession="acc-9", filing_date="20250615", source="data/raw/nvda.txt")

    assert len(chunks) > 1
    for c in chunks:
        assert c["ticker"] == "NVDA"
        assert c["accession"] == "acc-9"
        assert c["filing_date"] == "20250615"
        assert c["source"] == "data/raw/nvda.txt"
        assert isinstance(c["text"], str) and c["text"]
