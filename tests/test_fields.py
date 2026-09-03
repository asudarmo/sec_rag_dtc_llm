from rag.fields import matches_filter, to_chroma_where


def test_matches_filter_none_or_empty_matches_everything():
    assert matches_filter({"ticker": "GOOGL"}, None) is True
    assert matches_filter({"ticker": "GOOGL"}, {}) is True


def test_matches_filter_single_field():
    metadata = {"ticker": "GOOGL", "sector": "tech"}
    assert matches_filter(metadata, {"sector": "tech"}) is True
    assert matches_filter(metadata, {"sector": "banks"}) is False


def test_matches_filter_multiple_fields_is_and():
    metadata = {"ticker": "GOOGL", "sector": "tech"}
    assert matches_filter(metadata, {"sector": "tech", "ticker": "GOOGL"}) is True
    assert matches_filter(metadata, {"sector": "tech", "ticker": "MSFT"}) is False


def test_to_chroma_where_none_or_empty():
    assert to_chroma_where(None) is None
    assert to_chroma_where({}) is None


def test_to_chroma_where_single_field_is_flat():
    assert to_chroma_where({"sector": "tech"}) == {"sector": "tech"}


def test_to_chroma_where_multiple_fields_uses_and():
    # ChromaDB requires this exact shape for multi-condition filters (verified
    # empirically against the real collection — see rag/fields.py docstring).
    result = to_chroma_where({"sector": "tech", "ticker": "GOOGL"})
    assert result == {"$and": [{"sector": "tech"}, {"ticker": "GOOGL"}]}
