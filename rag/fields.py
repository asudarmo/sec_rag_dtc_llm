"""Filter-field vocabulary, deliberately named after the DTC course's own minsearch
API (01-agentic-rag/lessons/05-search.md): a TEXT_FIELD is tokenized/embedded and
ranked by relevance; a KEYWORD_FIELD is an exact-match filter, applied via a
`filter_dict` — same idea as minsearch's `Index(text_fields=..., keyword_fields=...)`
+ `.search(query, filter_dict=...)`. Every retriever in rag/ (dense, sparse, hybrid,
pipeline) takes the same `filter_dict: dict[str, str] | None` parameter end to end,
so this is the one place that maps our two exact-match fields onto each backend's
own filtering mechanism (ChromaDB's `where`, or a hand-rolled BM25 predicate).

For our corpus, there's exactly one text field (the chunk text — embedded for dense
retrieval, tokenized for BM25) and two keyword fields: ticker (e.g. "GOOGL") and
sector (e.g. "tech", derived from ticker via SECTOR below). A caller can filter by
either, or both together (AND semantics) — e.g. {"sector": "tech"} for a sector-wide
question, {"ticker": "GOOGL"} for a single-company one.
"""

TEXT_FIELDS = ("text",)
KEYWORD_FIELDS = ("ticker", "sector")

# ticker -> sector. Single source of truth — was previously duplicated in
# ingestion/embed.py and rag/retrieve_sparse.py, which is exactly the kind of drift
# risk this module exists to remove.
SECTOR = {
    "GOOGL": "tech", "MSFT": "tech", "NVDA": "tech",
    "JPM": "banks", "GS": "banks", "BAC": "banks",
}


def matches_filter(metadata: dict, filter_dict: dict[str, str] | None) -> bool:
    """Exact-match AND predicate over KEYWORD_FIELDS — used by rag/retrieve_sparse.py,
    since BM25 has no native filtering mechanism, only a manual post-scoring check.
    """
    if not filter_dict:
        return True
    return all(metadata.get(field) == value for field, value in filter_dict.items())


def to_chroma_where(filter_dict: dict[str, str] | None) -> dict | None:
    """Translate a generic filter_dict into ChromaDB's `where` clause.

    ChromaDB requires multi-condition filters to be wrapped in "$and" explicitly —
    verified empirically: a flat multi-key dict raises "Expected where to have
    exactly one operator". A single-condition filter can be passed directly.
    """
    if not filter_dict:
        return None
    if len(filter_dict) == 1:
        return dict(filter_dict)
    return {"$and": [{field: value} for field, value in filter_dict.items()]}
