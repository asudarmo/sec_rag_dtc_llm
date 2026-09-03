"""Single retrieval entry point wiring dense/sparse/hybrid retrieval plus optional
reranking and query rewriting behind config flags.

This exists so Day 3's evaluation sweep (retrieval Hit Rate/MRR across configs, and
the LLM-judge generation comparison) can select any combination of stages by just
passing different arguments to one function, rather than each combination needing
its own hand-wired call chain. rag/generate.py's answer() calls this as its retrieval
step.
"""

import logging

from rag import hybrid, query_rewrite, retrieve_dense, retrieve_sparse
from rag import rerank as rerank_module
from rag.rrf import reciprocal_rank_fusion

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Modules, not bound functions — resolved via getattr at call time (like hybrid.py
# does for retrieve_dense/retrieve_sparse) so tests can monkeypatch each module's
# .retrieve independently instead of it being frozen in a dict at import time.
_BASE_MODULES = {
    "dense": retrieve_dense,
    "sparse": retrieve_sparse,
    "hybrid": hybrid,
}


def retrieve(
    query: str,
    k: int = 5,
    filter_dict: dict[str, str] | None = None,
    method: str = "hybrid",
    use_rerank: bool = False,
    use_query_rewrite: bool = False,
    pool: int = 20,
) -> list[dict]:
    """Retrieve top-k chunks for `query` using the given method/stage combination.

    method: base retriever — "dense", "sparse", or "hybrid" (RRF-fused dense+sparse).
    filter_dict: exact-match filter on KEYWORD_FIELDS (rag/fields.py) — e.g.
        {"sector": "banks"} or {"ticker": "GOOGL"}.
    use_rerank: apply the cross-encoder reranker (rag/rerank.py) to the candidate pool.
    use_query_rewrite: rewrite/decompose the query first (rag/query_rewrite.py). A
        decomposed (multi-sub-query) rewrite retrieves each sub-query separately —
        scoped to that sub-query's own filter_dict when the rewrite tagged one (a
        sector for a sector-level comparison, or a ticker for a company-level one),
        since a sub-query with no filter would still search the whole corpus and
        could still be crowded out by a larger sector/company, defeating the point
        of splitting the query. Without reranking, the per-sub-query pools are
        RRF-fused (rag/rrf.py) — symmetric across sub-queries, so this stays
        balanced. WITH reranking, each sub-query's pool is reranked independently
        and a fixed quota is taken from each, rather than reranking the merged pool
        and taking a flat top-k — the latter re-scores everything against general
        relevance to the whole (undecomposed) question and can silently re-skew the
        result back toward whichever sub-query's content happens to score higher
        (e.g. a sector with more extensive matching language in the corpus),
        undoing the balance the sub-querying was meant to guarantee. Found via a
        live report: "compare tech vs banks" with both rewrite and rerank on still
        returned 5 bank passages vs 1 tech, despite the retrieval pools themselves
        being balanced going into the (then-global) rerank step.
    pool: candidate pool size used when rewriting and/or reranking need more than k
        candidates to work with (ignored otherwise — base retrievers just return k).
    """
    if method not in _BASE_MODULES:
        raise ValueError(f"Unknown retrieval method: {method!r} (expected one of {list(_BASE_MODULES)})")
    base_retrieve = _BASE_MODULES[method].retrieve

    # Uniform shape whether or not rewriting ran: a list of {"query", "filter_dict"}
    # items. A sub-query's own filter_dict (set by query_rewrite for a decomposed
    # comparison) takes priority; None/empty means "defer to the caller's filter_dict"
    # (the no-rewrite case, and a rewrite that judged the question single-topic).
    items = query_rewrite.rewrite(query) if use_query_rewrite else [{"query": query, "filter_dict": filter_dict}]
    per_query_k = pool if (use_rerank or len(items) > 1) else k

    if len(items) == 1:
        it = items[0]
        hits = base_retrieve(it["query"], k=per_query_k, filter_dict=it["filter_dict"] or filter_dict)  # already sorted
        return rerank_module.rerank(query, hits, k=k) if use_rerank else hits[:k]

    per_query_lists = [
        base_retrieve(it["query"], k=per_query_k, filter_dict=it["filter_dict"] or filter_dict) for it in items
    ]

    if not use_rerank:
        return reciprocal_rank_fusion(per_query_lists)[:k]

    # Balance-preserving reranking: quota per sub-query sums to exactly k, any
    # remainder going to the first sub-queries (arbitrary but deterministic).
    n = len(items)
    base_quota, remainder = divmod(k, n)
    quotas = [base_quota + (1 if i < remainder else 0) for i in range(n)]
    hits: list[dict] = []
    for candidates, quota in zip(per_query_lists, quotas):
        hits.extend(rerank_module.rerank(query, candidates, k=quota))
    return hits
