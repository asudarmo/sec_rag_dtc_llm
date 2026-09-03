from rag import pipeline


def test_multi_query_rewrite_uses_rrf_not_naive_union(monkeypatch):
    """Regression test for the bug this was built to fix: before, a multi-query
    rewrite was merged with a plain dict union (first occurrence wins, no re-ranking
    by combined evidence). A doc appearing in both sub-queries' results should now
    outrank one appearing in only a single sub-query's results, even if that single
    appearance was ranked #1 there.
    """
    per_sub_query_hits = {
        "sub-query 1": [
            {"id": "only-in-q1", "distance": 0.01},  # top hit of q1, but nowhere else
            {"id": "shared", "distance": 0.20},
        ],
        "sub-query 2": [
            {"id": "shared", "distance": 0.05},
            {"id": "only-in-q2", "distance": 0.30},
        ],
    }

    monkeypatch.setattr(
        pipeline.query_rewrite, "rewrite",
        lambda query: [{"query": q, "filter_dict": None} for q in per_sub_query_hits],
    )
    monkeypatch.setattr(
        pipeline.retrieve_dense, "retrieve",
        lambda query, k, filter_dict=None: per_sub_query_hits[query],
    )

    hits = pipeline.retrieve("irrelevant combined query", k=3, method="dense", use_query_rewrite=True)
    result_ids = [h["id"] for h in hits]

    assert result_ids[0] == "shared"  # appears in both lists -> RRF-fused score wins
    assert set(result_ids) == {"only-in-q1", "shared", "only-in-q2"}


def test_multi_query_rewrite_scopes_each_sub_query_to_its_own_filter(monkeypatch):
    """Regression test for the filter-routing gap: a decomposed sub-query must be
    retrieved with ITS OWN filter_dict, not the caller's (or none) — otherwise a
    sub-query's results can still be crowded out by an unrelated sector/company's
    chunks. Covers both granularities query_rewrite can produce: sector-level and
    ticker-level (company-level) filters.
    """
    monkeypatch.setattr(
        pipeline.query_rewrite, "rewrite",
        lambda query: [
            {"query": "tech sub-query", "filter_dict": {"sector": "tech"}},
            {"query": "banks sub-query", "filter_dict": {"sector": "banks"}},
        ],
    )
    seen_filters = []

    def fake_retrieve(query, k, filter_dict=None):
        seen_filters.append(filter_dict)
        return [{"id": f"{filter_dict}-hit", "distance": 0.1}]

    monkeypatch.setattr(pipeline.retrieve_dense, "retrieve", fake_retrieve)

    pipeline.retrieve("compare tech vs banks", k=2, method="dense", use_query_rewrite=True)

    assert {"sector": "tech"} in seen_filters
    assert {"sector": "banks"} in seen_filters


def test_multi_query_rewrite_supports_ticker_level_filters(monkeypatch):
    monkeypatch.setattr(
        pipeline.query_rewrite, "rewrite",
        lambda query: [
            {"query": "Google sub-query", "filter_dict": {"ticker": "GOOGL"}},
            {"query": "Microsoft sub-query", "filter_dict": {"ticker": "MSFT"}},
        ],
    )
    seen_filters = []

    def fake_retrieve(query, k, filter_dict=None):
        seen_filters.append(filter_dict)
        return [{"id": f"{filter_dict}-hit", "distance": 0.1}]

    monkeypatch.setattr(pipeline.retrieve_dense, "retrieve", fake_retrieve)

    pipeline.retrieve("compare Google vs Microsoft", k=2, method="dense", use_query_rewrite=True)

    assert {"ticker": "GOOGL"} in seen_filters
    assert {"ticker": "MSFT"} in seen_filters


def test_multi_query_rewrite_reranks_each_sub_query_independently_for_balance(monkeypatch):
    """Regression test for a live-reported bug: reranking the globally-fused pool
    can re-skew results back toward whichever sub-query's content scores higher for
    general relevance, even when the underlying retrieval was balanced going in
    ("compare tech vs banks" returned 5 bank passages vs 1 tech despite both rewrite
    and rerank being on). Reranking each sub-query's own pool independently and
    taking a fixed quota from each guarantees the final result stays balanced
    regardless of how the reranker scores things.
    """
    tech_candidates = [{"id": f"tech-{i}", "distance": 0.1 * i} for i in range(5)]
    bank_candidates = [{"id": f"bank-{i}", "distance": 0.1 * i} for i in range(5)]

    monkeypatch.setattr(
        pipeline.query_rewrite, "rewrite",
        lambda query: [
            {"query": "tech sub-query", "filter_dict": {"sector": "tech"}},
            {"query": "banks sub-query", "filter_dict": {"sector": "banks"}},
        ],
    )
    monkeypatch.setattr(
        pipeline.retrieve_dense, "retrieve",
        lambda query, k, filter_dict=None: tech_candidates if filter_dict == {"sector": "tech"} else bank_candidates,
    )

    rerank_calls = []

    def fake_rerank(query, hits, k):
        rerank_calls.append(hits)
        return hits[:k]  # deterministic: just take the first k of whatever it's given

    monkeypatch.setattr(pipeline.rerank_module, "rerank", fake_rerank)

    hits = pipeline.retrieve(
        "compare tech vs banks", k=6, method="dense",
        use_query_rewrite=True, use_rerank=True,
    )

    # Reranked each sub-query's own pool separately, not a merged/fused pool.
    assert len(rerank_calls) == 2
    assert {h["id"] for h in rerank_calls[0]} == {c["id"] for c in tech_candidates}
    assert {h["id"] for h in rerank_calls[1]} == {c["id"] for c in bank_candidates}

    # Final result is balanced: 3 from each side for k=6, n=2 sub-queries.
    result_ids = [h["id"] for h in hits]
    assert sum(1 for i in result_ids if i.startswith("tech-")) == 3
    assert sum(1 for i in result_ids if i.startswith("bank-")) == 3


def test_single_query_skips_fusion_and_uses_base_retriever_order(monkeypatch):
    hits = [{"id": "a", "distance": 0.1}, {"id": "b", "distance": 0.2}]
    monkeypatch.setattr(pipeline.retrieve_dense, "retrieve", lambda query, k, filter_dict=None: hits)

    result = pipeline.retrieve("a normal single-topic question", k=2, method="dense")

    assert [h["id"] for h in result] == ["a", "b"]
