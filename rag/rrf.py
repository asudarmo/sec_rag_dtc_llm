"""Generic Reciprocal Rank Fusion (RRF): merge N ranked lists into one fused ranking.

Matches the DTC LLM Zoomcamp course's own implementation (06-best-practices,
lessons/02-hybrid-search.md and 03-reranking.md) almost exactly — same formula,
same k=60 constant — generalized here to N ranked lists rather than hardcoded to 2
(the course's own function already takes a list of ranked lists, not just two), so
it's reusable for two structurally different fusion problems in this project:

  - rag/hybrid.py: fuse dense + sparse rankings for a SINGLE query ("hybrid search").
  - rag/pipeline.py: fuse per-sub-query rankings when query rewriting produces
    multiple queries. This is the course's actual "reranking" technique (RRF
    reordering results drawn from more than one ranked list) — a genuinely
    different fusion problem from hybrid search (fusing across queries, not across
    retrieval methods), so it gives different results, not just a relabeled copy of
    hybrid.py. See README/PLAN.md for how this compares to rag/rerank.py's
    cross-encoder, which the course's own reranking lesson does not use at all.
"""

RRF_K = 60


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """Fuse multiple ranked lists of hit dicts (each needs an "id" key) into one list,
    sorted by fused rrf_score descending. A hit appearing in more than one list gets
    its per-list RRF contributions (1 / (k + rank)) summed, boosting docs ranked
    consistently across lists over docs that only appear in a single list.
    """
    scores: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            doc_id = hit["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(doc_id, hit)

    fused = [{**by_id[doc_id], "rrf_score": score} for doc_id, score in scores.items()]
    fused.sort(key=lambda h: h["rrf_score"], reverse=True)
    return fused
