"""Hybrid retrieval: fuse dense (ChromaDB) and sparse (BM25) rankings with
Reciprocal Rank Fusion (RRF) — one of this project's Best Practices bonus items.

RRF combines ranked lists using only rank position, not raw scores — see
rag/rrf.py for the fusion algorithm itself (shared with rag/pipeline.py's
multi-query fusion; see that module and README/PLAN.md for how the two differ).
Rank-based fusion sidesteps the scale mismatch between cosine distance (dense) and
BM25 score (sparse): there's no shared scale to normalize, so score-level fusion
would need tuning per corpus, while rank-level fusion just works.

CLI:
    uv run python -m rag.hybrid "What are the main risk factors?" --k 5 --sector banks
"""

import argparse
import logging

from rag import retrieve_dense, retrieve_sparse
from rag.fields import SECTOR
from rag.rrf import reciprocal_rank_fusion

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Cast a wider net than the final k so fusion has enough candidates from each side
# to actually combine, rather than just returning whichever retriever's top-k first.
CANDIDATE_POOL = 20


def retrieve(query: str, k: int = 5, filter_dict: dict[str, str] | None = None, pool: int = CANDIDATE_POOL) -> list[dict]:
    """Fuse dense + BM25 rankings via RRF and return the top-k chunks (by fused rrf_score).
    `filter_dict` is passed through unchanged to both base retrievers — see rag/fields.py.
    """
    dense_hits = retrieve_dense.retrieve(query, k=pool, filter_dict=filter_dict)
    sparse_hits = retrieve_sparse.retrieve(query, k=pool, filter_dict=filter_dict)
    return reciprocal_rank_fusion([dense_hits, sparse_hits])[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description="hybrid (dense + BM25, RRF-fused) retrieval test")
    parser.add_argument("query", help="the query")
    parser.add_argument("--k", type=int, default=5, help="how many to return (default 5)")
    parser.add_argument("--sector", choices=["tech", "banks"], help="restrict to one sector")
    parser.add_argument("--ticker", choices=sorted(SECTOR), help="restrict to one company")
    args = parser.parse_args()

    filter_dict = {k: v for k, v in [("sector", args.sector), ("ticker", args.ticker)] if v}
    hits = retrieve(args.query, k=args.k, filter_dict=filter_dict)
    print(f"\nQuery: {args.query} (k={args.k}, filter={filter_dict or 'none'})\n")
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        print(f"--- #{i} | {m['ticker']} ({m['sector']}) | rrf_score {h['rrf_score']:.5f} ---")
        print(h["text"][:300].strip(), "...\n")


if __name__ == "__main__":
    main()
