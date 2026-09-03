"""Cross-encoder reranking over a candidate set (typically hybrid.retrieve's pool),
for a higher-precision final ordering than embedding distance or BM25 score alone —
one of this project's Best Practices bonus items.

A cross-encoder scores (query, passage) pairs jointly, rather than independently
embedding each side (as dense retrieval does) — more accurate, but too slow to run
over the whole corpus, hence: retrieve a candidate pool cheaply first (dense/sparse/
hybrid), then rerank only that pool.

The model is loaded lazily and cached (module-level, via lru_cache) since it's the
slowest stage in the pipeline and not every retrieval config uses it — see the Day 3
evaluation sweep in PLAN.md, which compares configs with and without this stage.

CLI:
    uv run python -m rag.rerank "What are the main risk factors?" --k 5 --sector banks
"""

import argparse
import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder

from rag import fields, hybrid

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _get_model() -> CrossEncoder:
    log.info("Loading cross-encoder reranker %s...", MODEL_NAME)
    return CrossEncoder(MODEL_NAME)


def rerank(query: str, hits: list[dict], k: int = 5) -> list[dict]:
    """Score each hit's text against the query with a cross-encoder, return top-k re-sorted."""
    if not hits:
        return hits
    model = _get_model()
    pairs = [(query, h["text"]) for h in hits]
    scores = model.predict(pairs)
    reranked = [{**h, "rerank_score": float(s)} for h, s in zip(hits, scores)]
    reranked.sort(key=lambda h: h["rerank_score"], reverse=True)
    return reranked[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description="hybrid retrieval + cross-encoder rerank test")
    parser.add_argument("query", help="the query")
    parser.add_argument("--k", type=int, default=5, help="how many to return after reranking (default 5)")
    parser.add_argument("--pool", type=int, default=20, help="candidate pool size before reranking (default 20)")
    parser.add_argument("--sector", choices=["tech", "banks"], help="restrict to one sector")
    parser.add_argument("--ticker", choices=sorted(fields.SECTOR), help="restrict to one company")
    args = parser.parse_args()

    filter_dict = {k: v for k, v in [("sector", args.sector), ("ticker", args.ticker)] if v}
    candidates = hybrid.retrieve(args.query, k=args.pool, filter_dict=filter_dict)
    hits = rerank(args.query, candidates, k=args.k)
    print(f"\nQuery: {args.query} (k={args.k}, pool={args.pool}, filter={filter_dict or 'none'})\n")
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        print(f"--- #{i} | {m['ticker']} ({m['sector']}) | rerank_score {h['rerank_score']:.3f} ---")
        print(h["text"][:300].strip(), "...\n")


if __name__ == "__main__":
    main()
