"""BM25 sparse retrieval over the same chunk corpus used for dense embeddings.

Replaces the old TF-IDF baseline (finsignal-rag's retrieve_tfidf.py) — BM25 is the
standard keyword-search scoring function for hybrid search (matches the course's
terminology) and needs no fitted vectorizer to persist alongside the vector store.

Builds an in-memory BM25 index from data/processed/chunks.json once per process and
caches it (module-level, via lru_cache) — re-tokenizing ~5000 chunks on every query
would be wasteful for an interactive app.

Filtering by KEYWORD_FIELDS (ticker and/or sector — see rag/fields.py) is a manual
post-scoring predicate here, since BM25 has no native filter mechanism the way
ChromaDB's `where` does; same `filter_dict` contract as the other retrievers though,
matching the DTC course's own minsearch terminology (01-agentic-rag/lessons/05-search.md).

CLI:
    uv run python -m rag.retrieve_sparse "What are the main risk factors?" --k 5 --sector banks
    uv run python -m rag.retrieve_sparse "What are Google's AI risks?" --k 5 --ticker GOOGL
"""

import argparse
import json
import logging
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

from rag.fields import SECTOR, matches_filter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

CHUNKS_FILE = Path(__file__).resolve().parents[1] / "data" / "processed" / "chunks.json"


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class _Bm25Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])


@lru_cache(maxsize=1)
def _get_index() -> _Bm25Index:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"{CHUNKS_FILE} not found. Run ingestion first (see README).")
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    log.info("Building BM25 index over %d chunks...", len(chunks))
    return _Bm25Index(chunks)


def _metadata(chunk: dict) -> dict:
    return {
        "ticker": chunk["ticker"],
        "sector": SECTOR.get(chunk["ticker"], "unknown"),
        "accession": chunk["accession"],
        "filing_date": chunk["filing_date"],
        "source": chunk["source"],
    }


def retrieve(query: str, k: int = 5, filter_dict: dict[str, str] | None = None) -> list[dict]:
    """Return the top-k chunks by BM25 score; each has id / text / metadata / score (higher = better).
    `filter_dict` restricts to exact matches on KEYWORD_FIELDS, e.g. {"sector": "banks"}
    or {"ticker": "GOOGL"} (or both together — see rag/fields.py for how these compose).
    """
    index = _get_index()
    scores = index.bm25.get_scores(_tokenize(query))

    candidates = [
        (i, s) for i, s in enumerate(scores)
        if matches_filter(_metadata(index.chunks[i]), filter_dict)
    ]
    candidates.sort(key=lambda pair: pair[1], reverse=True)

    hits = []
    for i, score in candidates[:k]:
        c = index.chunks[i]
        hits.append({"id": c["id"], "text": c["text"], "metadata": _metadata(c), "score": float(score)})
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 sparse retrieval test")
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
        print(f"--- #{i} | {m['ticker']} ({m['sector']}) | score {h['score']:.3f} ---")
        print(h["text"][:300].strip(), "...\n")


if __name__ == "__main__":
    main()
