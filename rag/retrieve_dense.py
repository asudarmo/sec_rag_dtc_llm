"""Query the ChromaDB vector store and return top-k relevant chunks (dense retrieval).

Connects to the persistent collection built by ingestion/embed.py, turns the user's
question into a query embedding, runs a vector-similarity search, and returns the
top-k most relevant chunks (with metadata and distance). Optionally filters by
KEYWORD_FIELDS (ticker and/or sector — see rag/fields.py) via a `filter_dict`,
matching the DTC course's own minsearch terminology (01-agentic-rag/lessons/05-search.md).

Ported from finsignal-rag's src/retrieve.py; this is the "dense" half of the
hybrid retriever added in rag/hybrid.py (Day 2) — kept as its own module so it
stays swappable/comparable in the Day 3 retrieval evaluation.

CLI:
    uv run python -m rag.retrieve_dense "What are the main risk factors?" --k 5 --sector banks
    uv run python -m rag.retrieve_dense "What are Google's AI risks?" --k 5 --ticker GOOGL
"""

import argparse
import logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from rag.fields import SECTOR, to_chroma_where

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

CHROMA_DIR = Path(__file__).resolve().parents[1] / "chroma_db"
COLLECTION_NAME = "filings"
EMBED_MODEL = "all-MiniLM-L6-v2"  # must match ingestion/embed.py, or query vectors won't line up


def get_collection() -> chromadb.Collection:
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(f"Vector store {CHROMA_DIR} not found. Run python ingestion/embed.py first.")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)


def retrieve(query: str, k: int = 5, filter_dict: dict[str, str] | None = None) -> list[dict]:
    """Return the top-k most relevant chunks; each has id / text / metadata / distance (smaller = closer).

    `id` matches the chunk id assigned in ingestion/ingest.py (`{ticker}_{accession}_{i}`) —
    needed so rag/hybrid.py can match the same chunk across dense and sparse result lists.
    `filter_dict` restricts to exact matches on KEYWORD_FIELDS, e.g. {"sector": "banks"}
    or {"ticker": "GOOGL"} (or both together — see rag/fields.py for how these compose).
    """
    collection = get_collection()
    res = collection.query(query_texts=[query], n_results=k, where=to_chroma_where(filter_dict))
    hits = []
    for doc_id, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({"id": doc_id, "text": doc, "metadata": meta, "distance": dist})
    return hits


def retrieve_balanced(query: str, k: int = 6, sectors: tuple[str, ...] = ("tech", "banks")) -> list[dict]:
    """Retrieve an equal share of chunks from each sector, then combine.

    Naive top-k over the whole corpus is dominated by the larger-corpus sector
    (banks have ~3x more chunks than tech), so a cross-sector comparison query
    tends to retrieve bank-only passages. This forces a fixed quota per sector so
    both are always represented — the original (pre-Day-2) stopgap fix for the
    imbalance problem now properly addressed by hybrid search + query rewriting
    (rag/pipeline.py); kept here as a baseline for the Day 3 evaluation comparison.
    Hits are ordered by distance (closest first) after merging.
    """
    per_sector = max(1, k // len(sectors))
    hits: list[dict] = []
    for sec in sectors:
        hits.extend(retrieve(query, k=per_sector, filter_dict={"sector": sec}))
    hits.sort(key=lambda h: h["distance"])
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="dense vector retrieval test")
    parser.add_argument("query", help="the query")
    parser.add_argument("--k", type=int, default=5, help="how many to return (default 5)")
    parser.add_argument("--sector", choices=["tech", "banks"], help="restrict to one sector")
    parser.add_argument("--ticker", choices=sorted(SECTOR), help="restrict to one company")
    parser.add_argument("--balanced", action="store_true", help="retrieve an equal share from each sector")
    args = parser.parse_args()

    filter_dict = {k: v for k, v in [("sector", args.sector), ("ticker", args.ticker)] if v}
    hits = retrieve_balanced(args.query, k=args.k) if args.balanced else retrieve(args.query, k=args.k, filter_dict=filter_dict)
    print(f"\nQuery: {args.query} (k={args.k}, filter={filter_dict or 'none'})\n")
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        print(f"--- #{i} | {m['ticker']} ({m['sector']}) | distance {h['distance']:.3f} ---")
        print(h["text"][:300].strip(), "...\n")


if __name__ == "__main__":
    main()
