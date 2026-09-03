"""Embed text chunks and store them in ChromaDB.

Reads the chunks.json produced by ingest.py (or dlt_pipeline.py), computes
embeddings with a local sentence-transformers model (all-MiniLM-L6-v2, free and
offline), and stores them in a persistent ChromaDB collection under chroma_db/.
No API key required. Ported from finsignal-rag; logic unchanged.
"""

import json
import logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from rag.fields import SECTOR

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

CHUNKS_FILE = Path(__file__).resolve().parents[1] / "data" / "processed" / "chunks.json"
CHROMA_DIR = Path(__file__).resolve().parents[1] / "chroma_db"
COLLECTION_NAME = "filings"
EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 256


def load_chunks() -> list[dict]:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"{CHUNKS_FILE} not found. Run python ingestion/ingest.py first.")
    return json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))


def get_collection() -> chromadb.Collection:
    """Create (or recreate) the persistent collection with a sentence-transformers embedder."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    # Drop the old collection on re-run so chunks don't stack up
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
        log.info("Deleted existing collection '%s'", COLLECTION_NAME)
    return client.create_collection(name=COLLECTION_NAME, embedding_function=ef)


def embed_all() -> None:
    chunks = load_chunks()
    log.info("Loaded %d chunks, embedding now (model: %s)...", len(chunks), EMBED_MODEL)
    collection = get_collection()

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "ticker": c["ticker"],
                    "sector": SECTOR.get(c["ticker"], "unknown"),
                    "accession": c["accession"],
                    "filing_date": c["filing_date"],
                    "source": c["source"],
                }
                for c in batch
            ],
        )
        log.info("Embedded %d / %d", min(start + BATCH_SIZE, len(chunks)), len(chunks))

    log.info("Done. Collection '%s' holds %d items, stored at %s",
             COLLECTION_NAME, collection.count(), CHROMA_DIR)


def main() -> None:
    embed_all()


if __name__ == "__main__":
    main()
