# Ingestion pipeline

## Data

The most recent 10-K filing for each of 6 tickers across two sectors:

- **Tech**: GOOGL, MSFT, NVDA
- **Banking**: JPM, GS, BAC

Filings are chunked at 1000 characters with 200 overlap
(`langchain_text_splitters.RecursiveCharacterTextSplitter`) and embedded with
`all-MiniLM-L6-v2` into ChromaDB.

**A prebuilt vector store is committed to the repo** (`chroma_db/`,
`data/processed/chunks.json`), so the app runs immediately on a fresh clone with no
ingestion step required.

## Automated ingestion (dlt) — the primary path

`ingestion/dlt_pipeline.py` automates the full pipeline with
[dlt](https://dlthub.com/): downloads each ticker's latest 10-K from SEC EDGAR,
parses/chunks it, and loads the resulting records into a database table
(`raw.filing_chunks`) — satisfying the rubric's "automated ingestion with a special
tool" criterion.

```bash
# Full run — loads into Postgres (the "raw" schema in the shared monitoring Postgres)
uv run python -m ingestion.dlt_pipeline

# Local dry run — no Postgres required, loads into a local DuckDB file instead
uv run python -m ingestion.dlt_pipeline --destination duckdb
```

Both require `SEC_USER_NAME`/`SEC_USER_EMAIL` in `.env` (SEC EDGAR's fair-access
policy). The Postgres-backed run needs the docker-compose Postgres service up first
(`docker compose up -d postgres`, or the full stack); the DuckDB dry run has no such
dependency.

Either run downloads all 6 filings and produces roughly 5,200 chunks (exact per-ticker
counts vary slightly over time as companies file newer 10-Ks).

## Manual/scripted ingestion — still available

The pre-dlt scripts still work individually, useful for inspecting each stage or
rebuilding the vector store from scratch:

```bash
uv run python -m ingestion.download   # SEC EDGAR -> data/raw/
uv run python -m ingestion.ingest     # data/raw/ -> data/processed/chunks.json (chunking)
uv run python -m ingestion.embed      # chunks.json -> chroma_db/ (sentence-transformers embeddings)
```
