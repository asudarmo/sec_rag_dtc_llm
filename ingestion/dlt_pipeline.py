"""Automated SEC 10-K ingestion via dlt: download -> parse/chunk -> Postgres staging table.

This is what upgrades ingestion from "semi-automated script" to "automated ingestion
with a specialized tool" on the DTC rubric. It doesn't reimplement the download/parse
logic — it wraps the same functions from ingestion/download.py and ingestion/ingest.py
in a dlt resource, so there's exactly one place that logic lives.

Destination defaults to Postgres (the same shared container the monitoring stack uses
— see .env / PLAN.md), loading into the `raw` dataset/schema, table `filing_chunks`.
For a quick local dry run without a running Postgres, pass --destination duckdb.

ingestion/embed.py currently reads data/processed/chunks.json directly; once this
pipeline is the primary ingestion path, embed.py can instead read the `filing_chunks`
table from Postgres (tracked in PLAN.md) — kept as a follow-up so this file stays
focused on ingestion, not retrieval-side wiring.

CLI:
    uv run python -m ingestion.dlt_pipeline                    # loads into Postgres
    uv run python -m ingestion.dlt_pipeline --destination duckdb  # local dry run, no Postgres needed
"""

import argparse
import logging
import os

import dlt
from dotenv import load_dotenv

from ingestion.download import FILING_TYPE, LIMIT, TICKERS, build_downloader, load_env
from ingestion.ingest import RAW_DIR, parse_filing

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def get_postgres_credentials() -> str:
    load_dotenv()
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "sec_rag")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


@dlt.resource(name="filing_chunks", write_disposition="replace")
def filing_chunks():
    """Download the 6 tracked tickers' latest 10-K, then yield one record per chunk."""
    name, email = load_env()
    dl = build_downloader(name, email)
    all_tickers = [t for group in TICKERS.values() for t in group]
    for ticker in all_tickers:
        try:
            log.info("Downloading %s for %s (limit=%d)...", FILING_TYPE, ticker, LIMIT)
            dl.get(FILING_TYPE, ticker, limit=LIMIT)
        except Exception as exc:
            log.warning("Download failed for %s: %s", ticker, exc)

    for txt_path in sorted(RAW_DIR.glob("*/10-K/*/full-submission.txt")):
        ticker = txt_path.parts[txt_path.parts.index("sec-edgar-filings") + 1]
        try:
            yield from parse_filing(txt_path, ticker)
        except Exception as exc:
            log.warning("Parse failed for %s: %s", txt_path, exc)


@dlt.source
def sec_filings_source():
    return filing_chunks()


def run(destination: str = "postgres", dataset_name: str = "raw") -> None:
    dest = dlt.destinations.postgres(credentials=get_postgres_credentials()) if destination == "postgres" else destination
    pipeline = dlt.pipeline(
        pipeline_name="sec_filings_ingestion",
        destination=dest,
        dataset_name=dataset_name,
    )
    load_info = pipeline.run(sec_filings_source())
    print(load_info)


def main() -> None:
    parser = argparse.ArgumentParser(description="dlt-orchestrated SEC 10-K ingestion")
    parser.add_argument(
        "--destination",
        default="postgres",
        help="dlt destination (default: postgres; use 'duckdb' for a local dry run without Postgres)",
    )
    parser.add_argument("--dataset-name", default="raw", help="dlt dataset/schema name (default: raw)")
    args = parser.parse_args()
    run(destination=args.destination, dataset_name=args.dataset_name)


if __name__ == "__main__":
    main()
