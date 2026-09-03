"""Download 10-K annual reports from SEC EDGAR into data/raw/ using sec-edgar-downloader.

Ported from finsignal-rag (SMM694 Applied NLP assignment); logic unchanged.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from sec_edgar_downloader import Downloader

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Scoped down: enough for a short report, keeping a "tech vs banks" sector contrast
TICKERS = {
    "tech": ["GOOGL", "MSFT", "NVDA"],
    "banks": ["JPM", "GS", "BAC"],
}
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
FILING_TYPE = "10-K"
LIMIT = 1  # most recent 10-K per ticker (6 filings total)


def load_env() -> tuple[str, str]:
    load_dotenv()
    name = os.environ.get("SEC_USER_NAME", "").strip()
    email = os.environ.get("SEC_USER_EMAIL", "").strip()
    if not name or not email:
        raise RuntimeError(
            "SEC_USER_NAME and SEC_USER_EMAIL must be set in .env "
            "(required by SEC fair-access policy)"
        )
    return name, email


def build_downloader(name: str, email: str) -> Downloader:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return Downloader(name, email, download_folder=str(RAW_DIR))


def download_all(dl: Downloader) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {"ok": [], "failed": []}
    all_tickers = [t for group in TICKERS.values() for t in group]
    for ticker in all_tickers:
        try:
            log.info("Downloading %s for %s (limit=%d)...", FILING_TYPE, ticker, LIMIT)
            dl.get(FILING_TYPE, ticker, limit=LIMIT)
            result["ok"].append(ticker)
        except Exception as exc:
            log.warning("Failed %s: %s", ticker, exc)
            result["failed"].append(ticker)
    return result


def main() -> None:
    name, email = load_env()
    dl = build_downloader(name, email)
    summary = download_all(dl)
    ok, failed = len(summary["ok"]), len(summary["failed"])
    log.info("Download complete. %d succeeded, %d failed. Files in %s", ok, failed, RAW_DIR)
    if summary["failed"]:
        log.warning("Failed tickers: %s", summary["failed"])


if __name__ == "__main__":
    main()
