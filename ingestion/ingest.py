"""Parse raw SEC 10-K filings into text chunks and save to data/processed/.

Ported from finsignal-rag; logic unchanged. `parse_filing` is also reused directly
by `dlt_pipeline.py` so the dlt-orchestrated ingestion and this standalone script
stay in sync (one chunking implementation, two ways to run it).
"""

import json
import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "sec-edgar-filings"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUT_FILE = OUT_DIR / "chunks.json"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

_DOC_PATTERN = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.DOTALL)
_PERIOD_PATTERN = re.compile(r"CONFORMED PERIOD OF REPORT:\s*(\d+)")


def extract_html(path: Path) -> str:
    """Return the HTML text of the first <DOCUMENT> block in an SGML filing."""
    content = path.read_text(encoding="utf-8", errors="replace")
    match = _DOC_PATTERN.search(content)
    if not match:
        raise ValueError(f"No <DOCUMENT> block found in {path}")
    return match.group(1)


def extract_filing_date(path: Path) -> str:
    """Return the CONFORMED PERIOD OF REPORT value from the SGML header."""
    header_text = path.read_text(encoding="utf-8", errors="replace")[:4000]
    match = _PERIOD_PATTERN.search(header_text)
    return match.group(1) if match else "unknown"


def clean_text(html: str) -> str:
    """Strip HTML tags and return plain text, excluding XBRL metadata noise."""
    soup = BeautifulSoup(html, "html.parser")
    # Remove <head> (XBRL context definitions live here)
    for tag in soup.find_all("head"):
        tag.decompose()
    # Remove inline XBRL header blocks (ix:header / ixv:header)
    for tag in soup.find_all(re.compile(r"ix:header|ixv:header", re.I)):
        tag.decompose()
    # Remove hidden divs (SEC filings hide XBRL context data in display:none blocks)
    for tag in soup.find_all("div", style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def chunk_text(
    text: str, ticker: str, accession: str, filing_date: str, source: str
) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    pieces = splitter.split_text(text)
    return [
        {
            "id": f"{ticker}_{accession}_{i}",
            "ticker": ticker,
            "accession": accession,
            "filing_date": filing_date,
            "text": piece,
            "source": source,
        }
        for i, piece in enumerate(pieces)
    ]


def parse_filing(txt_path: Path, ticker: str) -> list[dict]:
    accession = txt_path.parent.name
    filing_date = extract_filing_date(txt_path)
    html = extract_html(txt_path)
    text = clean_text(html)
    # Store a repo-relative path only — never the absolute local path, which
    # would leak the machine username / account into committed data.
    source = str(txt_path.relative_to(Path(__file__).resolve().parents[1]))
    return chunk_text(text, ticker, accession, filing_date, source)


def ingest_all() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks: list[dict] = []
    parsed, failed = 0, 0

    for txt_path in sorted(RAW_DIR.glob("*/10-K/*/full-submission.txt")):
        ticker = txt_path.parts[txt_path.parts.index("sec-edgar-filings") + 1]
        try:
            chunks = parse_filing(txt_path, ticker)
            all_chunks.extend(chunks)
            log.info("Parsed %s/%s → %d chunks", ticker, txt_path.parent.name, len(chunks))
            parsed += 1
        except Exception as exc:
            log.warning("Failed %s: %s", txt_path, exc)
            failed += 1

    OUT_FILE.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(
        "%d filings parsed, %d failed. %d chunks saved to %s",
        parsed, failed, len(all_chunks), OUT_FILE,
    )


def main() -> None:
    ingest_all()


if __name__ == "__main__":
    main()
