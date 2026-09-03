"""LLM-based query rewriting: expand/decompose the user's question before retrieval —
one of this project's Best Practices bonus items.

Motivated by a concrete failure mode already observed in this project (see README):
naive retrieval for a cross-sector comparison query ("compare tech vs banks risk")
tends to be dominated by whichever sector has more chunks in the corpus, because a
single embedding/BM25 query for the combined question doesn't retrieve evenly from
both sides. finsignal-rag's old retrieve_balanced() patched this with a metadata-only
stopgap (force an equal quota per sector); this instead rewrites the question into
one focused sub-query per side of the comparison, so each side is retrieved on its
own terms — a genuine query-rewriting fix, not a retrieval-time quota.

For a single-topic (non-comparison) question, the rewrite is a no-op: returns the
original query unchanged, since there's nothing to split and a rewrite could only
make retrieval worse by drifting from the user's actual wording.

Each returned item also carries a "filter_dict" (see rag/fields.py for the
KEYWORD_FIELDS this can restrict on — ticker and/or sector) so rag/pipeline.py can
scope each sub-query's retrieval accordingly — without this, a decomposed sub-query
still searches the WHOLE corpus (no filter), so a larger/more-relevant-sounding
sector or company can still crowd out the sub-query's own intended scope,
undermining the whole point of splitting the query. A sector-level comparison
("tech vs banks") gets {"sector": ...} per sub-query; a company-level comparison
("Google vs Microsoft") gets the finer-grained {"ticker": ...} instead. A
single-topic question gets "filter_dict": null, meaning "defer to whatever filter
the caller already passed in" rather than forcing one.

CLI:
    uv run python -m rag.query_rewrite "Compare tech vs banks AI risk factors"
    uv run python -m rag.query_rewrite "Compare Google vs Microsoft's AI risk factors"
"""

import argparse
import json
import logging
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from rag.fields import SECTOR

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL = "gemini-3.1-flash-lite"

SYSTEM_PROMPT = """You rewrite user questions about SEC 10-K filings into one or more
focused search queries, to improve retrieval from a vector/keyword search index over
passages from 6 companies: GOOGL, MSFT, NVDA (tech sector) and JPM, GS, BAC (banking
sector).

If the question compares two SECTORS (e.g. "tech vs banks"), split it into one
focused sub-query per sector, naming the relevant tickers explicitly, and tag each
with filter_dict {"sector": "tech"} or {"sector": "banks"}.
If the question compares two or more specific COMPANIES (e.g. "Google vs
Microsoft"), split it into one focused sub-query per company instead, tagged with
the finer-grained filter_dict {"ticker": "GOOGL"} / {"ticker": "MSFT"} / etc.
If the question is about a single company/sector/topic, return it unchanged as the
only item, with "filter_dict" set to null.

Respond with ONLY a JSON array of objects, no other text, no markdown fences. Each
object has "query" (string) and "filter_dict" (an object with exactly one key,
either "sector" or "ticker", or null). Example:
[{"query": "What are Google's, Microsoft's, and Nvidia's main AI risk factors?", "filter_dict": {"sector": "tech"}}, {"query": "What are JPMorgan's, Goldman Sachs's, and Bank of America's main AI risk factors?", "filter_dict": {"sector": "banks"}}]
"""

_VALID_FILTERS = ({"sector": "tech"}, {"sector": "banks"}, *({"ticker": t} for t in SECTOR), None)


def rewrite(query: str) -> list[dict]:
    """Return one or more {"query": str, "filter_dict": dict|None} items. Falls back
    to [{"query": query, "filter_dict": None}] on any failure (missing API key,
    malformed response, API error) so callers never have to special-case query
    rewriting being unavailable."""
    fallback = [{"query": query, "filter_dict": None}]
    load_dotenv(override=True)
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        log.warning("GEMINI_API_KEY not set — skipping query rewrite, using original query")
        return fallback

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=MODEL,
            contents=query,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=256,
                temperature=0,
            ),
        )
        items = json.loads(response.text)
        if (
            isinstance(items, list) and items
            and all(isinstance(it, dict) and "query" in it and "filter_dict" in it for it in items)
            and all(it["filter_dict"] in _VALID_FILTERS for it in items)
        ):
            return items
        log.warning("Unexpected query rewrite response, falling back to original query: %r", response.text)
    except Exception as exc:
        log.warning("Query rewrite failed (%s), falling back to original query", exc)
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="query rewrite test")
    parser.add_argument("query", help="the question to rewrite")
    args = parser.parse_args()

    items = rewrite(args.query)
    print(f"\nOriginal: {args.query}\n")
    print("Rewritten:")
    for it in items:
        print(f"  - [{it['filter_dict'] or 'any'}] {it['query']}")


if __name__ == "__main__":
    main()
