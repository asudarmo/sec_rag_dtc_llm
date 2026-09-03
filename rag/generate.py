"""Combine retrieved chunks with a prompt and call the LLM to generate an answer.

The final stage of RAG: retrieve the most relevant 10-K passages, assemble them
into a "context", and send it together with the question to the LLM (Google
Gemini, free tier), instructing it to answer ONLY from the provided passages and
to cite its sources. Source-traceable answers are exactly what makes RAG stronger
than asking the LLM directly.

Ported from finsignal-rag's src/generate.py: retrieval now goes through
rag.pipeline.retrieve instead of calling rag.retrieve_dense directly, so the
method/rerank/query-rewrite stages added in Day 2 are all reachable from here (and
from the Day 3 evaluation sweep, which calls the same function with different args).

Requires GEMINI_API_KEY in .env (free key at https://aistudio.google.com/apikey).

CLI:
    uv run python -m rag.generate "Compare the main risks for tech vs banks"
    uv run python -m rag.generate "What are Google's revenue sources?" --ticker GOOGL --k 5
    uv run python -m rag.generate "Compare tech vs banks AI risk" --rewrite
    uv run python -m rag.generate "What are the main risk factors?" --no-rerank
"""

import argparse
import logging
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from rag.fields import SECTOR
from rag.pipeline import retrieve

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL = "gemini-3.1-flash-lite"  # pinned concrete version; free tier RPD=500 vs 20 for full Flash-tier models
MAX_TOKENS = 2048

SYSTEM_PROMPT = """You are a financial-document analysis assistant. The user gives you a
question plus passages retrieved from public companies' 10-K annual reports
(each tagged with [number] and ticker).

Rules:
1. Answer ONLY from the provided passages. Do not add facts that are not in them.
2. Cite the passage [number] after each claim so the reader can trace the source.
3. If the passages are insufficient to answer, say so explicitly. Do not fabricate.
4. When comparing companies or sectors, state the differences clearly.
Answer in English."""


def build_context(hits: list[dict]) -> str:
    """Assemble retrieved chunks into a numbered context string with provenance."""
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        blocks.append(
            f"[{i}] ticker={m['ticker']} sector={m['sector']} filing_date={m['filing_date']}\n"
            f"{h['text'].strip()}"
        )
    return "\n\n".join(blocks)


def generate_answer(query: str, hits: list[dict]) -> str:
    """Build context from already-retrieved hits, call Gemini, return a cited answer."""
    load_dotenv(override=True)
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in .env (https://aistudio.google.com/apikey)")

    if not hits:
        return "No relevant passages found. Build the vector store first (uv run python -m ingestion.embed)."

    context = build_context(hits)
    user_message = (
        f"Question: {query}\n\n"
        f"Passages retrieved from 10-K filings:\n\n{context}\n\n"
        f"Answer the question using only these passages, and cite the [number] sources."
    )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=MAX_TOKENS,
        ),
    )
    return response.text or ""


def answer(
    query: str,
    k: int = 6,
    filter_dict: dict[str, str] | None = None,
    method: str = "dense",
    use_rerank: bool = True,
    use_query_rewrite: bool = False,
) -> str:
    """Retrieve top-k chunks, then generate a cited answer (CLI convenience wrapper).

    Defaults to dense retrieval + reranking — the config eval/retrieval_eval.py's full
    method x rerank sweep found best (Hit Rate@5=0.620, MRR@5=0.482), narrowly ahead of
    hybrid+rerank (0.613/0.457); see eval/results/retrieval_eval.json. Reranking helps
    every base method substantially and consistently — it's not hybrid-specific — but
    once it's doing the heavy lifting, BM25's weaker signal in the hybrid-fused pool
    doesn't add value and mildly hurts. Hybrid search is still fully implemented and
    available via --method hybrid (and still evaluated) for the rubric's hybrid-search
    bonus point; it's just not the empirically best default. use_query_rewrite defaults
    off since it's only useful for comparison-style questions (see eval/llm_eval.py) —
    turning it on for a single-topic question wastes an LLM call for no benefit.
    filter_dict restricts to exact matches on KEYWORD_FIELDS (rag/fields.py), e.g.
    {"sector": "banks"} or {"ticker": "GOOGL"}.
    """
    hits = retrieve(query, k=k, filter_dict=filter_dict, method=method, use_rerank=use_rerank, use_query_rewrite=use_query_rewrite)
    return generate_answer(query, hits)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG question answering")
    parser.add_argument("query", help="the question to ask")
    parser.add_argument("--k", type=int, default=6, help="how many chunks to retrieve as context (default 6)")
    parser.add_argument("--sector", choices=["tech", "banks"], help="restrict to one sector")
    parser.add_argument("--ticker", choices=sorted(SECTOR), help="restrict to one company")
    parser.add_argument("--method", choices=["dense", "sparse", "hybrid"], default="dense", help="retrieval method (default dense — see eval/results/retrieval_eval.json)")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false", help="disable cross-encoder reranking (on by default — see eval/results/retrieval_eval.json)")
    parser.add_argument("--rewrite", action="store_true", help="apply LLM query rewriting before retrieval")
    args = parser.parse_args()

    filter_dict = {k: v for k, v in [("sector", args.sector), ("ticker", args.ticker)] if v}
    print("\nRetrieving and generating answer...\n")
    print(answer(
        args.query, k=args.k, filter_dict=filter_dict,
        method=args.method, use_rerank=args.rerank, use_query_rewrite=args.rewrite,
    ))


if __name__ == "__main__":
    main()
