"""Agentic RAG: an LLM-driven loop that decides for itself when and how to search,
instead of a fixed retrieve-then-generate pipeline (rag/pipeline.py + rag/generate.py).

Matches the DTC course's own agentic-RAG pattern (01-agentic-rag, lessons
13-function-calling.md / 14-agentic-loop.md), adapted from their OpenAI-based
example to Gemini's function-calling API. Per the course: a fixed pipeline "always
does the same thing, no matter what" and has "no recovery" if a search returns
garbage; an agent instead lets the model "decide which actions to take and in
which order" — search again, rephrase, or split a comparison into separate
searches, adaptively, rather than via our own fixed rag/query_rewrite.py schema.

Deliberately ONE tool (search_filings), matching the course's own single-tool
example — see PLAN.md for why: our domain has exactly one kind of action
(retrieve passages, optionally filtered by sector/ticker), so multiple tools would
add complexity without adding real capability. The tool wraps rag.pipeline.retrieve
with the empirically-best config (dense+rerank, see eval/results/retrieval_eval.json),
so this reuses the same retrieval engine as the traditional pipeline — only the
orchestration (fixed function vs. LLM-driven loop) differs.

Automatic function calling (AFC) is explicitly disabled: we hand-roll the loop
(matching course pedagogy) so every tool call can be logged for the traditional-
vs-agentic comparison (eval/agentic_eval.py) and the app's monitoring/n_tool_calls
tracking, and so a MAX_ITERATIONS safety cap applies (the course's own bare example
has none, but flags this as worth adding).

CLI:
    uv run python -m rag.agent "Compare the main AI-related risk factors between tech and banking companies"
"""

import argparse
import logging
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from rag.fields import SECTOR
from rag.generate import MODEL, SYSTEM_PROMPT
from rag.pipeline import retrieve

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MAX_ITERATIONS = 5
SLEEP_BETWEEN_CALLS = 4  # seconds; same free-tier RPM pacing as eval scripts (rag/judge.py)
SEARCH_K = 5
SEARCH_METHOD = "dense"  # the empirically-best base method, per Day 3 eval
SEARCH_USE_RERANK = True

AGENT_SYSTEM_PROMPT = SYSTEM_PROMPT + """

You have a search_filings tool. Call it as many times as you need — once per
company when comparing companies, once per sector when comparing sectors, or
again with different wording if a search doesn't return what you need — before
giving your final answer. Only answer once you have enough passages to cite for
every part of the question."""

_SEARCH_TOOL = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="search_filings",
        description=(
            "Search SEC 10-K filing passages for a query. Optionally filter to one "
            "sector or one company (ticker) — use this to scope a search to just one "
            "side of a comparison."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query text."},
                "sector": {"type": "string", "enum": ["tech", "banks"], "description": "Restrict to one sector."},
                "ticker": {"type": "string", "enum": sorted(SECTOR), "description": "Restrict to one company."},
            },
            "required": ["query"],
        },
    )
])


def _search_filings(query: str, sector: str | None = None, ticker: str | None = None) -> list[dict]:
    """The tool's actual implementation: calls the same rag.pipeline.retrieve() the
    traditional pipeline uses, with the empirically-best config."""
    filter_dict = {k: v for k, v in [("sector", sector), ("ticker", ticker)] if v} or None
    return retrieve(query, k=SEARCH_K, filter_dict=filter_dict, method=SEARCH_METHOD, use_rerank=SEARCH_USE_RERANK)


def _format_context(hits: list[dict], id_to_number: dict[str, int]) -> str:
    """Same block format as rag.generate.build_context, but numbered from a shared
    id_to_number map that persists across every tool call within one agentic_answer()
    run, instead of restarting at [1] for each call. Without this, a second tool call
    (e.g. the banking side of a sector comparison) would be numbered [1]-[N] in what
    the model sees, while the deduped `hits` list shown in the UI's Sources section
    numbers it [N+1]-[2N] — the model's citations and the on-screen sources would
    disagree by exactly the size of the first call's result set."""
    blocks = []
    for h in hits:
        m = h["metadata"]
        blocks.append(
            f"[{id_to_number[h['id']]}] ticker={m['ticker']} sector={m['sector']} filing_date={m['filing_date']}\n"
            f"{h['text'].strip()}"
        )
    return "\n\n".join(blocks)


def _call_with_retry(client: genai.Client, contents: list, max_attempts: int = 5):
    """Same transient-error retry pattern as rag/judge.py's with_retry, inlined here
    to avoid a rag.judge -> rag.generate -> rag.agent import cycle risk."""
    import random
    for attempt in range(max_attempts):
        try:
            return client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=AGENT_SYSTEM_PROMPT,
                    tools=[_SEARCH_TOOL],
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
        except Exception as exc:
            msg = str(exc)
            retryable = any(code in msg for code in ("429", "500", "503")) or "UNAVAILABLE" in msg or "RESOURCE_EXHAUSTED" in msg
            if not retryable or attempt == max_attempts - 1:
                raise
            wait = min(2 ** attempt + random.random(), 30)
            log.warning("Gemini transient error (%s); retry %d/%d in %.1fs", msg[:80], attempt + 1, max_attempts - 1, wait)
            time.sleep(wait)


def agentic_answer(query: str, max_iterations: int = MAX_ITERATIONS) -> dict:
    """Run the agentic loop for one question. Returns:
    {"answer": str, "hits": list[dict] (deduped across all tool calls),
     "tool_calls": list[dict] (each {"query", "sector", "ticker"}, in call order),
     "n_iterations": int}
    """
    load_dotenv(override=True)
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in .env (https://aistudio.google.com/apikey)")

    client = genai.Client(api_key=api_key)
    contents = [types.Content(role="user", parts=[types.Part(text=query)])]
    all_hits: dict[str, dict] = {}
    id_to_number: dict[str, int] = {}  # shared citation numbering across all tool calls — see _format_context
    tool_calls: list[dict] = []

    for iteration in range(1, max_iterations + 1):
        resp = _call_with_retry(client, contents)
        candidate_content = resp.candidates[0].content
        parts = candidate_content.parts or []
        function_calls = [p.function_call for p in parts if p.function_call]

        if not function_calls:
            final_text = "".join(p.text or "" for p in parts)
            return {
                "answer": final_text,
                "hits": list(all_hits.values()),
                "tool_calls": tool_calls,
                "n_iterations": iteration,
            }

        contents.append(candidate_content)  # the model's own turn (preserves internal fields like thought_signature)
        response_parts = []
        for fc in function_calls:
            args = fc.args or {}
            log.info("[iter %d] search_filings(%r)", iteration, args)
            time.sleep(SLEEP_BETWEEN_CALLS)
            hits = _search_filings(args.get("query", query), args.get("sector"), args.get("ticker"))
            for h in hits:
                if h["id"] not in id_to_number:
                    id_to_number[h["id"]] = len(id_to_number) + 1
                    all_hits[h["id"]] = h
            context_text = _format_context(hits, id_to_number)
            tool_calls.append({"query": args.get("query", query), "sector": args.get("sector"), "ticker": args.get("ticker")})
            response_parts.append(types.Part(function_response=types.FunctionResponse(
                id=fc.id, name=fc.name, response={"result": context_text},
            )))
        contents.append(types.Content(role="user", parts=response_parts))
        time.sleep(SLEEP_BETWEEN_CALLS)

    # Hit MAX_ITERATIONS without a final answer — the course's bare example has no
    # such cap at all; we force a stop and ask for a best-effort answer from
    # whatever's been gathered so far, rather than looping indefinitely.
    log.warning("Hit max_iterations=%d without a final answer; forcing a stop.", max_iterations)
    contents.append(types.Content(role="user", parts=[types.Part(
        text="You've reached the maximum number of searches. Answer now using only what you've found so far."
    )]))
    resp = _call_with_retry(client, contents)
    final_text = "".join(p.text or "" for p in (resp.candidates[0].content.parts or []))
    return {"answer": final_text, "hits": list(all_hits.values()), "tool_calls": tool_calls, "n_iterations": max_iterations}


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG question answering")
    parser.add_argument("query", help="the question to ask")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    args = parser.parse_args()

    result = agentic_answer(args.query, max_iterations=args.max_iterations)
    print(f"\n{len(result['tool_calls'])} search(es) over {result['n_iterations']} iteration(s):")
    for tc in result["tool_calls"]:
        print(f"  - {tc}")
    print(f"\nAnswer:\n{result['answer']}")


if __name__ == "__main__":
    main()
