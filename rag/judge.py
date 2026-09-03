"""Shared Gemini call helper (retry-with-backoff, JSON mode, rate-limit-friendly
pacing) plus the live-app LLM-as-judge scorer. Lives in rag/, not eval/, because
it's used both by the offline eval scripts (eval/generate_ground_truth.py,
eval/llm_eval.py) and by app.py's live, opt-in judge-scoring toggle — eval/ is for
offline-only harnesses.

Ported from finsignal-rag's evaluate.py — the retry/backoff plus
temperature=0+seed=42-for-reproducibility pattern already proved itself there
(temperature=1.0 gave inconsistent repeat scores: 0.00/0.00/0.68 on the same input).
"""

import json
import logging
import os
import random
import time
from typing import Callable, TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import types

from rag.generate import build_context

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL = "gemini-3.1-flash-lite"
SLEEP_BETWEEN_CALLS = 4  # seconds; keeps us under the free-tier requests-per-minute limit

T = TypeVar("T")

ANSWER_QUALITY_PROMPT = """You are evaluating an AI assistant's answer to a question
about SEC 10-K filings. The numbered PASSAGES below (in order [1], [2], ...) were
given to the assistant as context.

Do THREE things:

1. FAITHFULNESS: break the ANSWER into its atomic factual claims. For each claim,
   decide whether it is actually supported by the PASSAGES (not fabricated).
2. CONTEXT PRECISION: for each numbered PASSAGE, in order, decide whether it is
   relevant to answering the QUESTION.
3. RELEVANCE: classify whether the ANSWER overall addresses the QUESTION: one of
   "RELEVANT" (fully addresses it), "PARTLY_RELEVANT" (addresses part of it, or is
   too vague/generic), "NON_RELEVANT" (does not address it).

Respond with ONLY this JSON object:
{{
  "claims": [{{"claim": "...", "supported": true}}, ...],
  "passage_relevance": [true, false, ...],
  "relevance": "RELEVANT",
  "reasoning": "<one sentence covering all three judgments>"
}}

QUESTION: {question}

PASSAGES:
{context}

ANSWER:
{answer}
"""


def with_retry(call: Callable[[], T], max_attempts: int = 5) -> T:
    """Run `call`, retrying transient Gemini errors (429/5xx/overloaded) with
    exponential backoff + jitter. Non-retryable errors (bad key, depleted quota)
    raise immediately instead of retrying pointlessly."""
    for attempt in range(max_attempts):
        try:
            return call()
        except Exception as exc:
            msg = str(exc)
            if "prepayment" in msg.lower() or "API key not valid" in msg:
                raise
            retryable = (
                any(code in msg for code in ("429", "500", "503"))
                or "UNAVAILABLE" in msg
                or "RESOURCE_EXHAUSTED" in msg
                or "overloaded" in msg.lower()
            )
            if not retryable or attempt == max_attempts - 1:
                raise
            wait = min(2 ** attempt + random.random(), 30)
            log.warning("Gemini transient error (%s); retry %d/%d in %.1fs", msg[:80], attempt + 1, max_attempts - 1, wait)
            time.sleep(wait)


def call_json(prompt: str, max_attempts: int = 5) -> dict | list:
    """Call Gemini with JSON-mode output, greedy decoding (temperature=0, seed=42)
    for reproducibility, and transient-error retry. Returns the parsed JSON."""
    load_dotenv(override=True)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def _call():
        resp = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
                seed=42,
            ),
        )
        return json.loads(resp.text)

    return with_retry(_call, max_attempts)


def judge_answer_quality(question: str, hits: list[dict], answer: str) -> dict:
    """Score a single live answer on three distinct metrics, in one Gemini call:

    - faithfulness (RAGAS-style): fraction of the answer's atomic claims actually
      supported by the retrieved passages — ported from finsignal-rag's evaluate.py.
    - context_precision (RAGAS-style, rank-aware): are the retrieved passages
      relevant, weighted toward ones ranked earlier — same as above.
    - relevance: the DTC course's own live-judge categorization (Module 5, lesson
      09-built-in-judge.md) — RELEVANT / PARTLY_RELEVANT / NON_RELEVANT.

    A single blended score was tried first and found too insensitive: the
    generation system prompt already forbids fabrication, so faithfulness alone
    rarely varies, and blending it with relevance into one number obscured which
    dimension was actually failing when it did. Asking the judge to decompose into
    claims / per-passage verdicts (rather than output a bare aggregate float
    directly) keeps the same rigor as computing them via two separate calls, at
    the cost of one call instead of two — worth it given this runs live, opt-in,
    per question (rate-limit exposure), not just in batch eval jobs.

    Used by app.py's opt-in "Score this answer" toggle — NOT called by default.
    Returns {"faithfulness": float|None, "context_precision": float,
    "relevance": str, "reasoning": str}. faithfulness is None if the answer had no
    extractable claims (e.g. "insufficient information" refusals).
    """
    context = build_context(hits)
    data = call_json(ANSWER_QUALITY_PROMPT.format(question=question, context=context, answer=answer))

    claims = data.get("claims") or []
    faithfulness = round(sum(1 for c in claims if c.get("supported")) / len(claims), 3) if claims else None

    passage_relevance = [bool(v) for v in (data.get("passage_relevance") or [])]
    total_relevant = sum(passage_relevance)
    if total_relevant == 0:
        context_precision = 0.0
    else:
        running_hits = 0
        weighted = 0.0
        for i, relevant in enumerate(passage_relevance, 1):
            if relevant:
                running_hits += 1
                weighted += running_hits / i
        context_precision = round(weighted / total_relevant, 3)

    return {
        "faithfulness": faithfulness,
        "context_precision": context_precision,
        "relevance": data.get("relevance"),
        "reasoning": data.get("reasoning", ""),
    }
