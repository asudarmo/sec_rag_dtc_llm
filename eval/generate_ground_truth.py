"""Generate a retrieval ground-truth set: one (question, source_chunk_id) pair per
sampled chunk, via an LLM. Used by eval/retrieval_eval.py to compute Hit Rate@k and
MRR@k across retrieval configs — that scoring step needs zero further LLM calls,
only this generation step does.

Samples STRATIFIED by ticker (25 chunks/ticker x 6 tickers = 150), not uniformly at
random — a uniform sample would mirror the corpus's own ~3:1 bank/tech chunk-count
imbalance and secretly evaluate retrieval quality mostly on bank content.

CLI:
    uv run python -m eval.generate_ground_truth
"""

import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path

from rag.judge import SLEEP_BETWEEN_CALLS, call_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

CHUNKS_FILE = Path(__file__).resolve().parents[1] / "data" / "processed" / "chunks.json"
OUT_FILE = Path(__file__).resolve().parents[1] / "eval" / "results" / "ground_truth.json"
PER_TICKER = 25
SEED = 42  # fixed for reproducibility — same sample every run, per the rubric's "versions specified" bar

QUESTION_PROMPT = """You are generating a retrieval-evaluation question for a RAG system
over SEC 10-K filings. Given the passage below, write ONE natural question a real user
might ask that this SPECIFIC passage answers. Paraphrase — do not copy phrases
verbatim from the passage, as a real user wouldn't know the exact wording in advance.

Respond with ONLY a JSON object: {{"question": "..."}}

PASSAGE:
{text}
"""


def sample_chunks(chunks: list[dict], per_ticker: int, seed: int) -> list[dict]:
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_ticker[c["ticker"]].append(c)

    rng = random.Random(seed)
    sampled = []
    for ticker in sorted(by_ticker):
        sampled.extend(rng.sample(by_ticker[ticker], min(per_ticker, len(by_ticker[ticker]))))
    return sampled


def generate_question(chunk: dict) -> str:
    data = call_json(QUESTION_PROMPT.format(text=chunk["text"]))
    return data["question"]


def main() -> None:
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    sampled = sample_chunks(chunks, PER_TICKER, SEED)
    log.info("Sampled %d chunks (%d per ticker), generating questions...", len(sampled), PER_TICKER)

    ground_truth = []
    for i, chunk in enumerate(sampled, 1):
        question = generate_question(chunk)
        ground_truth.append({"question": question, "chunk_id": chunk["id"], "ticker": chunk["ticker"]})
        log.info("[%d/%d] %s -> %s", i, len(sampled), chunk["ticker"], question)
        time.sleep(SLEEP_BETWEEN_CALLS)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(ground_truth, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Saved %d ground-truth questions to %s", len(ground_truth), OUT_FILE)


if __name__ == "__main__":
    main()
