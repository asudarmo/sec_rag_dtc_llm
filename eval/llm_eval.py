"""LLM-judge generation evaluation: does query rewriting actually improve final
answer quality on comparison questions, not just retrieval hit-rate (which
eval/retrieval_eval.py structurally can't measure for rewriting, since it only
activates on comparison-style questions that a per-chunk ground-truth set doesn't
naturally produce)?

A 2x2 sweep — method (hybrid vs dense) x use_query_rewrite (False vs True), both
with reranking on — on the hand-curated comparison-question set
(eval/comparison_questions.json), scored by an LLM judge for whether the answer
substantively covers BOTH sides of the comparison (not just one, or generic
financial boilerplate standing in for the other).

The method axis was added after eval/retrieval_eval.py's extended sweep found
dense+rerank narrowly beating hybrid+rerank (the app's default changed
accordingly) — this checks whether the query-rewrite conclusion (originally
measured only against hybrid+rerank) still holds against the new default, rather
than assuming it carries over. The hybrid_* configs' results are the *original*
run, preserved and reused via the resume/checkpoint mechanism below (matched by
config name in eval/results/llm_eval_raw.json) rather than recomputed — same
answers, same judge scores, no wasted API calls redoing already-settled work.

Checkpoints after every single item to eval/results/llm_eval_raw.json and resumes
from it on restart (skips any (config, question) pair already done) — the free-tier
rate limit (15 RPM for gemini-3.1-flash-lite) makes this run slow and occasionally
throttled, and a version without this crashed once already, losing all progress
since it only wrote output at the very end.

CLI:
    uv run python -m eval.llm_eval
"""

import json
import logging
import time
from pathlib import Path

from rag.judge import SLEEP_BETWEEN_CALLS, call_json
from rag.generate import answer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

QUESTIONS_FILE = Path(__file__).resolve().parent / "comparison_questions.json"
RAW_FILE = Path(__file__).resolve().parent / "results" / "llm_eval_raw.json"
OUT_FILE = Path(__file__).resolve().parent / "results" / "llm_eval.json"

CONFIGS = [
    {"name": "hybrid_no_rewrite", "method": "hybrid", "use_query_rewrite": False},
    {"name": "hybrid_with_rewrite", "method": "hybrid", "use_query_rewrite": True},
    {"name": "dense_no_rewrite", "method": "dense", "use_query_rewrite": False},
    {"name": "dense_with_rewrite", "method": "dense", "use_query_rewrite": True},
]

JUDGE_PROMPT = """You are evaluating an AI assistant's answer to a COMPARISON question
about SEC 10-K risk factors (comparing two sectors or two companies). Score whether
the answer substantively addresses BOTH sides of the comparison with specific,
on-topic content — not generic financial boilerplate that could apply to any company.

Respond with ONLY a JSON object:
{{"covers_both_sides": true/false, "score": <float 0-1, where 1.0 = both sides
covered with specific relevant content, 0.5 = one side weak/generic, 0.0 = one side
missing entirely or both generic>, "reasoning": "<one sentence>"}}

QUESTION: {question}

ANSWER:
{answer}
"""


def judge_answer(question: str, generated_answer: str) -> dict:
    return call_json(JUDGE_PROMPT.format(question=question, answer=generated_answer))


def load_raw() -> list[dict]:
    if RAW_FILE.exists():
        return json.loads(RAW_FILE.read_text(encoding="utf-8"))
    return []


def save_raw(raw: list[dict]) -> None:
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    RAW_FILE.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize(raw: list[dict]) -> list[dict]:
    summary = []
    for config in CONFIGS:
        config_results = [r for r in raw if r["config"] == config["name"]]
        if not config_results:
            continue
        n = len(config_results)
        summary.append({
            "config": config["name"],
            "avg_score": round(sum(r["score"] for r in config_results) / n, 3),
            "pct_covers_both_sides": round(sum(1 for r in config_results if r["covers_both_sides"]) / n, 3),
            "n_questions": n,
        })
    return summary


def main() -> None:
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    raw = load_raw()
    done = {(r["config"], r["question"]) for r in raw}
    total = len(CONFIGS) * len(questions)
    log.info("Resuming: %d/%d (config, question) pairs already done", len(done), total)

    for config in CONFIGS:
        for item in questions:
            if (config["name"], item["question"]) in done:
                continue

            ans = answer(item["question"], k=6, method=config["method"], use_query_rewrite=config["use_query_rewrite"])
            time.sleep(SLEEP_BETWEEN_CALLS)
            verdict = judge_answer(item["question"], ans)
            time.sleep(SLEEP_BETWEEN_CALLS)

            raw.append({"config": config["name"], "question": item["question"], "type": item["type"], "answer": ans, **verdict})
            save_raw(raw)  # incremental checkpoint — an interrupt loses at most this one item
            log.info(
                "[%s] %s -> score=%.2f covers_both=%s (%d/%d total)",
                config["name"], item["question"][:60], verdict.get("score", 0), verdict.get("covers_both_sides"),
                len(raw), total,
            )

    summary = summarize(raw)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({"summary": summary, "details": raw}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n===== LLM-judge generation evaluation (method x query-rewrite, n={len(questions)}) =====")
    print(f"{'config':<15} {'avg_score':>10} {'pct_both_sides':>15}")
    for r in summary:
        print(f"{r['config']:<15} {r['avg_score']:>10.3f} {r['pct_covers_both_sides']:>15.1%}")
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
