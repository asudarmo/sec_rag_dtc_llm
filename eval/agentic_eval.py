"""Agentic RAG vs. traditional pipeline: does letting the LLM decide its own
search strategy (rag/agent.py) beat our best *fixed* config for comparison
questions — the case rag/query_rewrite.py was specifically built to handle?

Runs both on the same 15 comparison questions used by eval/llm_eval.py
(eval/comparison_questions.json), scored by the same judge
(rag.judge.judge_answer_quality) for direct comparability with those existing
results. The traditional config is dense+rerank+rewrite — our best *fixed*
config, not a weaker baseline, so this is a fair comparison, not a stacked deck.

Also records operational metrics distinctly: latency and total Gemini calls per
question. Agentic RAG's real trade-off vs. the fixed pipeline is adaptability at
the cost of more, less-predictable LLM calls — worth measuring explicitly, not
just answer quality (see PLAN.md for why: every agentic loop iteration is a full
Gemini round-trip, and today's free-tier 15 RPM limit makes this a real cost, not
a theoretical one).

Checkpoints after every item to eval/results/agentic_eval_raw.json and resumes
from it on restart — same pattern as eval/llm_eval.py, for the same reason (a
version without this once crashed and lost everything).

CLI:
    uv run python -m eval.agentic_eval
"""

import json
import logging
import time
from pathlib import Path

from rag.agent import agentic_answer
from rag.generate import generate_answer
from rag.judge import SLEEP_BETWEEN_CALLS, judge_answer_quality
from rag.pipeline import retrieve

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

QUESTIONS_FILE = Path(__file__).resolve().parent / "comparison_questions.json"
RAW_FILE = Path(__file__).resolve().parent / "results" / "agentic_eval_raw.json"
OUT_FILE = Path(__file__).resolve().parent / "results" / "agentic_eval.json"

CONFIGS = ["traditional", "agentic"]


def _run_traditional(question: str) -> dict:
    """Best fixed config: dense+rerank+rewrite. Retrieves once, then generates from
    those same hits directly (rag.generate.answer() would retrieve a second time
    internally, double-firing the rewrite call — avoided here on purpose)."""
    hits = retrieve(question, k=6, method="dense", use_rerank=True, use_query_rewrite=True)
    ans = generate_answer(question, hits)
    # rewrite always fires here since these are all comparison questions -> 1 rewrite call + 1 generate call.
    return {"answer": ans, "hits": hits, "n_gemini_calls": 2}


def _run_agentic(question: str) -> dict:
    result = agentic_answer(question)
    return {"answer": result["answer"], "hits": result["hits"], "n_gemini_calls": result["n_iterations"], "n_tool_calls": len(result["tool_calls"])}


def load_raw() -> list[dict]:
    return json.loads(RAW_FILE.read_text(encoding="utf-8")) if RAW_FILE.exists() else []


def save_raw(raw: list[dict]) -> None:
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    RAW_FILE.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize(raw: list[dict]) -> list[dict]:
    summary = []
    for config in CONFIGS:
        rows = [r for r in raw if r["config"] == config]
        if not rows:
            continue
        n = len(rows)
        faith_rows = [r["faithfulness"] for r in rows if r.get("faithfulness") is not None]
        summary.append({
            "config": config,
            "avg_faithfulness": round(sum(faith_rows) / len(faith_rows), 3) if faith_rows else None,
            "avg_context_precision": round(sum(r["context_precision"] for r in rows) / n, 3),
            "pct_relevant": round(sum(1 for r in rows if r["relevance"] == "RELEVANT") / n, 3),
            "avg_latency_s": round(sum(r["latency_s"] for r in rows) / n, 2),
            "avg_gemini_calls": round(sum(r["n_gemini_calls"] for r in rows) / n, 2),
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
        run_fn = _run_traditional if config == "traditional" else _run_agentic
        for item in questions:
            if (config, item["question"]) in done:
                continue

            start = time.monotonic()
            run_result = run_fn(item["question"])
            latency_s = round(time.monotonic() - start, 2)
            time.sleep(SLEEP_BETWEEN_CALLS)

            verdict = judge_answer_quality(item["question"], run_result["hits"], run_result["answer"])
            time.sleep(SLEEP_BETWEEN_CALLS)

            row = {
                "config": config, "question": item["question"], "type": item["type"],
                "answer": run_result["answer"], "latency_s": latency_s,
                "n_gemini_calls": run_result["n_gemini_calls"],
                "n_tool_calls": run_result.get("n_tool_calls"),
                **verdict,
            }
            raw.append(row)
            save_raw(raw)  # incremental checkpoint — an interrupt loses at most this one item
            log.info(
                "[%s] %s -> faithfulness=%s context_precision=%.2f relevance=%s latency=%.1fs (%d/%d total)",
                config, item["question"][:60], row.get("faithfulness"), row["context_precision"],
                row["relevance"], latency_s, len(raw), total,
            )

    summary = summarize(raw)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({"summary": summary, "details": raw}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n===== Agentic RAG vs traditional (dense+rerank+rewrite), n={len(questions)} =====")
    print(f"{'config':<12} {'faithfulness':>12} {'ctx_precision':>13} {'%relevant':>10} {'latency(s)':>11} {'gemini_calls':>13}")
    for r in summary:
        faith = f"{r['avg_faithfulness']:.3f}" if r["avg_faithfulness"] is not None else "n/a"
        print(f"{r['config']:<12} {faith:>12} {r['avg_context_precision']:>13.3f} {r['pct_relevant']:>10.1%} {r['avg_latency_s']:>11.2f} {r['avg_gemini_calls']:>13.2f}")
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
