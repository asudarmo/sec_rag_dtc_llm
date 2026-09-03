"""Retrieval evaluation: Hit Rate@k and MRR@k across retrieval configs, against the
ground-truth set from eval/generate_ground_truth.py. No LLM calls needed here —
scoring is just "does the known source chunk id appear in the retrieved results,
and at what rank."

Configs compared: dense-only, sparse-only, hybrid (RRF), hybrid+rerank (cross-encoder).
Query rewriting isn't in this sweep — per-chunk ground-truth questions are inherently
single-topic, so rewriting mostly no-ops against them; see eval/llm_eval.py instead,
which evaluates rewriting on a purpose-built comparison-question set where it
actually activates.

CLI:
    uv run python -m eval.retrieval_eval
"""

import json
import logging
from pathlib import Path

from rag import pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

GROUND_TRUTH_FILE = Path(__file__).resolve().parent / "results" / "ground_truth.json"
OUT_FILE = Path(__file__).resolve().parent / "results" / "retrieval_eval.json"
K = 5

CONFIGS = [
    {"name": "dense", "method": "dense", "use_rerank": False},
    {"name": "dense+rerank", "method": "dense", "use_rerank": True},
    {"name": "sparse", "method": "sparse", "use_rerank": False},
    {"name": "sparse+rerank", "method": "sparse", "use_rerank": True},
    {"name": "hybrid", "method": "hybrid", "use_rerank": False},
    {"name": "hybrid+rerank", "method": "hybrid", "use_rerank": True},
]


def evaluate_config(ground_truth: list[dict], config: dict, k: int) -> dict:
    hits_at_k = 0
    reciprocal_ranks = []
    for item in ground_truth:
        results = pipeline.retrieve(item["question"], k=k, method=config["method"], use_rerank=config["use_rerank"])
        result_ids = [h["id"] for h in results]
        if item["chunk_id"] in result_ids:
            hits_at_k += 1
            reciprocal_ranks.append(1.0 / (result_ids.index(item["chunk_id"]) + 1))
        else:
            reciprocal_ranks.append(0.0)

    n = len(ground_truth)
    return {
        "config": config["name"],
        "hit_rate": round(hits_at_k / n, 4),
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        "n_questions": n,
    }


def main() -> None:
    ground_truth = json.loads(GROUND_TRUTH_FILE.read_text(encoding="utf-8"))
    log.info("Evaluating %d configs against %d ground-truth questions (k=%d)...", len(CONFIGS), len(ground_truth), K)

    results = []
    for config in CONFIGS:
        result = evaluate_config(ground_truth, config, K)
        log.info("%s: hit_rate=%.3f mrr=%.3f", config["name"], result["hit_rate"], result["mrr"])
        results.append(result)

    best = max(results, key=lambda r: r["mrr"])

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({"k": K, "results": results, "best_config": best["config"]}, indent=2), encoding="utf-8")

    print(f"\n===== Retrieval evaluation (k={K}, n={len(ground_truth)}) =====")
    print(f"{'config':<15} {'hit_rate':>10} {'mrr':>8}")
    for r in results:
        marker = "  <- best (by MRR)" if r["config"] == best["config"] else ""
        print(f"{r['config']:<15} {r['hit_rate']:>10.3f} {r['mrr']:>8.3f}{marker}")
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
