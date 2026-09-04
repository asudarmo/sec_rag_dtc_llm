# Evaluation

Two evaluation stages: retrieval quality (does the right passage come back?) and
generation quality (does the LLM use it well?). See [agentic-rag.md](agentic-rag.md)
for the third comparison — agentic RAG against the best fixed config from here.

## Retrieval evaluation

**Method**: Hit Rate@5 / MRR@5 across 6 retrieval configs (method × rerank on/off),
against a 150-question ground-truth set — one LLM-generated question per chunk,
stratified 25/ticker so the corpus's ~3:1 bank/tech chunk-count imbalance doesn't
secretly skew the eval toward bank content. This step makes **no LLM calls** at
scoring time (it just checks whether the known source chunk id was retrieved), so the
full 6-config sweep is cheap to run.

```bash
uv run python -m eval.generate_ground_truth   # ~150 Gemini calls, one-time, ~15-25 min (free-tier rate limits)
uv run python -m eval.retrieval_eval          # no LLM calls, scores against the ground-truth chunk ids
```

**Results** (`eval/results/retrieval_eval.json`):

| config | hit_rate | mrr |
|---|---|---|
| dense | 0.527 | 0.352 |
| **dense+rerank** | **0.620** | **0.482** |
| sparse | 0.347 | 0.230 |
| sparse+rerank | 0.453 | 0.359 |
| hybrid | 0.513 | 0.336 |
| hybrid+rerank | 0.613 | 0.457 |

**`dense+rerank` wins**, narrowly ahead of `hybrid+rerank` — now the app's default.
Reranking helps *every* base method substantially and consistently (roughly +0.10
hit_rate each) — a general effect, not hybrid-specific. Once the cross-encoder is
doing the heavy lifting, BM25's weaker signal in the hybrid-fused pool doesn't add
value and mildly hurts. Hybrid search remains fully implemented and evaluated
(`--method hybrid`) for the rubric's hybrid-search bonus point — it's just not the
empirically best default.

## LLM-judge generation evaluation

**Method**: does query rewriting actually improve final answer quality on comparison
questions? A 2×2 sweep — `method` (hybrid vs. dense) × `use_query_rewrite` (off vs.
on), both with reranking on — over 15 hand-curated comparison questions
(`eval/comparison_questions.json`, 7 sector-level + 8 company-level), judged for
whether the answer substantively covers **both** sides of the comparison rather than
one side (or generic boilerplate standing in for it).

```bash
uv run python -m eval.llm_eval   # ~60 answer+judge pairs; resumable — checkpoints after every item
```

**Results** (`eval/results/llm_eval.json`):

| config | avg_score | % covers both sides |
|---|---|---|
| hybrid_no_rewrite | 0.627 | 53.3% |
| hybrid_with_rewrite | 0.760 | 66.7% |
| dense_no_rewrite | 0.667 | 53.3% |
| **dense_with_rewrite** | **0.827** | **66.7%** |

Query rewriting genuinely improves comparison-question quality regardless of base
method, and `dense` scores a bit higher than `hybrid` in both rewrite conditions —
consistent with the retrieval evaluation's finding above. Not a complete fix, though:
even with rewrite, a third of questions still fail to cover both sides properly (an
unusual cross-sector pairing like "JPMorgan vs Microsoft" scored low under both
methods). `use_query_rewrite` stays off by default in `generate.answer()` since it's a
per-question judgment call — only worth the extra LLM call for genuine comparison
questions, not a global win.

**A rate-limit note if you re-run these**: the Gemini free tier's binding constraint
is 15 requests/minute, not the commonly-cited 500/day figure — run one eval script at
a time (see [CHANGELOG.md](../CHANGELOG.md) for the full story).
