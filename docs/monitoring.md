# Monitoring

## What's logged

Every question asked in the Streamlit app is logged to Postgres: question, answer,
retrieval config used (method/rerank/rewrite/filter, or agentic mode + tool-call
count), and latency — plus any 👍/👎 feedback (with an optional free-text comment)
and any LLM-judge scores, all linked back to that conversation.
`monitoring/db.py` owns the schema:

- `monitoring.conversations` — one row per question asked
- `monitoring.feedback` — one row per thumbs up/down, linked by `conversation_id`

This is a separate schema from dlt's `raw` ingestion staging tables, in the same
shared Postgres container (see [setup.md](setup.md)).

## Grafana dashboard

Auto-provisioned from `monitoring/grafana/` — no manual setup needed, comes up with
`docker compose up`. **13 panels**:

- Conversations over time
- Average latency over time
- Feedback: thumbs up vs. down
- Retrieval method usage
- Reranking usage rate
- Query-rewrite usage rate
- Average faithfulness (when scored)
- Average context precision (when scored)
- Relevance distribution (when scored)
- Recent questions (table)
- Recent feedback comments (table)
- Traditional vs. agentic RAG usage
- Average tool calls per agentic conversation

## In-app feedback & sources

Every answer is shown alongside a "Sources (N passages retrieved)" section — one
expander per retrieved passage (ticker, sector, filing date, full text) — so you can
check the answer is actually grounded in what was retrieved. Feedback is a colored
confirmation box after clicking 👍/👎, with an optional comment bundled into the same
submission.

## LLM-as-judge (opt-in)

A **"Score this answer (LLM-as-judge)"** sidebar toggle (off by default — it adds an
extra Gemini call per question) scores the answer right after generation and shows it
as KPI cards:

- **Faithfulness** — is every claim actually supported by the retrieved passages?
- **Context precision** — how much of the retrieved context was actually relevant?
- **Relevance** — RELEVANT / PARTLY_RELEVANT / NON_RELEVANT, the DTC course's own
  live-judge categorization (Module 5, `09-built-in-judge.md`)

All three come from one Gemini call: the judge decomposes the answer into atomic
claims and rates each retrieved passage's relevance, and the three numbers are
computed from those structured judgments — same rigor as separate calls per metric,
without the extra rate-limit cost. (Why three metrics instead of one blended score is
covered in [CHANGELOG.md](../CHANGELOG.md) — it's a decision history, not needed to
use the dashboard.)
