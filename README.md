# SEC 10-K RAG Assistant

A RAG system that answers questions about SEC 10-K annual filings for 6 public companies across two sectors — **tech** (GOOGL, MSFT, NVDA) and **banking** (JPM, GS, BAC) — grounding every answer in retrieved passages with traceable `[number]` citations. Built as a DataTalksClub LLM Zoomcamp capstone, adapted from an earlier university NLP class project ([finsignal-rag](#acknowledgements)).

**Problem it solves:** 10-K filings are long, dense, and hard to compare across companies or sectors. This assistant lets you ask a question in plain English (e.g. *"Compare the main AI-related risk factors between tech and banking companies"*) and get back a cited answer drawn only from the actual filing text — no fabrication, and every claim traceable to its source passage.

## Project status

This project is being built incrementally against a 5-day plan. **[PLAN.md](PLAN.md)** has the full day-by-day breakdown, target architecture, and a progress checklist — check there for what's done vs. still in progress. This README documents each component as it's completed; sections not yet built are marked 🚧 with a pointer to the relevant day in the plan.

## Setup

**Prerequisites:** Python 3.12, [uv](https://docs.astral.sh/uv/) (this project's package/environment manager — see `CLAUDE.md` for why).

```bash
uv sync              # installs all dependencies from pyproject.toml / uv.lock into .venv
cp .env.example .env  # then fill in the values below
```

`.env` variables:

| Variable | Required for | Notes |
|---|---|---|
| `GEMINI_API_KEY` | generation (`rag/generate.py`) | Free tier key at https://aistudio.google.com/apikey |
| `SEC_USER_NAME`, `SEC_USER_EMAIL` | downloading filings (`ingestion/download.py`, `ingestion/dlt_pipeline.py`) | Required by SEC EDGAR's fair-access policy — use your real name/email. Not needed if you're only querying the app, since a prebuilt vector store is committed (see below). |
| `POSTGRES_*` | ingestion staging + monitoring (Day 4) | Defaults match the docker-compose `postgres` service; not yet in use locally. |

Run anything in the project with `uv run <command>` (e.g. `uv run python -m rag.generate "..."`).

## Data & ingestion pipeline

**Domain:** the most recent 10-K filing for each of 6 tickers (GOOGL, MSFT, NVDA, JPM, GS, BAC), chunked at 1000 characters with 200 overlap (`langchain_text_splitters.RecursiveCharacterTextSplitter`).

**A prebuilt vector store is committed to the repo** (`chroma_db/`, `data/processed/chunks.json`) so you can run the app immediately without downloading or embedding anything yourself.

### Automated ingestion (dlt) — the primary path

`ingestion/dlt_pipeline.py` automates the full ingestion pipeline with [dlt](https://dlthub.com/): it downloads each ticker's latest 10-K from SEC EDGAR, parses/chunks it (same logic as `ingestion/ingest.py`), and loads the resulting records into a database table (`raw.filing_chunks`).

```bash
# Full run — loads into Postgres (the "raw" schema in the shared monitoring Postgres, see PLAN.md Day 4)
uv run python -m ingestion.dlt_pipeline

# Local dry run — no Postgres required, loads into a local DuckDB file instead
uv run python -m ingestion.dlt_pipeline --destination duckdb
```

Both require `SEC_USER_NAME`/`SEC_USER_EMAIL` in `.env` (real SEC EDGAR download, ~6 requests). Verified with a live run against `--destination duckdb`: downloaded all 6 filings and produced 5,201 chunks (BAC 1101, GOOGL 431, GS 1366, JPM 1472, MSFT 405, NVDA 426) — counts match the originally-committed `chunks.json` almost exactly (MSFT differs slightly since SEC EDGAR now serves a newer filing than when the seed data was built).

The Postgres-backed run (the default) needs the Day 4 docker-compose stack up first — not yet runnable standalone.

### Manual/scripted ingestion — still available

The pre-dlt scripts still work individually if you want to inspect each stage or rebuild the vector store from scratch:

```bash
uv run python -m ingestion.download   # SEC EDGAR -> data/raw/
uv run python -m ingestion.ingest     # data/raw/ -> data/processed/chunks.json (chunking)
uv run python -m ingestion.embed      # chunks.json -> chroma_db/ (sentence-transformers embeddings)
```

## Retrieval & generation

Three retrieval methods, each independently combinable with reranking and/or query rewriting, all wired behind a single entry point (`rag/pipeline.py`):

- **dense** *(default)* — ChromaDB vector search (`all-MiniLM-L6-v2`)
- **sparse** — BM25 keyword search (`rank_bm25`), in-memory index over `data/processed/chunks.json`
- **hybrid** — dense + sparse fused with Reciprocal Rank Fusion (RRF)

Plus two optional stages:
- **Reranking** — a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores the candidate pool for a higher-precision final ordering.
- **Query rewriting** — an LLM call decomposes comparison questions (e.g. "tech vs banks") into one focused sub-query per side before retrieving, fixing a real bias: naive retrieval for cross-sector queries is dominated by whichever sector has more chunks in the corpus (banks have ~3x more than tech here).

**Filtering** uses the DTC course's own minsearch vocabulary (01-agentic-rag/lessons/05-search.md): a `text_field` is tokenized/embedded and relevance-ranked (for us, just the chunk text); a `keyword_field` is an exact-match filter applied via a `filter_dict`, same as minsearch's `Index(text_fields=..., keyword_fields=...)` + `.search(query, filter_dict=...)`. `rag/fields.py` defines `KEYWORD_FIELDS = ("ticker", "sector")` and every retriever takes the same `filter_dict` parameter — so you can filter by sector (`{"sector": "tech"}`), by a specific company (`{"ticker": "GOOGL"}`), or both together.

```bash
# Retrieval only, no API key needed — swap the module for dense/sparse/hybrid
uv run python -m rag.retrieve_dense "What are the main AI-related risk factors?" --k 5 --sector tech
uv run python -m rag.retrieve_sparse "What are the main AI-related risk factors?" --k 5 --ticker GOOGL
uv run python -m rag.hybrid "What are the main AI-related risk factors?" --k 5 --ticker GOOGL
uv run python -m rag.rerank "What are the main AI-related risk factors?" --k 5 --sector tech

# Full retrieve + generate (needs GEMINI_API_KEY) — method/rerank/rewrite all toggleable
uv run python -m rag.generate "Compare the main AI-related risk factors between tech and banking companies" --k 6
uv run python -m rag.generate "Compare the main AI-related risk factors between tech and banking companies" --k 6 --method hybrid --rewrite
uv run python -m rag.generate "Compare Google vs Microsoft's AI risk factors" --k 6 --method hybrid --rewrite
```

Query rewriting picks the right filter granularity itself: a sector-level comparison ("tech vs banks") tags each sub-query with `{"sector": ...}`; a company-level comparison ("Google vs Microsoft") tags them with the finer-grained `{"ticker": ...}` instead — verified live, producing a clean, distinctly-sourced comparison for each company.

**Verified:** naive hybrid retrieval (no rewrite) still returns mostly bank passages for a cross-sector comparison question — confirming the imbalance is a genuine corpus-size skew, not something fusion alone fixes. Adding `--rewrite` produces an answer that correctly cites both sides (GOOGL/NVDA for tech, BAC/JPM for banking). **Reranking's effect on retrieval quality is now settled by Day 3's evaluation (see below): it's on by default, and helps every base method, not just hybrid.** `rag/generate.py`'s `answer()` defaults to `method="dense", use_rerank=True` — the config Hit Rate/MRR found best — with `--method`/`--no-rerank` to override.

**A follow-up bug, live-reported and fixed**: rewrite + rerank *together* could still return an imbalanced result (one real case: 5 bank passages vs 1 tech, despite rewrite being on) — because reranking re-scored the whole merged pool against the undecomposed question and picked a flat top-k, silently undoing the balance the per-sub-query retrieval had guaranteed going in. Fixed in `rag/pipeline.py` by reranking each sub-query's pool independently and taking a fixed quota from each, rather than reranking the merged pool. Verified with the exact reported query: 3 tech / 3 banks now.

**How this compares to the actual DTC course materials** (checked directly against Module 6, `06-best-practices`): the course's hybrid search lesson uses exactly this RRF formula (same `k=60`). Its "reranking" lesson, somewhat surprisingly, is *also* just RRF — it never uses a cross-encoder. So `rerank.py`'s cross-encoder here is not an adaptation of course code, it's a different technique — which is useful, since the rubric grades hybrid search and reranking as separate bonus points, and having two genuinely distinct techniques is a stronger claim than reusing RRF for both. To also have a faithful, *non-redundant* version of the course's actual RRF-based reranking (applying RRF to one query's dense+sparse pair would just be `hybrid.py` again), `rag/rrf.py` generalizes the fusion to N ranked lists and `pipeline.py` uses it to properly fuse the multiple per-sub-query lists that query rewriting produces — a genuinely different fusion problem (across queries, not across retrieval methods) that used to be a naive union.

That refactor also surfaced and fixed a real bug: query rewriting split the question's *wording* per sector but wasn't scoping *retrieval* to that sector, so a decomposed sub-query could still be crowded out by the larger sector's chunks. `query_rewrite.rewrite()` now tags each sub-query with its sector, and `pipeline.py` filters accordingly — verified live: before the fix, the tech side of a cross-sector answer surfaced generic R&D-expense text; after, it surfaced genuinely on-topic AI-risk content (EU AI Act exposure, ecosystem/supply-chain risk).

## Evaluation

**Retrieval evaluation** (`eval/retrieval_eval.py`): Hit Rate@5 / MRR@5 across 4 retrieval configs, against a 150-question ground-truth set (`eval/generate_ground_truth.py` — one LLM-generated question per chunk, stratified 25/ticker so the corpus's ~3:1 bank/tech chunk-count imbalance doesn't secretly skew the eval toward bank content).

| config | hit_rate | mrr |
|---|---|---|
| dense | 0.527 | 0.352 |
| **dense+rerank** | **0.620** | **0.482** |
| sparse | 0.347 | 0.230 |
| sparse+rerank | 0.453 | 0.359 |
| hybrid | 0.513 | 0.336 |
| hybrid+rerank | 0.613 | 0.457 |

**dense+rerank wins**, narrowly ahead of hybrid+rerank — now the app's default (`--method dense`, reranking on by default). Reranking helps *every* base method substantially and consistently (roughly +0.10 hit_rate each) — it's a general effect, not hybrid-specific. Once the cross-encoder is doing the heavy lifting, BM25's weaker signal in the hybrid-fused pool doesn't add value and mildly hurts. Hybrid search is still fully implemented and evaluated (`--method hybrid`) for the rubric's hybrid-search bonus point — it's just not the empirically best default.

```bash
uv run python -m eval.generate_ground_truth   # ~150 Gemini calls, one-time, ~15-25 min (free-tier rate limits)
uv run python -m eval.retrieval_eval          # no LLM calls, scores against the ground-truth chunk ids
```

**LLM-judge generation evaluation** (`eval/llm_eval.py`): does query rewriting actually improve final answer quality on comparison questions? A 2x2 sweep — `method` (hybrid vs dense) × `use_query_rewrite` (False vs True), both with reranking on — on 15 hand-curated comparison questions (`eval/comparison_questions.json`, 7 sector-level + 8 company-level), judged for whether the answer substantively covers *both* sides of the comparison rather than one side (or generic boilerplate standing in for it) — quantifying what Day 2 only showed anecdotally. The method axis was added after the retrieval eval found `dense+rerank` beating `hybrid+rerank`, to check whether the rewrite conclusion (originally measured only against hybrid) still holds against the new default.

```bash
uv run python -m eval.llm_eval   # ~60 answer+judge pairs; resumable — checkpoints after every item
```

| config | avg_score | % covers both sides |
|---|---|---|
| hybrid_no_rewrite | 0.627 | 53.3% |
| hybrid_with_rewrite | 0.760 | 66.7% |
| dense_no_rewrite | 0.667 | 53.3% |
| **dense_with_rewrite** | **0.827** | **66.7%** |

Query rewriting genuinely improves comparison-question quality regardless of base method — confirming Day 2's anecdotal finding with real numbers, and holding up under the actual production default (`dense`), which scores a bit higher than `hybrid` in both rewrite conditions. Not a complete fix, though: even with rewrite, a third of questions still fail to cover both sides properly (e.g. an unusual cross-sector company pairing, "JPMorgan vs Microsoft," scored low under both methods). `use_query_rewrite` stays off by default in `answer()` since it's a per-question judgment call — only worth the extra LLM call for genuine comparison questions, not a global win.

**A rate-limit lesson worth knowing if you re-run these**: the Gemini free tier's actual binding constraint is **15 requests/minute** for `gemini-3.1-flash-lite`, not the 500/day figure — running two Gemini-calling eval scripts concurrently exceeded it and crashed one outright. Run these one at a time, and expect them to be slower and less predictably paced than `n_calls × 4s` suggests (the SDK retries transient throttling internally, silently, without it surfacing as an error).

## Monitoring

Every question asked in `app.py` (the Streamlit UI) is logged to Postgres — question, answer, retrieval config used (method/rerank/rewrite/filter), and latency — plus any 👍/👎 feedback (with an optional free-text comment) and any LLM-judge scores, all linked back to that conversation. `monitoring/db.py` owns the schema (`monitoring.conversations`, `monitoring.feedback` — a separate schema from dlt's `raw` ingestion staging tables in the same shared Postgres container).

A Grafana dashboard (`monitoring/grafana/`, auto-provisioned — no manual setup needed) reads straight from that schema with **11 panels**: conversations over time, average latency over time, feedback thumbs up/down, retrieval method usage, reranking usage rate, query-rewrite usage rate, average faithfulness, average context precision, relevance distribution, a recent-questions table, and a recent-feedback-comments table.

Every answer is shown alongside a "Sources (N passages retrieved)" section — one expander per retrieved passage (ticker, sector, filing date, full text) — so you can check the answer is actually grounded in what was retrieved, not just take it on faith. Feedback is a colored confirmation box after clicking 👍/👎 (with an optional comment box bundled into the same submission), not just a caption.

There's also an opt-in **"Score this answer (LLM-as-judge)"** toggle (off by default) — turning it on scores the answer right after generation and shows it as KPI cards: **Faithfulness**, **Context Precision**, and **Relevance** (RELEVANT/PARTLY_RELEVANT/NON_RELEVANT). It's off by default deliberately: not required by the rubric (human feedback + the dashboard already satisfy that), and it adds an extra Gemini call per question — more rate-limit exposure on live usage than most casual questions warrant.

**Why three metrics instead of one blended score**: an earlier single relevance+faithfulness score turned out insensitive — the generation system prompt already forbids fabrication, so faithfulness rarely varies, and blending it with relevance obscured which dimension was actually failing. Faithfulness and context precision are RAGAS-style metrics (ported from finsignal-rag's `evaluate.py`); relevance (RELEVANT/PARTLY_RELEVANT/NON_RELEVANT) is the DTC course's own live-judge categorization (Module 5, `09-built-in-judge.md`) — checked directly against the actual course lesson files rather than assumed, since neither this project's nor finsignal-rag's RAGAS-inspired approach is what the course itself demonstrates (its Module 4 lesson `13-llm-as-judge.md` uses a different, ground-truth-answer-based method entirely). All three are computed from one Gemini call — the judge decomposes the answer into atomic claims and rates each retrieved passage's relevance, and the three numbers are computed from those structured judgments in Python, keeping the same rigor as two separate calls without the extra rate-limit cost.

```bash
uv run streamlit run app.py   # needs Postgres reachable — see Containerization below
```

```bash
uv run streamlit run app.py   # needs Postgres reachable — see Containerization below
```

**Verified end-to-end** against a running containerized stack: logged a real conversation + feedback round-trip, confirmed present in Postgres via `psql`, and confirmed all 4 distinct panel query types (timeseries, piechart, stat, table) return correct real data by querying Grafana's `/api/ds/query` directly — e.g. the reranking-usage stat panel correctly computed 100% from the one logged conversation (which had reranking on).

## Containerization

Everything runs via `docker-compose.yml`: `app` (this Streamlit UI, built from `Dockerfile`), `postgres` (shared by monitoring and dlt ingestion staging), and `grafana` (dashboard auto-provisioned from `monitoring/grafana/`). Ingestion is a separate on-demand profile, not part of the default stack:

```bash
docker compose up                                      # app + postgres + grafana
docker compose --profile ingestion run --rm ingestion  # run the dlt pipeline once, on demand
```

App: http://localhost:8501 · Grafana: http://localhost:3000 (anonymous viewer access enabled, no login needed)

**Two things worth knowing if you rebuild this image:**

1. `sentence-transformers` pulls in `torch`, whose default wheel bundles ~6GB of unused NVIDIA CUDA libraries (no GPU here). The documented `uv`-native fix (`[tool.uv.sources]` pinning `torch` to PyTorch's CPU-only wheel index) matched uv's own docs exactly but never actually took effect — confirmed via a full verbose resolver trace, after ruling out cache/env-var/config-file causes. The `Dockerfile` instead installs everything else first (`uv sync --no-install-package torch`), then installs `torch` separately from the CPU index (`uv pip install --index-url https://download.pytorch.org/whl/cpu torch`) — still fully within uv's own venv management, no plain `pip` mixed in.
2. Every `uv run` inside the built image uses `--no-sync`. Without it, `uv run` re-syncs the venv against the *full* lockfile before running anything — which silently reinstalled the default CUDA `torch` right back over the CPU-only one from step 1, and pulled the `dev` dependency group into the image, undoing both optimizations. Caught by checking `docker exec ... uv pip list` and seeing `torch==2.13.0` instead of the expected `2.14.0+cpu`.

**Verified end-to-end**: `docker compose up -d --build` brings up all 3 services correctly (postgres healthcheck gates app startup); the full RAG pipeline runs correctly inside the container; `docker compose --profile ingestion run --rm ingestion` downloaded all 6 filings and loaded 5,201 chunks into the `raw` schema — same per-ticker counts as the earlier DuckDB dry run — confirming `raw` and `monitoring` coexist correctly in the one shared Postgres container.

## Project structure

```
sec_rag_dtc_llm/
├── ingestion/       # download, chunk, embed — plus the dlt-orchestrated pipeline
├── rag/             # retrieval (dense/sparse/hybrid + rerank/rewrite/fields) + generation
├── eval/            # retrieval & LLM evaluation
├── monitoring/      # Postgres schema/logging + Grafana provisioning & dashboards
├── data/processed/  # chunks.json (committed seed data)
├── chroma_db/       # prebuilt vector store (committed seed data)
├── app.py           # Streamlit UI
├── Dockerfile
├── docker-compose.yml
└── tests/
```

## Acknowledgements

Adapted from finsignal-rag, a RAG project built for the SMM694 Applied NLP module at City St George's, University of London.
