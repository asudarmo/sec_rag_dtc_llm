# Changelog

Development history for this project — real bugs found and fixed, notable decisions,
and lessons learned along the way. For how the system currently works and how to run
it, see [README.md](README.md) and the linked [docs/](docs) files instead; this file
is background for anyone curious about the "why," not required reading.

## Retrieval pipeline

- **Query rewriting didn't scope retrieval to sector.** Rewriting split a comparison
  question's *wording* per sector (e.g. "tech vs banks" → two sub-queries) but didn't
  restrict *retrieval* to that sector, so a sub-query could still be crowded out by
  the larger sector's chunks. Fixed by having `query_rewrite.rewrite()` tag each
  sub-query with its sector/ticker, and `pipeline.py` apply that filter. Before the
  fix, the tech side of a cross-sector answer surfaced generic R&D-expense text;
  after, it surfaced genuinely on-topic AI-risk content.
- **Rewrite + rerank together could still return an imbalanced result** (one observed
  case: 5 bank passages vs. 1 tech, despite rewrite being on). Reranking was
  re-scoring the *entire merged pool* against the undecomposed question and picking a
  flat top-k — undoing the balance the per-sub-query retrieval had guaranteed going
  in. Fixed by reranking each sub-query's pool independently and taking a fixed quota
  from each. Verified with the exact reported query: 3 tech / 3 banks (was 5/1).
- **Filtering vocabulary aligned with the DTC course's own minsearch terms**
  (`text_fields`/`keyword_fields`/`filter_dict`, `01-agentic-rag/lessons/05-search.md`)
  rather than a project-specific `sector`-only parameter. `rag/fields.py` now defines
  `KEYWORD_FIELDS = ("ticker", "sector")`, and every retriever takes the same
  `filter_dict` parameter. Ticker-level filtering fell out of this for free (the field
  was already in chunk metadata, just never exposed).
- **Reciprocal Rank Fusion generalized to N ranked lists** (`rag/rrf.py`), matching
  the course's own function shape, and reused to properly fuse the multiple
  per-sub-query lists that query rewriting produces — replacing what had been a naive
  "first occurrence wins" dict union.
- **Course-alignment note**: the DTC course's own "reranking" lesson (Module 6,
  `06-best-practices`) turns out to just be RRF again — it never uses a cross-encoder.
  This project's `rerank.py` cross-encoder is therefore a genuinely different
  technique from `hybrid.py`'s RRF, not a re-skin of the same idea — useful since the
  rubric grades hybrid search and reranking as separate bonus points.

## Evaluation

- **Retrieval evaluation sweep extended** from the originally-planned 4 configs
  (dense/sparse/hybrid/hybrid+rerank) to a full 6-config method×rerank sweep, once it
  was clear the eval makes zero LLM calls and is cheap to run either way. Result:
  `dense+rerank` won, narrowly ahead of `hybrid+rerank`.
- **LLM-judge evaluation expanded from a 1-D to a 2×2 sweep** (`method`: hybrid/dense
  × `use_query_rewrite`: on/off), after the retrieval eval found `dense+rerank` beat
  `hybrid+rerank` — to check whether the original rewrite conclusion (measured only
  against hybrid) still held against the new default. It did, and strengthened:
  `dense_with_rewrite` scored highest (0.827 vs. 0.760 for `hybrid_with_rewrite`).
  Existing `hybrid_*` results were migrated in place rather than recomputed, so the
  checkpoint/resume mechanism only had to compute the 30 new `dense_*` pairs.
- **Rate-limit lesson**: the Gemini free tier's actual binding constraint is **15
  requests/minute** for `gemini-3.1-flash-lite`, not the 500/day figure assumed during
  planning. Running two Gemini-calling scripts concurrently exceeded it and crashed
  one outright once its retry budget was exhausted fighting the other for quota — now
  a hard rule: never run two Gemini-calling scripts at once. Hitting the limit even
  solo causes silent internal SDK retries (not raised as exceptions), so eval scripts
  run slower and less predictably than `n_calls × sleep_seconds` suggests. Every eval
  script now checkpoints after each item and resumes from where it left off, since an
  earlier version without this lost all progress on a crash.

## Agentic RAG

- **Citation-numbering bug, live-reported and fixed**: on multi-search comparison
  questions, the agent's citations for the *second* search were off by exactly the
  size of the first search's result set (e.g. banking claims cited `[1]`-`[5]` when
  the Sources panel showed them at `[6]`-`[10]`). Each tool call had been numbering
  its own passages `[1]`-`[N]` independently, while the final Sources display numbers
  everything globally in first-seen order. Fixed by giving `rag/agent.py` a shared
  citation-number map that persists across every tool call in one run, plus a
  regression test asserting the second call's numbering continues from the first's.
- **"Compare both" mode added**, after live-testing showed switching the mode toggle
  back and forth made side-by-side comparison awkward. Traditional and agentic now
  run concurrently via a thread pool (~max(latency) instead of the sum, verified at
  23.5s wall-clock vs. an expected ~38.7s sequential), with a per-column loading
  indicator that flips from "⏳" to "✅ Done in X.Xs" as each side finishes.

## Monitoring & UI

- **Sources display regression, found while verifying an answer's grounding**:
  `app.py` called the `rag.generate.answer()` convenience wrapper, which retrieves
  and generates internally but only returns the answer string, discarding the
  retrieved passages. Fixed by having `app.py` call `retrieve()` + `generate_answer()`
  directly so the passages are available to display.
- **Judge metric redesign**: an original single blended relevance+faithfulness score
  was insensitive — the generation prompt already forbids fabrication, so faithfulness
  rarely varies, and blending obscured which dimension was actually failing. Checked
  the DTC course's own two different judge approaches (Module 4's ground-truth-based
  semantic-equivalence method vs. Module 5's live RELEVANT/PARTLY_RELEVANT/NON_RELEVANT
  classification) before landing on three separate metrics computed from one Gemini
  call: **faithfulness**, **context_precision** (both RAGAS-style), and **relevance**
  (the course's own categorical metric). Verified live: faithfulness scored 1.0 (as
  expected) while context_precision scored 0.5 on the same answer — exactly the
  discriminating signal the old blended score couldn't surface.
- **UI polish round** (prompted by live use): judge scores became KPI cards instead of
  a caption; a feedback comment box was bundled into the same submission as
  thumbs-up/down; the button row is replaced by a colored confirmation box after
  voting.

## Containerization

- **CPU-only `torch` in the Docker image**: `sentence-transformers` pulls in `torch`
  transitively, whose default wheel bundles ~6GB of unused NVIDIA CUDA libraries (no
  GPU in this deployment). The documented uv-native fix (`[tool.uv.sources]` pinning
  torch to PyTorch's CPU-only index in `pyproject.toml`) matched uv's own docs exactly
  but never took effect, even after ruling out cache/env-var/config-file causes via a
  full verbose resolver trace. Resolved at the CLI level instead: the `Dockerfile`
  runs `uv sync --no-install-package torch`, then installs torch separately from the
  CPU-only index — still fully within uv's own venv management.
- **A second bug this surfaced**: `uv run` without `--no-sync` re-syncs the venv
  against the *full* lockfile before running anything, which silently reinstalled the
  default CUDA-inclusive torch right back over the CPU-only build, and pulled the
  `dev` dependency group into the runtime image. Every `uv run` inside the built image
  now uses `--no-sync`.
