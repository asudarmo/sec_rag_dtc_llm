# Architecture: retrieval & generation

## Flow

```
question ──▶ [optional query rewrite] ──▶ retrieve (dense/sparse/hybrid)
          ──▶ [optional rerank] ──▶ build context ──▶ Gemini generate ──▶ cited answer
```

All stages are wired behind a single entry point, `rag/pipeline.py`'s `retrieve()`,
with each stage independently toggleable via config flags — this is what makes the
[evaluation](evaluation.md) sweep possible without duplicated call chains.

## Retrieval methods

- **dense** *(default)* — ChromaDB vector search (`all-MiniLM-L6-v2`)
- **sparse** — BM25 keyword search (`rank_bm25`), in-memory index over
  `data/processed/chunks.json`
- **hybrid** — dense + sparse fused with Reciprocal Rank Fusion (RRF, `rag/rrf.py`,
  generalized to N ranked lists)

`rag/generate.py`'s default is `method="dense", use_rerank=True` — the config the
[retrieval evaluation](evaluation.md) found best.

## Reranking

A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores the candidate pool
for a higher-precision final ordering. Helps every base retrieval method
substantially and consistently (see [evaluation.md](evaluation.md)), not just hybrid.

## Query rewriting

An LLM call decomposes comparison questions (e.g. "tech vs banks", "Google vs
Microsoft") into one focused sub-query per side before retrieving. This fixes a real
bias: naive retrieval for a cross-sector query is dominated by whichever sector has
more chunks in the corpus (banks have ~3x more than tech here). Off by default —
it's a per-question judgment call, only worth the extra Gemini call for genuine
comparison questions (see [evaluation.md](evaluation.md) for the quantified effect).

Query rewriting picks the right filter granularity itself: a sector-level comparison
tags each sub-query with `{"sector": ...}`; a company-level comparison tags them with
the finer-grained `{"ticker": ...}` instead.

## Filtering

Filtering uses the DTC course's own minsearch vocabulary
(`01-agentic-rag/lessons/05-search.md`): a `text_field` is tokenized/embedded and
relevance-ranked (here, just the chunk text); a `keyword_field` is an exact-match
filter applied via a `filter_dict`. `rag/fields.py` defines
`KEYWORD_FIELDS = ("ticker", "sector")`, and every retriever takes the same
`filter_dict` parameter — filter by sector (`{"sector": "tech"}`), by a specific
company (`{"ticker": "GOOGL"}`), or both together.

## Generation

`rag/generate.py` assembles retrieved chunks into a numbered context
(`[1]`, `[2]`, ...) and prompts Gemini (`gemini-3.1-flash-lite`) to answer only from
those passages, citing the `[number]` after each claim. This is what makes answers
traceable back to their source filing text rather than fabricated.

## CLI examples

```bash
# Retrieval only, no API key needed — swap the module for dense/sparse/hybrid
uv run python -m rag.retrieve_dense "What are the main AI-related risk factors?" --k 5 --sector tech
uv run python -m rag.retrieve_sparse "What are the main AI-related risk factors?" --k 5 --ticker GOOGL
uv run python -m rag.hybrid "What are the main AI-related risk factors?" --k 5 --ticker GOOGL
uv run python -m rag.rerank "What are the main AI-related risk factors?" --k 5 --sector tech

# Full retrieve + generate (needs GEMINI_API_KEY) — method/rerank/rewrite all toggleable
uv run python -m rag.generate "Compare the main AI-related risk factors between tech and banking companies" --k 6
uv run python -m rag.generate "Compare Google vs Microsoft's AI risk factors" --k 6 --method hybrid --rewrite
```

For the agentic RAG alternative to this fixed pipeline, see
[agentic-rag.md](agentic-rag.md).
