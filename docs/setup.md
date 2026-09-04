# Setup & running

Two ways to run this project: **Docker Compose** (everything — app, Postgres,
Grafana — recommended, satisfies the containerization rubric criterion directly), or
**local with `uv`** (faster iteration on the code itself, no monitoring stack).

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose (for the
  recommended path)
- Python 3.12 and [uv](https://docs.astral.sh/uv/) (for local/manual runs)
- A free [Gemini API key](https://aistudio.google.com/apikey)

## Configuration

```bash
cp .env.example .env
```

| Variable | Required for | Notes |
|---|---|---|
| `GEMINI_API_KEY` | generation, evaluation, agentic RAG | Free tier key at https://aistudio.google.com/apikey |
| `SEC_USER_NAME`, `SEC_USER_EMAIL` | re-running ingestion | Required by SEC EDGAR's fair-access policy — use your real name/email. Not needed just to run the app: a prebuilt vector store is committed (see [ingestion.md](ingestion.md)). |
| `POSTGRES_*` | monitoring + ingestion staging | Defaults already match the docker-compose `postgres` service — only change these if running Postgres somewhere else. |

## Option A: Docker Compose (recommended)

```bash
docker compose up -d --build
```

Brings up three services:

| Service | URL | Notes |
|---|---|---|
| `app` | http://localhost:8501 | Streamlit UI |
| `postgres` | localhost:5432 | Shared by monitoring and dlt ingestion staging (separate schemas) |
| `grafana` | http://localhost:3000 | Dashboard auto-provisioned, anonymous viewer access enabled — no login needed |

The `app` container waits for Postgres's healthcheck before starting. The prebuilt
vector store (`chroma_db/`, `data/processed/chunks.json`) is baked into the image, so
the app works immediately with no ingestion step required.

Automated ingestion (re-downloading and re-chunking the SEC filings — see
[ingestion.md](ingestion.md)) is a separate on-demand profile, not part of the
default stack, since it only needs to run once:

```bash
docker compose --profile ingestion run --rm ingestion
```

## Option B: Local with `uv`

```bash
uv sync              # installs all dependencies from pyproject.toml / uv.lock into .venv
uv run streamlit run app.py
```

Monitoring won't work without a reachable Postgres — either point `POSTGRES_*` in
`.env` at one you're running yourself, or just use `docker compose up -d postgres`
alongside a local `uv run streamlit run app.py`.

Run any script in the project with `uv run <command>`, e.g.:

```bash
uv run python -m rag.generate "What are Google's main AI risk factors?"
```

## Rebuilding the Docker image

If you change dependencies and rebuild, two things are handled deliberately and worth
knowing about (both explained in full in [CHANGELOG.md](../CHANGELOG.md)):

1. `torch` (a transitive dependency of `sentence-transformers`) is installed from
   PyTorch's CPU-only wheel index rather than the default ~6GB CUDA-inclusive wheel —
   done via a separate `uv pip install --index-url ...` step in the `Dockerfile`, not
   a `pyproject.toml` source pin (which didn't take effect for this uv version).
2. Every `uv run` inside the built image uses `--no-sync`, so the container never
   re-syncs against the full lockfile at runtime (which would silently undo #1 and
   pull in dev-only dependencies).

## Reproducibility checklist

- All dependencies are pinned via `pyproject.toml` + `uv.lock`.
- The vector store and processed chunks are committed, so a fresh clone works without
  any ingestion step.
- `.env.example` matches exactly what the code reads — no stale/renamed variables.
- `docker compose up -d --build` from a clean clone (plus `.env` filled in) brings up
  the full stack with no manual steps.
