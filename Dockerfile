# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

# Install everything except torch first, then install torch separately from
# PyTorch's CPU-only wheel index (confirmed working via this CLI form) — the
# pyproject.toml-native [tool.uv.sources] pin didn't take effect for this uv
# version for reasons not fully root-caused (see PLAN.md), so this sidesteps it
# rather than shipping the default wheel's ~6GB of unused NVIDIA CUDA libs.
RUN uv sync --frozen --no-dev --no-install-package torch
RUN uv pip install --index-url https://download.pytorch.org/whl/cpu torch

# Pre-download the embedding + reranker models into the image so the container
# doesn't fetch them from Hugging Face on cold start. --no-sync is essential here:
# without it, `uv run` re-syncs the venv against the full lockfile first, which
# silently reinstalls the default (CUDA-inclusive) torch over the CPU-only one
# just installed above, and pulls in the dev dependency group too — undoing both
# `--no-install-package torch` and `--no-dev` from the sync step. (Caught this by
# checking `uv pip list` inside the built container — torch had reverted to
# 2.13.0 instead of the CPU build, and duckdb/pytest had appeared unbidden.)
RUN uv run --no-sync python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN uv run --no-sync python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy app code and the prebuilt vector store (chroma_db/ is read-only at
# runtime). data/raw/ is excluded via .dockerignore (large, re-downloadable).
COPY . .

# --no-sync for the same reason as above — the container must never re-sync
# against the full lockfile at runtime (would undo the CPU-only torch install
# and require network access to Hugging Face/PyPI on every container start).
EXPOSE 8501
CMD ["uv", "run", "--no-sync", "streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
