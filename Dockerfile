# syntax=docker/dockerfile:1

# Prediction API image: production dependencies, the package, and the trained model.
# Train first (uv run python -m readmission.train), then:
#   docker build -t readmission-api .
#   docker run -p 8000:8000 readmission-api

# ---- Build stage: install dependencies and the package with uv ----
FROM python:3.13.14-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies before code: this slow layer is reused until pyproject.toml or uv.lock changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# The package itself, installed non-editable so the runtime stage doesn't need src/.
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# ---- Runtime stage: only the finished environment and the model ----
FROM python:3.13.14-slim

# XGBoost needs the OpenMP runtime, which the slim base image doesn't include.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 app

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY models/model.joblib /app/models/model.joblib

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    READMISSION_MODEL_PATH=/app/models/model.joblib

USER app
EXPOSE 8000

# The slim image has no curl, so the health check uses Python's standard library.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "readmission.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
