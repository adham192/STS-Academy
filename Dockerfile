# syntax=docker/dockerfile:1.7
#   docker build --target backend  -t rag-backend  .
#   docker build --target frontend -t rag-frontend .
ARG PYTHON_VERSION=3.14

# ---- shared base: Python slim + uv -------------------------------------------------- #
FROM python:${PYTHON_VERSION}-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"


WORKDIR /app

# ---- dependency builders (uv is only needed here, never in the final images) -------- #
FROM base AS backend-deps
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --only-group backend



FROM base AS frontend-deps
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --only-group frontend


# ---- backend: FastAPI + Speculative RAG + PostgreSQL/Redis clients ------------------- #
FROM python:${PYTHON_VERSION}-slim AS backend
ENV PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH"
WORKDIR /app
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/src/session3/chroma_db /app/src/session3/pdfs \
    && chown -R app:app /app
COPY --from=backend-deps --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app src/logging_config.py src/logging_config.py
COPY --chown=app:app src/session3/scripts/backend src/session3/scripts/backend
COPY --chown=app:app src/session3/scripts/RAG-Engine src/session3/scripts/RAG-Engine
USER app
EXPOSE 8000
CMD ["uvicorn", "api:app", "--app-dir", "src/session3/scripts/backend", "--host", "0.0.0.0", "--port", "8000"]

# ---- frontend: Streamlit UI only (no RAG engine, no database drivers) ---------------- #
FROM python:${PYTHON_VERSION}-slim AS frontend
ENV PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH"
WORKDIR /app
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/src/session3/JSON-chat-history \
    && chown -R app:app /app
COPY --from=frontend-deps --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app src/logging_config.py src/logging_config.py
COPY --chown=app:app src/session3/scripts/app src/session3/scripts/app
USER app
EXPOSE 8501
CMD ["streamlit", "run", "src/session3/scripts/app/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]