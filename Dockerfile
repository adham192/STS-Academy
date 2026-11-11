# Build from the repository root:  docker build -t chatbot-api .
FROM python:3.14-slim

# uv (the package manager this repo uses), copied from its official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# libpq is the PostgreSQL client library that psycopg loads at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    APP_HOST=0.0.0.0 \
    APP_RELOAD=false

WORKDIR /app

# 1) Install dependencies first. This layer is rebuilt only when
#    pyproject.toml or uv.lock change, so code edits stay fast to rebuild.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) Copy only the code this service needs
COPY src/session4-assignments ./src/session4-assignments

# 3) Run as a normal user, not root. The app writes its logs inside this folder.
RUN useradd --create-home appuser \
    && mkdir -p /app/src/session4-assignments/logs \
    && chown -R appuser:appuser /app/src/session4-assignments/logs
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "src/session4-assignments/scripts/run_dev.py"]