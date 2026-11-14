# syntax=docker/dockerfile:1.7
#   docker build --target backend  -t rag-backend  .
#   docker build --target frontend -t rag-frontend .
#djdj
ARG PYTHON_VERSION=3.14

# ---- shared base: Python slim + uv -------------------------------------------------- #
FROM python:${PYTHON_VERSION}-slim AS base
# Downloads and installs uv
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /usr/local/bin/  

# UV_COMPILE_BYTECODE: uv precompiles all Python code to bytecode, which is faster to load and execute
# UV_LINK_MODE=copy: uv copies the Python interpreter and all dependencies into the virtual environment, making it self-contained and portable
# UV_PYTHON_DOWNLOADS=never: uv does not download any additional Python versions, it uses the system Python interpreter
# PYTHONUNBUFFERED=1: Python output is not buffered, which is useful for logging and debugging
# PATH="/app/.venv/bin:$PATH": adds the virtual environment's bin directory to the PATH, so that Python and other executables can be run from anywhere in the container
ENV UV_COMPILE_BYTECODE=1 \ 
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"


WORKDIR /app

# ---- dependency builders (uv is only needed here, never in the final images) -------- #

# Downloads the packages specified for backend dependencies in pyproject.toml and uv.lock, and installs them into a virtual environment in /app/.venv. 
# The --frozen option ensures that the exact versions specified in uv.lock are installed, 
# and the --no-install-project option prevents the current project from being installed into the virtual environment. 
# The --only-group backend option restricts the installation to only the dependencies listed under the [tool.uv.groups.backend] section of pyproject.toml.
FROM base AS backend-deps
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --only-group backend


# Downloads the packages specified for frontend dependencies in pyproject.toml and uv.lock, and installs them into a virtual environment in /app/.venv.
# The --frozen option ensures that the exact versions specified in uv.lock are installed,
# and the --no-install-project option prevents the current project from being installed into the virtual environment.
# The --only-group frontend option restricts the installation to only the dependencies listed under the [tool.uv.groups.frontend] section of pyproject.toml.
FROM base AS frontend-deps
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --only-group frontend


# ---- backend: FastAPI + Speculative RAG + PostgreSQL/Redis clients ------------------- #
FROM python:${PYTHON_VERSION}-slim AS backend
ENV PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH"
WORKDIR /app
# The useradd command creates a new user named app with a home directory and a specific user ID (1000). 
# The mkdir command creates the necessary directories for the application, 
# and chown changes the ownership of the /app directory to the app user. 
# This ensures that the application runs with non-root privileges, which is a best practice for security. 
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