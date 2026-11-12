# STS-Academy

Course projects for the STS Academy, one branch per session.

## Branches

| Branch | What it contains |
|---|---|
| `main` | Guide to each branch |
| `session1` | Hosting Model on cloud |
| `session1-assignments` | Hosting Model locally and on cloud |
| `session2` | PDF embedding and retrieval with ChromaDB and a local Ollama model |
| `session2-assignments` | Multi-PDF RAG assistant with Streamlit UI, document routing, and quantitative evaluation |
| `session3` | Speculative RAG assistant with parallel Ollama drafts, Gemini verification, Streamlit chat UI, central logging, and stage-by-stage evaluation against a baseline |
| `session4` | FastAPI chatbot API with Pydantic input and output validation, per-request logging to separate files, and a switchable LLM (Gemini in the cloud or a local Ollama model) |
| `session4-assignments` | Extends the session 4 chatbot API with a PostgreSQL database (every question and answer is stored per session and can be listed, read back, and deleted), a Redis cache for repeated questions, and Docker, so the API, PostgreSQL, and Redis start together with one command |

## Project structure (`session4` branch)

```
STS-Academy/
├── src/
│   ├── logging_config.py                               # logging setup from session 3 (not used by session 4)
│   └── session4/
│       ├── app/
│       │   ├── main.py                                 # builds the FastAPI app and wires everything together
│       │   ├── api/
│       │   │   └── routes/
│       │   │       ├── chat.py                         # POST /api/v1/chat
│       │   │       └── health.py                       # GET /health
│       │   ├── core/
│       │   │   ├── config.py                           # settings read from .env (validated with Pydantic)
│       │   │   └── logging_config.py                   # logging.config.dictConfig, log files and request ID
│       │   ├── middleware/
│       │   │   └── request_logging.py                  # request ID and one access-log line per request
│       │   ├── schemas/
│       │   │   └── chat.py                             # ChatRequest and ChatResponse (Pydantic)
│       │   └── services/
│       │       ├── chat_service.py                     # takes a question, asks the LLM, returns the answer
│       │       └── llm_clients.py                      # Gemini client, Ollama client, and the provider switch
│       ├── logs/                                       # app.log, error.log, access.log (created at runtime)
│       └── scripts/
│           └── run_dev.py                              # starts the API server
├── .env.example                                        # template for environment variables
├── .gitignore
├── .python-version
├── pyproject.toml                                      # project dependencies (used by uv)
├── uv.lock                                             # exact dependency versions (used by uv)
└── README.md
```

## What session 4 does

This branch turns a question-and-answer chatbot into a small **FastAPI service**. You send a question over HTTP and an LLM sends an answer back. The focus is on a clean folder structure, strict input and output validation, and a logging system that records every event in its own file from the moment the app starts.

**`main.py`** builds the application:

1. Sets up logging first, so everything that happens afterwards is captured.
2. Builds the LLM client at startup, so a missing key or model stops the server immediately with a clear error.
3. Adds the request-logging middleware and registers the routes under `/api/v1`.

**`logging_config.py`** (in `core/`) is called once at startup and writes to three rotating files (5 MB each, 5 backups kept) in `logs/`:

- `app.log`: everything from DEBUG and above, including uvicorn's own messages.
- `error.log`: errors and tracebacks only.
- `access.log`: one line per HTTP request, with method, path, status code, and duration.

Every line has the same format: `timestamp | level | request_id | logger:line | message`. Loggers are named `app.*` (your code), `app.access` (requests), and `uvicorn`.

**`request_logging.py`** gives every request a short ID, measures how long it takes, and logs it. The ID appears on every log line written during that request and is returned in the `X-Request-ID` response header, so one request can be followed across all the log files. Unhandled errors are logged with their full traceback.

**`schemas/chat.py`** validates data in and out with Pydantic:

- **Input (`ChatRequest`):** `session_id` must be 1 to 64 characters (letters, digits, `_` or `-`), and `message` must be 1 to 1000 characters. Surrounding whitespace is stripped first. Invalid input is rejected with a `422` before any application code runs.
- **Output (`ChatResponse`):** `session_id`, `answer` (1 to 10000 characters), and a UTC `timestamp`. The route uses it as its `response_model`, so a malformed response is never sent to the client.

**`chat_service.py`** and **`llm_clients.py`** answer the question:

- **Gemini (cloud):** called over HTTP with your API key.
- **Ollama (local):** called on your local Ollama server.
- `LLM_PROVIDER` in `.env` chooses which one is used. If the LLM fails or returns an empty answer, the API returns a `502` and the real cause is written to `error.log`.
- The question and answer text is logged at DEBUG level only. At the default INFO level, the logs record lengths and timings.

**`config.py`** reads all settings from `.env` and validates them. The API key is stored as a secret, so it never appears in logs.

## Project structure (`session4-assignments` branch)

```
STS-Academy/
├── src/
│   ├── logging_config.py                               # logging setup from session 3 (not used by session 4)
│   └── session4-assignments/
│       ├── app/
│       │   ├── main.py                                 # builds the FastAPI app, creates the tables, wires everything together
│       │   ├── api/
│       │   │   ├── deps.py                             # dependencies: database session, Redis client, LLM client, ChatService
│       │   │   ├── errors.py                           # turns LLM failures into a 502 response
│       │   │   └── routes/
│       │   │       ├── chat.py                         # chat endpoints under /api/v1/chat
│       │   │       └── health.py                       # GET /health (also checks the database)
│       │   ├── core/
│       │   │   ├── config.py                           # settings read from .env (validated with Pydantic)
│       │   │   └── logging_config.py                   # logging.config.dictConfig, log files and request ID
│       │   ├── db/
│       │   │   ├── base.py                             # SQLAlchemy Base class shared by all models
│       │   │   ├── redis_client.py                     # one shared Redis client built from REDIS_URL
│       │   │   └── session.py                          # engine, per-request session, table creation at startup
│       │   ├── middleware/
│       │   │   └── request_logging.py                  # request ID and one access-log line per request
│       │   ├── models/
│       │   │   └── chat_message.py                     # ChatMessage table (one question and its answer)
│       │   ├── schemas/
│       │   │   ├── chat.py                             # ChatRequest, ChatResponse, ChatHistoryResponse, ChatSessionSummary
│       │   │   └── health.py                           # HealthResponse
│       │   └── services/
│       │       ├── chat_service.py                     # asks the LLM (with Redis cache) and reads/writes conversations in PostgreSQL
│       │       └── llm_clients.py                      # Gemini client, Ollama client, and the provider switch
│       ├── logs/                                       # app.log, error.log, access.log (created at runtime)
│       └── scripts/
│           └── run_dev.py                              # starts the API server (host, port, and reload come from the environment)
├── .dockerignore                                       # files kept out of the Docker build
├── .env.example                                        # template for environment variables
├── .gitignore
├── .python-version
├── Dockerfile                                          # builds the backend image
├── docker-compose.yml                                  # runs the API, PostgreSQL, and Redis together
├── pyproject.toml                                      # project dependencies (used by uv)
├── uv.lock                                             # exact dependency versions (used by uv)
└── README.md
```

## What session 4 assignments does

This branch builds on the session 4 chatbot API and adds **memory**, a **cache**, and **containers**. Every question and the answer the LLM gave are saved in a **PostgreSQL** database under a `session_id`, so a conversation can be listed, read back later, and deleted. A **Redis** cache remembers answers, so a repeated question is answered without calling the LLM again. **Docker Compose** starts the API, PostgreSQL, and Redis together. The code is split into layers: routes only handle HTTP, the service holds the logic, and the database, cache, and LLM code live in their own modules.

**`main.py`** builds the application:

1. Sets up logging first, so everything that happens afterwards is captured.
2. Builds the LLM client at startup, so a missing key or model stops the server immediately with a clear error.
3. Creates the database tables on startup if they do not exist yet.
4. Adds the request-logging middleware, registers the error handler, and mounts the health route at `/health` and the chat routes under `/api/v1`.

**`db/`** connects the app to PostgreSQL and Redis:

- `session.py` creates the SQLAlchemy engine from `DATABASE_URL` (with `pool_pre_ping`, so dead connections are dropped after a database restart) and gives every request its own database session.
- `base.py` holds the `Base` class that all models inherit from.
- `redis_client.py` builds one shared Redis client from `REDIS_URL`. It connects lazily and uses 1-second timeouts, so a missing Redis never makes a request hang.
- Passwords are hidden when the connection is logged.

**`models/chat_message.py`** defines the `chat_messages` table. Each row is one exchange:

| Column | Type | Meaning |
|---|---|---|
| `id` | integer, primary key | Row number, also used to keep history in order |
| `session_id` | string (64), indexed | The conversation the exchange belongs to |
| `question` | text | What the user asked |
| `answer` | text | What the LLM answered |
| `created_at` | timestamp with time zone | When the exchange was saved (UTC) |

**`services/chat_service.py`** (`ChatService`) contains all the logic:

- `ask`: gets an answer (from the Redis cache or the LLM), rejects an empty answer, saves the exchange, and returns it.
- `list_sessions`: lists conversations with their message count and the time of the latest message, newest first, with pagination.
- `history`: returns the stored exchanges of one session, oldest first, with pagination.
- `clear`: deletes all messages of one session and returns how many rows were removed.

**`services/llm_clients.py`** talks to the model:

- **Gemini (cloud):** called over HTTP with your API key. The answer is joined from all response parts, so long answers are not cut after the first part.
- **Ollama (local):** called on your local Ollama server.
- Both clients set an output limit of 4096 tokens and write a warning to `app.log` if the model stopped for any reason other than a normal finish (for example `MAX_TOKENS` or `SAFETY`). If an answer ever looks cut off, that line shows why.
- `LLM_PROVIDER` in `.env` chooses which one is used. A failed call raises `LLMError`.

**`api/`** is the HTTP layer:

- `routes/chat.py` has the chat endpoints (table below).
- `routes/health.py` runs `SELECT 1` against the database. It returns `503` if the database cannot be reached. Redis is not part of the health check, because the app works without it.
- `deps.py` builds a `ChatService` for each request from the database session, the Redis client, and the shared LLM client.
- `errors.py` turns an `LLMError` into a `502` with a safe message. The real cause is already in `error.log`.

**`schemas/`** validates data in and out with Pydantic:

- **Input (`ChatRequest`):** `session_id` must be 1 to 64 characters (letters, digits, `_` or `-`), and `message` must be 1 to 1000 characters. Surrounding whitespace is stripped first. The same `session_id` rule is applied to the URL path of the history and delete endpoints. Invalid input is rejected with a `422`.
- **Output (`ChatResponse`):** `id`, `session_id`, `question`, `answer` (1 to 10000 characters), and `created_at`.
- `ChatHistoryResponse` wraps a session's messages with its `session_id` and a `count`. `ChatSessionSummary` is one row of the conversation list. `HealthResponse` has `status` and `database`.

**`core/` and `middleware/`** work as in session 4: settings come from `.env`, and three rotating log files (`app.log`, `error.log`, `access.log`) use the format `timestamp | level | request_id | logger:line | message`. Logs are also printed to the console, so they show up in `docker compose logs`. Every request gets an ID, returned in the `X-Request-ID` header. SQL statements are only logged when `LOG_LEVEL=DEBUG`.

### Redis cache

Calling an LLM is slow and can cost money, so the answer to each question is cached in Redis:

1. The question is lowercased and hashed (SHA-256). The cache key is `answer:<hash>`.
2. If the key exists, the cached answer is returned and the LLM is **not** called (`Cache hit` in `app.log`).
3. If not, the LLM is called and the answer is stored with an expiry of `CACHE_TTL_SECONDS` (default 3600 seconds, one hour) (`Cache stored` in `app.log`).
4. The exchange is saved in PostgreSQL either way, so the history stays complete even when the answer came from the cache.

Things to know:

- The cache is shared by all sessions: the same question from two users gets the same answer. The question is the only part of the key, so the conversation is not used as context.
- If Redis is down or slow, the app logs a warning (`Redis unavailable, skipping cache read/write`) and calls the LLM as usual. Nothing fails.

### Endpoints

| Method | Path | What it does | Success |
|---|---|---|---|
| `POST` | `/api/v1/chat` | Ask the LLM (or the cache) and save the question and answer | `201` |
| `GET` | `/api/v1/chat/sessions` | List conversations (`limit` 1 to 100, `offset`) | `200` |
| `GET` | `/api/v1/chat/{session_id}/history` | Read one conversation, oldest first (`limit`, `offset`) | `200` |
| `DELETE` | `/api/v1/chat/{session_id}` | Delete one conversation | `204` |
| `GET` | `/health` | Check that the API and the database are up | `200` |

| Status | Meaning |
|---|---|
| `201` | Question answered and saved |
| `204` | Conversation deleted |
| `404` | Deleting a session that has no history |
| `422` | Invalid input (rejected by Pydantic) |
| `502` | The LLM failed or returned an empty answer |
| `503` | The database is unreachable (health check) |

### Docker

The backend is packaged as a Docker image, and `docker-compose.yml` runs three containers together:

| Service | Container | Image | Port on your machine | Purpose |
|---|---|---|---|---|
| `postgres` | `chatbot-postgres` | `postgres:18` | `POSTGRES_PORT` (default 5432) | Stores the conversations |
| `redis` | `chatbot-redis` | `redis:7-alpine` | `6379` | Caches answers |
| `api` | `chatbot-api` | built from `Dockerfile` | `8000` | The FastAPI backend |

How it fits together:

- **One network.** Compose puts the three containers on a private network where each service name is a hostname. Inside the `api` container, the database is at `postgres:5432` and the cache is at `redis:6379`. For this reason, compose overrides `DATABASE_URL` and `REDIS_URL` with those addresses, and the `localhost` values in `.env` are only used when you run the app directly on your machine. The ports in the table are only for reaching the containers from your machine, and container-to-container traffic always uses the internal ports (5432 and 6379).
- **Start order.** Postgres and Redis have health checks, and the `api` container starts only after both report healthy. The API needs the database at startup, so this prevents a crash on a database that is still booting.
- **Credentials.** `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB` from `.env` are used twice: the Postgres container creates that user and database the first time it starts, and compose builds the API's `DATABASE_URL` from the same values, so the two always match.
- **Data survives restarts.** The database and cache are stored in the named volumes `pgdata` and `redisdata`.
- **The image.** The `Dockerfile` installs the dependencies from `uv.lock` first (so code changes rebuild in seconds), copies only `src/session4-assignments`, and runs the app as a non-root user. `.env` is never copied into the image, and the values are passed in when the container starts. The container reads `APP_HOST=0.0.0.0` (so it accepts outside connections) and `APP_RELOAD=false` (no auto-reload).

## Installation (using uv)

### 1. Requirements

- [Docker](https://docs.docker.com/get-docker/) with Docker Compose (to run everything in containers)
- [uv](https://docs.astral.sh/uv/) (manages virtual environments and Python versions, to run the app without Docker)
- A [Google Gemini](https://aistudio.google.com/) API key (if you use `LLM_PROVIDER=gemini`)
- [Ollama](https://ollama.com/) installed and running locally (if you use `LLM_PROVIDER=ollama`)
- A running [PostgreSQL](https://www.postgresql.org/) server and a [Redis](https://redis.io/) server (only to run without Docker, Compose provides both)

Install `uv` if you do not have it:

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows:

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Get the project

```bash
git clone <repo-url>
cd STS-Academy
git checkout session4-assignments
```

### 3. Install dependencies (without Docker)

```bash
uv sync
```

This sets up the virtual environment (`.venv`) and installs the required packages defined in `pyproject.toml` and `uv.lock`. If you only use Docker, skip this step.

### 4. Set up your `.env` file

From the repository root, copy the environment template:

```bash
cp .env.example .env
```

Fill in the values for the LLM you want to use. Only the values for the chosen provider are required, and the app stops with a clear error at startup if one is missing:

```ini
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL=your-gemini-model
OLLAMA_HOST=http://localhost:11434
LOCAL_MODEL=your-ollama-model
```

(The older names `api-key` and `model` still work.)

Set the PostgreSQL and Redis connection:

```ini
POSTGRES_USER=chatbot
POSTGRES_PASSWORD=your-password
POSTGRES_DB=chatbot
POSTGRES_PORT=5432
DATABASE_URL=postgresql+psycopg://chatbot:your-password@localhost:5432/chatbot
REDIS_URL=redis://localhost:6379/0
```

`POSTGRES_PORT` is the port published on your machine, so it must match the port in `DATABASE_URL`. With Docker, the containers ignore `DATABASE_URL` and `REDIS_URL` and use the service names instead (see [Docker](#docker)). If your password contains characters such as `@`, `/`, or `#`, they must be URL-encoded in `DATABASE_URL` (`@` becomes `%40`). Also, a `$` in `.env` must be written as `$$`, because Compose treats `$name` as a variable.

Optional settings (defaults shown):

```ini
APP_NAME=STS Chatbot
LOG_LEVEL=INFO
LLM_TIMEOUT=60
CACHE_TTL_SECONDS=3600
```

Set `LLM_PROVIDER=ollama` to use the local model instead of Gemini. Never commit `.env` files to git.

### 5. Start Ollama and pull the model (local provider only)

Make sure the Ollama daemon is running, then pull the LLM specified in your `LOCAL_MODEL`:

```bash
ollama pull <your-ollama-model>
```

If the API runs in Docker, it cannot reach Ollama at `localhost`, because that is the container itself. Set this in `.env` instead:

```ini
OLLAMA_HOST=http://host.docker.internal:11434
```

On Linux, Ollama must also listen on all interfaces (start it with `OLLAMA_HOST=0.0.0.0`), or the container cannot connect to it.

### 6. Create the PostgreSQL database (without Docker only)

With Docker, Postgres creates the user and database from your `.env` the first time it starts, so skip this step. Without Docker, create the user and database that match your `DATABASE_URL`, for example:

```sql
CREATE USER chatbot WITH PASSWORD 'your-password';
CREATE DATABASE chatbot OWNER chatbot;
```

You do not need to create the tables. The app creates `chat_messages` automatically when it starts.

## Running the project

There are two ways to run it. Docker is the simplest, because it starts the API, PostgreSQL, and Redis together.

### Option A: Run everything with Docker (recommended)

Run from the repository root:

```bash
docker compose up -d --build
```

The first build takes a few minutes while the dependencies install. When it finishes, check that all three containers are up:

```bash
docker compose ps
```

`chatbot-postgres` and `chatbot-redis` should show `healthy`, and `chatbot-api` should be `running`. The API is now at `http://localhost:8000`.

Useful commands:

```bash
docker compose logs -f api                           # follow the backend logs
docker compose exec postgres psql -U chatbot -d chatbot   # open a SQL shell (use your POSTGRES_USER and POSTGRES_DB)
docker compose exec redis redis-cli                  # open the Redis shell (KEYS * lists cached answers)
docker compose down                                  # stop everything, keep the data
docker compose down -v                               # stop everything and delete the data
```

Postgres reads its user and password only the first time it starts with an empty volume. If you change them in `.env` later, run `docker compose down -v` once so the database is created again with the new values.

### Option B: Run the API on your machine

Start PostgreSQL and Redis first (your own servers, or just those two services from Compose with `docker compose up -d postgres redis`), then run from inside the project directory:

```bash
cd src/session4-assignments
uv run python scripts/run_dev.py
```

(on the `session4` branch: `cd src/session4`)

- The server runs at `http://127.0.0.1:8000` and reloads automatically when you edit files in `app/`.
- `APP_HOST`, `APP_PORT`, and `APP_RELOAD` can be set to change this. The defaults are `127.0.0.1`, `8000`, and `true`, and the Docker image sets them to `0.0.0.0`, `8000`, and `false`.
- Startup logs show the chosen LLM provider and model, and that the database is ready.

The rest of this section works the same for both options.

### 1. Ask a question

Open the interactive docs at `http://localhost:8000/docs`, expand `POST /api/v1/chat`, click **Try it out**, and send:

```json
{"session_id": "user-1", "message": "What is FastAPI?"}
```

Or from a second terminal:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user-1", "message": "What is FastAPI?"}'
```

The response has this shape (`session4-assignments`):

```json
{
  "id": 1,
  "session_id": "user-1",
  "question": "What is FastAPI?",
  "answer": "...",
  "created_at": "2026-11-09T19:32:37.470930Z"
}
```

### 2. Try the cache

Send the same question again, even from a different `session_id`. The answer comes back almost instantly, because it is read from Redis instead of the LLM. `app.log` shows `Cache stored` after the first request and `Cache hit` after the second. Both exchanges are saved in PostgreSQL.

### 3. Read and delete conversations

List all conversations:

```bash
curl "http://localhost:8000/api/v1/chat/sessions?limit=10&offset=0"
```

Read one conversation, oldest message first:

```bash
curl http://localhost:8000/api/v1/chat/user-1/history
```

Delete a conversation (returns `204`, or `404` if it does not exist):

```bash
curl -X DELETE http://localhost:8000/api/v1/chat/user-1
```

Deleting a conversation does not remove its answers from the Redis cache. They expire after `CACHE_TTL_SECONDS`.

### 4. Try the validation

Send a request that breaks the rules, for example an empty message or a `session_id` with spaces:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "bad id!", "message": ""}'
```

The API returns a `422` that lists each failing field, and the request still appears in `access.log`.

### 5. Check that the server is alive

```bash
curl http://localhost:8000/health
```

On `session4-assignments` this also checks the database and returns `{"status": "ok", "database": "up"}`.

### 6. Read the logs

With Docker, the logs are printed to the console, so use:

```bash
docker compose logs -f api
```

The three files are also written inside the container (and in `logs/` when you run without Docker):

```bash
docker compose exec api tail -f /app/src/session4-assignments/logs/app.log    # with Docker
tail -f logs/app.log                                                           # without Docker
tail -f logs/error.log
tail -f logs/access.log
```

The `logs/` folder is created automatically the first time the server starts. The log files inside a container are removed when the container is recreated. To follow a single request across every file, take its ID from the `X-Request-ID` response header (or from any log line) and search for it:

```bash
grep -h "<request-id>" logs/*.log
```

### Troubleshooting

| Problem | Likely cause |
|---|---|
| `chatbot-postgres` is `unhealthy` and keeps restarting | Check `docker compose logs postgres`. A missing `POSTGRES_PASSWORD` in `.env`, or a port already in use on your machine, are the usual causes. |
| `password authentication failed` | The password in `.env` changed after the database was first created. Run `docker compose down -v` and start again. |
| `connection refused` to Postgres from your machine | `POSTGRES_PORT` and the port in `DATABASE_URL` do not match. |
| `Redis unavailable, skipping cache` in the log | Redis is not running or `REDIS_URL` is wrong. The app still works, it just calls the LLM every time. |
| `502` on `POST /api/v1/chat` | The LLM failed (wrong key or model, Ollama not reachable, or Gemini is overloaded). The cause is in `error.log`. |
