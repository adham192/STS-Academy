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
| `session4-assignments` | Extends the session 4 chatbot API with a PostgreSQL database: every question and answer is stored per session and can be listed, read back, and deleted, with a health check that also tests the database |

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
│       │   │   ├── deps.py                             # dependencies: database session, LLM client, ChatService
│       │   │   ├── errors.py                           # turns LLM failures into a 502 response
│       │   │   └── routes/
│       │   │       ├── chat.py                         # chat endpoints under /api/v1/chat
│       │   │       └── health.py                       # GET /health (also checks the database)
│       │   ├── core/
│       │   │   ├── config.py                           # settings read from .env (validated with Pydantic)
│       │   │   └── logging_config.py                   # logging.config.dictConfig, log files and request ID
│       │   ├── db/
│       │   │   ├── base.py                             # SQLAlchemy Base class shared by all models
│       │   │   └── session.py                          # engine, per-request session, table creation at startup
│       │   ├── middleware/
│       │   │   └── request_logging.py                  # request ID and one access-log line per request
│       │   ├── models/
│       │   │   └── chat_message.py                     # ChatMessage table (one question and its answer)
│       │   ├── schemas/
│       │   │   ├── chat.py                             # ChatRequest, ChatResponse, ChatHistoryResponse, ChatSessionSummary
│       │   │   └── health.py                           # HealthResponse
│       │   └── services/
│       │       ├── chat_service.py                     # asks the LLM and reads/writes conversations in PostgreSQL
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

## What session 4 assignments does

This branch builds on the session 4 chatbot API and adds **memory**. Every question and the answer the LLM gave are saved in a **PostgreSQL** database under a `session_id`, so a conversation can be listed, read back later, and deleted. The code is split into layers: routes only handle HTTP, the service holds the logic, and the database and LLM code live in their own modules.

**`main.py`** builds the application:

1. Sets up logging first, so everything that happens afterwards is captured.
2. Builds the LLM client at startup, so a missing key or model stops the server immediately with a clear error.
3. Creates the database tables on startup if they do not exist yet.
4. Adds the request-logging middleware, registers the error handler, and mounts the health route at `/health` and the chat routes under `/api/v1`.

**`db/`** connects the app to PostgreSQL:

- `session.py` creates the SQLAlchemy engine from `DATABASE_URL` (with `pool_pre_ping`, so dead connections are dropped after a database restart) and gives every request its own database session.
- `base.py` holds the `Base` class that all models inherit from.
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

- `ask`: sends the question to the LLM, rejects an empty answer, saves the exchange, and returns it.
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
- `routes/health.py` runs `SELECT 1` against the database. It returns `503` if the database cannot be reached.
- `deps.py` builds a `ChatService` for each request from the database session and the shared LLM client.
- `errors.py` turns an `LLMError` into a `502` with a safe message. The real cause is already in `error.log`.

**`schemas/`** validates data in and out with Pydantic:

- **Input (`ChatRequest`):** `session_id` must be 1 to 64 characters (letters, digits, `_` or `-`), and `message` must be 1 to 1000 characters. Surrounding whitespace is stripped first. The same `session_id` rule is applied to the URL path of the history and delete endpoints. Invalid input is rejected with a `422`.
- **Output (`ChatResponse`):** `id`, `session_id`, `question`, `answer` (1 to 10000 characters), and `created_at`.
- `ChatHistoryResponse` wraps a session's messages with its `session_id` and a `count`. `ChatSessionSummary` is one row of the conversation list. `HealthResponse` has `status` and `database`.

**`core/` and `middleware/`** work as in session 4: settings come from `.env`, and three rotating log files (`app.log`, `error.log`, `access.log`) use the format `timestamp | level | request_id | logger:line | message`. Every request gets an ID, returned in the `X-Request-ID` header. SQL statements are only logged when `LOG_LEVEL=DEBUG`.

### Endpoints

| Method | Path | What it does | Success |
|---|---|---|---|
| `POST` | `/api/v1/chat` | Ask the LLM and save the question and answer | `201` |
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

## Installation (using uv)

### 1. Requirements

- [uv](https://docs.astral.sh/uv/) (manages virtual environments and Python versions)
- A [Google Gemini](https://aistudio.google.com/) API key (if you use `LLM_PROVIDER=gemini`)
- [Ollama](https://ollama.com/) installed and running locally (if you use `LLM_PROVIDER=ollama`)
- A running [PostgreSQL](https://www.postgresql.org/) server (only for `session4-assignments`)

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

### 3. Install dependencies

```bash
uv sync
```

This sets up the virtual environment (`.venv`) and installs the required packages defined in `pyproject.toml` and `uv.lock`.

### 4. Set up your `.env` file

From the repository root, copy the environment template:

```bash
cp .env.example .env
```

Fill in the values for the LLM you want to use. Only the values for the chosen provider are required, and the app stops with a clear error at startup if one is missing:

```ini
LLM_PROVIDER=gemini
api-key=your-gemini-key
model=your-gemini-model
OLLAMA_HOST=http://localhost:11434
LOCAL_MODEL=your-ollama-model
```

For `session4-assignments`, also set the PostgreSQL connection:

```ini
POSTGRES_USER=chatbot
POSTGRES_PASSWORD=your-password
POSTGRES_DB=chatbot
POSTGRES_PORT=5432
DATABASE_URL=postgresql+psycopg://chatbot:your-password@localhost:5432/chatbot
```

Optional settings (defaults shown):

```ini
APP_NAME=STS Chatbot
LOG_LEVEL=INFO
LLM_TIMEOUT=60
```

Set `LLM_PROVIDER=ollama` to use the local model instead of Gemini. Never commit `.env` files to git.

### 5. Start Ollama and pull the model (local provider only)

Make sure the Ollama daemon is running, then pull the LLM specified in your `LOCAL_MODEL`:

```bash
ollama pull <your-ollama-model>
```

### 6. Create the PostgreSQL database (`session4-assignments` only)

Create the user and database that match your `DATABASE_URL`, for example:

```sql
CREATE USER chatbot WITH PASSWORD 'your-password';
CREATE DATABASE chatbot OWNER chatbot;
```

You do not need to create the tables. The app creates `chat_messages` automatically when it starts.

## Running the project

Run all commands from inside the project directory:

```bash
cd src/session4-assignments
```

(on the `session4` branch: `cd src/session4`)

### 1. Start the API server

```bash
uv run python scripts/run_dev.py
```

- The server runs at `http://127.0.0.1:8000` and reloads automatically when you edit files in `app/`.
- Startup logs show the chosen LLM provider and model, and that the database is ready.

### 2. Ask a question

Open the interactive docs at `http://127.0.0.1:8000/docs`, expand `POST /api/v1/chat`, click **Try it out**, and send:

```json
{"session_id": "user-1", "message": "What is FastAPI?"}
```

Or from a second terminal:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
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

### 3. Read and delete conversations

List all conversations:

```bash
curl "http://127.0.0.1:8000/api/v1/chat/sessions?limit=10&offset=0"
```

Read one conversation, oldest message first:

```bash
curl http://127.0.0.1:8000/api/v1/chat/user-1/history
```

Delete a conversation (returns `204`, or `404` if it does not exist):

```bash
curl -X DELETE http://127.0.0.1:8000/api/v1/chat/user-1
```

### 4. Try the validation

Send a request that breaks the rules, for example an empty message or a `session_id` with spaces:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "bad id!", "message": ""}'
```

The API returns a `422` that lists each failing field, and the request still appears in `access.log`.

### 5. Check that the server is alive

```bash
curl http://127.0.0.1:8000/health
```

On `session4-assignments` this also checks the database and returns `{"status": "ok", "database": "up"}`.

### 6. Read the logs

```bash
tail -f logs/app.log
tail -f logs/error.log
tail -f logs/access.log
```

The `logs/` folder is created automatically the first time the server starts. To follow a single request across every file, take its ID from the `X-Request-ID` response header (or from any log line) and search for it:

```bash
grep -h "<request-id>" logs/*.log
```
