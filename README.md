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

## Installation (using uv)

### 1. Requirements

- [uv](https://docs.astral.sh/uv/) (manages virtual environments and Python versions)
- A [Google Gemini](https://aistudio.google.com/) API key (if you use `LLM_PROVIDER=gemini`)
- [Ollama](https://ollama.com/) installed and running locally (if you use `LLM_PROVIDER=ollama`)

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
git checkout session4
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

## Running the project

Run all commands from inside the `src/session4` directory:

```bash
cd src/session4
```

### 1. Start the API server

```bash
uv run python scripts/run_dev.py
```

- The server runs at `http://127.0.0.1:8000` and reloads automatically when you edit files in `app/`.
- Startup logs show the chosen LLM provider and model.

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

The response has this shape:

```json
{"session_id": "user-1", "answer": "...", "timestamp": "2026-11-08T11:32:37.470930Z"}
```

### 3. Try the validation

Send a request that breaks the rules, for example an empty message or a `session_id` with spaces:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "bad id!", "message": ""}'
```

The API returns a `422` that lists each failing field, and the request still appears in `access.log`.

| Status | Meaning |
|---|---|
| `200` | Question answered |
| `422` | Invalid input (rejected by Pydantic) |
| `502` | The LLM failed or returned an empty answer |

### 4. Check that the server is alive

```bash
curl http://127.0.0.1:8000/health
```

### 5. Read the logs

```bash
tail -f logs/app.log
tail -f logs/error.log
tail -f logs/access.log
```

The `logs/` folder is created automatically the first time the server starts. To follow a single request across every file, take its ID from the `X-Request-ID` response header (or from any log line) and search for it:

```bash
grep -h "<request-id>" logs/*.log
```
