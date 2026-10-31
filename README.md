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

## Project structure (`session3` branch)

```
STS-Academy/
├── src/
│   ├── logging_config.py                               # shared logging setup (writes app.log)
│   └── session3/
│       ├── App-Log-Pic/                                # screenshots of the app and its log
│       ├── chroma_db/                                  # persistent vector store (created at runtime)
│       ├── JSON-chat-history/
│       │   └── chat_history.json                       # saved chats (created at runtime)
│       ├── JSON-evaluation-results/
│       │   └── eval_results.json                       # per-question evaluation output
│       ├── JSON-Test-cases/
│       │   └── ground_truth.json                       # evaluation questions and expected answers
│       ├── pdfs/                                       # every PDF to index (manual or uploaded from the UI)
│       ├── scripts/
│       │   ├── Chat-App/
│       │   │   ├── Launch.py                           # starts the Streamlit app
│       │   │   ├── app.py                              # Streamlit chat interface
│       │   │   └── settings.json                       # saved UI settings (created at runtime)
│       │   └── RAG-Engine/
│       │       ├── Speculative_RAG.py                  # indexing, retrieval, clustering, drafting & verification
│       │       └── evaluate.py                         # evaluation: Speculative RAG vs baseline RAG
│       └── app.log                                     # application log (created at runtime)
├── .env.example                                        # template for environment variables
├── .gitignore
├── .python-version
├── pyproject.toml                                      # project dependencies (used by uv)
├── uv.lock                                             # exact dependency versions (used by uv)
└── README.md
```

## What session 3 does

This branch replaces plain retrieve-and-answer RAG with **Speculative RAG**: a small local model writes several draft answers in parallel, each from a different subset of the retrieved chunks, and a stronger model scores the drafts and picks the best one. It adds a Streamlit chat app, central logging, and an evaluation suite that finds which stage lost each wrong answer.

**`Speculative_RAG.py`** handles the core pipeline:

1. **PDF Ingestion & Chunking:** Reads PDFs with PyMuPDF, drops page numbers and table-of-contents lines, then splits the text into whole sentences (up to 1000 characters) with a 250-character overlap.
2. **Embeddings & Vector Storage:** Batches chunks (20 at a time) to OpenRouter's embedding endpoint and stores vectors, text, and metadata (`source`, `file_hash`, `embed_model`) in a persistent ChromaDB collection using cosine distance. Unchanged files are skipped by SHA-256 hash, a changed file with the same name replaces its old chunks, and the index rebuilds automatically if the embedding model changes.
3. **Retrieval & Subsets:** Retrieves the top-k chunks (default 10), clusters their embeddings with KMeans, and builds one subset per draft. Every subset shares the two highest-ranked chunks and adds one chunk from each cluster, so the drafts see different evidence.
4. **Drafting (local Ollama model):** Each subset goes to the local model in parallel. It returns JSON with a `rationale` (an exact supporting quote) followed by a complete-sentence `answer`.
5. **Verification (Gemini):** Gemini scores every draft from 0 to 10 using only the source passages. The highest score wins, and ties are broken by majority vote, then by the longer answer. Calls are spaced 13 seconds apart to stay inside the free-tier limit, with retries and a clear stop on quota or model-name errors.

**`app.py`** provides a Streamlit chat interface:

- Upload PDFs and add them to the knowledge base, with a progress bar per file.
- List indexed documents with their chunk counts and remove any of them.
- Chat with saved history: new chat, recent chats, full history, and delete.
- Settings dialog for chunks per question (3 to 20), number of drafts (2 to 5), and whether to show sources under answers.

**`evaluate.py`** runs quantitative benchmarking against a plain RAG baseline (all chunks in one prompt, one model call):

- **Retrieval:** Hit rate@10, MRR, and subset coverage (did the answer survive into some subset).
- **Drafting:** Oracle accuracy (at least one draft is correct).
- **Verification:** Final accuracy when a correct draft existed.
- **End-to-end:** Speculative RAG vs baseline accuracy and average latency.
- **Failure diagnosis:** Every wrong answer is assigned to the first stage that lost it: `retrieval`, `subsets`, `drafter`, or `verifier`.
- Results are saved per question to `JSON-evaluation-results/eval_results.json`.

**`logging_config.py`** is called once at the top of each entry file and writes everything (DEBUG and above) to `src/session3/app.log` with a timestamp, level, and logger name (`app`, `rag`, `evaluate`, `launch`). Noisy libraries such as `httpx` and `chromadb` are limited to warnings.

## Installation (using uv)

### 1. Requirements

- [uv](https://docs.astral.sh/uv/) (manages virtual environments and Python versions)
- [Ollama](https://ollama.com/) installed and running locally
- An [OpenRouter](https://openrouter.ai/) API key (for embeddings)
- A [Google Gemini](https://aistudio.google.com/) API key (for verification)

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
git checkout session3
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

Ensure your `.env` file contains your credentials and model setup. All six values are required, and the app stops with a clear error if one is missing:

```ini
OPENROUTER_API_KEY=your-openrouter-key
OPENROUTER_MODEL=your-embedding-model
OLLAMA_HOST=http://localhost:11434
LOCAL_MODEL=your-ollama-model
api-key=your-gemini-key
model=your-gemini-model
```

Never commit `.env` files to git.

### 5. Start Ollama and pull the model

Make sure the Ollama daemon is running, then pull the LLM specified in your `LOCAL_MODEL`:

```bash
ollama pull <your-ollama-model>
```

### 6. Add your documents

Put PDFs in `src/session3/pdfs/`, or upload them later from the app's sidebar. Everything in that folder is indexed on startup.

## Running the project

Run all commands from inside the `src/session3` directory:

```bash
cd src/session3
```

### 1. Launch the Streamlit application

```bash
uv run python scripts/Chat-App/Launch.py
```

- Open the local URL displayed in your terminal (typically `http://localhost:8501`).
- Use `--port 8502` to change the port, or `--headless` to skip opening a browser tab.
- Upload PDFs in the sidebar and click **Add to knowledge base**.
- Ask questions in the chat box. Answers show the source documents underneath.
- Open **Settings** to change the number of chunks and drafts.

### 2. Ask questions from the terminal

```bash
uv run python scripts/RAG-Engine/Speculative_RAG.py
```

Prints every draft with its Gemini score and rationale, followed by the final answer. Type `exit` to quit.

### 3. Run the evaluation benchmark

Add your questions to `JSON-Test-cases/ground_truth.json` as a list of objects. `ground_truth` can be a string or a list of acceptable strings:

```json
[
  {"question": "Which model was used for X?", "ground_truth": ["Model A", "Model-A"]}
]
```

Then run:

```bash
uv run python scripts/RAG-Engine/evaluate.py
```

This indexes the PDFs, runs both Speculative RAG and the baseline on every question, prints the metrics (retrieval hit rate and MRR, oracle draft accuracy, end-to-end accuracy, latency, and where each wrong answer was lost), and saves the per-question details to `JSON-evaluation-results/eval_results.json`. Gemini calls are spaced 13 seconds apart, so a long test set takes a while.

### 4. Read the logs

```bash
tail -f app.log
```

`app.log` is created automatically in `src/session3` the first time any entry file runs.
