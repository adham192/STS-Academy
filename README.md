# Session 1: Assignment Text Summarization (Cloud vs. Local LLMs)

This branch contains the practical assignment solutions for Session 1. The core objective is implementing text summarization with strict output constraints (3–4 lines) across two architectural paradigms:

1. **Cloud-Hosted Model:** Leveraging the Google GenAI SDK to interact with the Gemini API.
2. **Locally-Hosted Model:** Interfacing with a self-hosted model (`qwen2.5-coder:latest`) running locally via Ollama.

## Branches

| Branch                           | What it contains                                                          |
|----------------------------------|---------------------------------------------------------------------------|
| `session1`                       | Environment setup and the Gemini API connectivity smoke test             |
| `session1-assignments` (this one)| Text summarization assignment with a cloud model (Gemini) and a local model (Ollama) |
| `session2`                       | PDF embedding and retrieval with ChromaDB and a local Ollama model        |

## Branch Structure

```
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
├── src/
│   └── session1/
│       └── assignments/
│           ├── cloud-hosted/
│           │   ├── .env.example
│           │   └── Summarization.py
│           └── locally-hosted/
│               ├── .env.example
│               └── Summarization.py
└── uv.lock
```

## What this branch does

Both scripts are named `Summarization.py` and do the same job: take a text and summarize it in 3–4 lines. They differ in where the model runs.

* **`cloud-hosted/`** sends the text to the Gemini API through the Google GenAI SDK. It needs an API key and an internet connection.
* **`locally-hosted/`** sends the text to `qwen2.5-coder:latest`, running on your own computer through Ollama. It needs no API key and no internet connection once the model is downloaded.

## Installation (using uv)

### 1. Requirements

* [uv](https://docs.astral.sh/uv/) (it also installs the pinned Python version for you)
* For the cloud-hosted script: a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
* For the locally-hosted script: [Ollama](https://ollama.com/) installed and running

Install uv if you don't have it:

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows, use `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`.

### 2. Get the project

```bash
git clone <repo-url>
cd STS-Academy
git checkout session1-assignments
```

### 3. Install the dependencies

```bash
uv sync
```

This reads `.python-version`, `pyproject.toml` and `uv.lock`, then creates the `.venv` folder and installs the exact dependency versions. You never need to activate the virtual environment manually.

### 4. Set up your `.env` files

Each script has its own `.env.example`. Copy it to `.env` in the same folder and fill in your values (use the variable names written in the template):

```bash
cp src/session1/assignments/cloud-hosted/.env.example src/session1/assignments/cloud-hosted/.env
cp src/session1/assignments/locally-hosted/.env.example src/session1/assignments/locally-hosted/.env
```

Never commit your `.env` files. They are listed in `.gitignore`.

### 5. Download the local model (locally-hosted script only)

Make sure Ollama is running, then pull the model:

```bash
ollama pull qwen2.5-coder:latest
```

## Running the project

Cloud-hosted (Gemini):

```bash
uv run src/session1/assignments/cloud-hosted/Summarization.py
```

Locally-hosted (Ollama):

```bash
uv run src/session1/assignments/locally-hosted/Summarization.py
```

Each script prints a summary of 3–4 lines.
