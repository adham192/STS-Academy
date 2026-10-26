# Session 1: Environment Setup & Gemini API Quickstart

This repository contains the materials, setups, and baseline scripts developed during Session 1. The goal of this session was to establish a modern Python development environment, explore LLM deployment options, and verify connectivity to the Gemini API via Google AI Studio.

## Branches

| Branch                 | What it contains                                                          |
|------------------------|---------------------------------------------------------------------------|
| `session1` (this one)  | Environment setup and the Gemini API connectivity smoke test             |
| `session1-assignments` | Text summarization assignment with a cloud model (Gemini) and a local model (Ollama) |
| `session2`             | PDF embedding and retrieval with ChromaDB and a local Ollama model        |

## Session Overview

During this session, we covered:

1. **Version Control Setup:** Initializing the GitHub repository, cloning locally, and configuring standard Git hygiene (`.gitignore`, `.env.example`).
2. **Modern Package Management with `uv`:**
   * Understanding what `uv` is (an extremely fast Python package and project manager written in Rust).
   * Managing Python versions (`.python-version`), dependencies (`pyproject.toml`), and lockfiles (`uv.lock`) seamlessly without manual `venv` activation overhead.
3. **LLM Hosting Landscape:**
   * Evaluating local LLM hosting tools (e.g., Ollama, LM Studio, vLLM, LocalAI) vs. cloud API providers.
   * Trade-offs in hardware demands, latency, privacy, and development speed.
4. **Google AI Studio Integration:**
   * Generating an API key from Google AI Studio.
   * Safely storing credentials using environment variables.
5. **Connectivity Smoke Test:**
   * Writing a minimal Python script (`First-Day.py`) to query the Gemini model and verify API responses.

## Repository Structure

```
├── .venv/               # Virtual environment managed by uv (ignored by git)
├── src/                 # Source code directory
├── .env.example         # Template for required environment variables
├── .gitignore           # Git ignore rules for virtualenvs, keys, and cache
├── .python-version      # Pinned Python version
├── First-Day.py         # Baseline script testing Gemini API connectivity
├── pyproject.toml       # Project configuration and dependency specifications
├── README.md            # Session documentation
└── uv.lock              # Deterministic lockfile for project dependencies
```

## Installation (using uv)

### 1. Requirements

* [uv](https://docs.astral.sh/uv/) (it also installs the pinned Python version for you)
* A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

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
git checkout session1
```

### 3. Install the dependencies

```bash
uv sync
```

This reads `.python-version`, `pyproject.toml` and `uv.lock`, then creates the `.venv` folder and installs the exact dependency versions. You never need to activate the virtual environment manually.

### 4. Set up your `.env` file

Copy the template and fill in your own values (use the variable names written in `.env.example`):

```bash
cp .env.example .env
```

Never commit your `.env` file. It is listed in `.gitignore`.

## Running the project

Run the connectivity smoke test:

```bash
uv run First-Day.py
```

If everything is set up correctly, the script prints a response from the Gemini model.
