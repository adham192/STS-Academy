# STS-Academy

Course projects for the STS Academy. Each session lives in its own branch, so this page is your map: find what you want to do, then go to the branch that has it.

## How to use this repository

1. Clone it once:

   ```bash
   git clone <repo-url>
   cd STS-Academy
   ```

2. Switch to the branch you need (see the guide below):

   ```bash
   git checkout <branch-name>
   ```

3. Follow the README of that branch. Each branch has its own README with full setup and run instructions.

Tips:

* All projects use [uv](https://docs.astral.sh/uv/) for Python and dependencies. After switching branches, run `uv sync` so your environment matches that branch.
* Every branch has its own `.env.example`. Copy it to `.env` and fill in your own values. Never commit your `.env` file.

Install uv if you don't have it:

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows, use `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`.

## Which branch do I need?

### If you want to set up your environment and test the Gemini API

Go to **`session1`**:

```bash
git checkout session1
```

You will find: how to set up `uv`, the Git basics used in the course (`.gitignore`, `.env.example`), an overview of local vs. cloud LLM hosting, and `First-Day.py`, a small script that checks your Gemini API key works.

### If you want to see text summarization with a cloud model and a local model

Go to **`session1-assignments`**:

```bash
git checkout session1-assignments
```

You will find: the same summarization task (3–4 line summaries) solved twice, once with the Gemini API (cloud-hosted) and once with `qwen2.5-coder:latest` running locally through Ollama (locally-hosted).

### If you want to build a "chat with your PDF" system (RAG)

Go to **`session2`**:

```bash
git checkout session2
```

You will find: a PDF that is cleaned, split into overlapping chunks, turned into embeddings (OpenRouter), and stored in ChromaDB, plus a script that answers questions about the PDF using a local Ollama model. The answers come only from the PDF.

## Branch summary

| Branch                 | Topic                                          | Needs                                  |
|------------------------|------------------------------------------------|----------------------------------------|
| `main`                 | This guide                                     | Nothing                                |
| `session1`             | Environment setup and Gemini API quickstart    | Gemini API key                         |
| `session1-assignments` | Summarization: cloud vs. local LLM             | Gemini API key, Ollama                 |
| `session2`             | PDF embeddings, ChromaDB and question answering| OpenRouter API key, Ollama             |

## Suggested order

If you are new to the course, go through the branches in this order:

1. `session1`: set up your tools and check that everything works.
2. `session1-assignments`: use a cloud model and a local model on a simple task.
3. `session2`: combine what you learned into a small RAG project.

## Seeing all branches

```bash
git branch -a
```

To go back to this page at any time:

```bash
git checkout main
```
