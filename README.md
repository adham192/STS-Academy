# Session 1: Assignment Text Summarization (Cloud vs. Local LLMs)

This branch contains the practical assignment solutions for **Session 1**. The core objective is implementing text summarization with strict output constraints (3–4 lines) across two architectural paradigms:
1. **Cloud-Hosted Model:** Leveraging the Google GenAI SDK to interact with the Gemini API.
2. **Locally-Hosted Model:** Interfacing with a self-hosted model (`qwen2.5-coder:latest`) running locally via Ollama.

---

## Branch Structure

```text
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
