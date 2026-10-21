# Session 1: Environment Setup & Gemini API Quickstart

This repository contains the materials, setups, and baseline scripts developed during **Session 1**. The goal of this session was to establish a modern Python development environment, explore LLM deployment options, and verify connectivity to the Gemini API via Google AI Studio.

---

## Session Overview

During this session, we covered:
1. **Version Control Setup**: Initializing the GitHub repository, cloning locally, and configuring standard Git hygiene (`.gitignore`, `.env.example`).
2. **Modern Package Management with `uv`**:
   - Understanding what `uv` is (an extremely fast Python package and project manager written in Rust).
   - Managing Python versions (`.python-version`), dependencies (`pyproject.toml`), and lockfiles (`uv.lock`) seamlessly without manual `venv` activation overhead.
3. **LLM Hosting Landscape**:
   - Evaluating local LLM hosting tools (e.g., Ollama, LM Studio, vLLM, LocalAI) vs. cloud API providers.
   - Trade-offs in hardware demands, latency, privacy, and development speed.
4. **Google AI Studio Integration**:
   - Generating an API key from Google AI Studio.
   - Safely storing credentials using environment variables.
5. **Connectivity Smoke Test**:
   - Writing a minimal Python script (`First-Day.py`) to query the Gemini model and verify API responses.

---

## Repository Structure

```text
├── .venv/               # Virtual environment managed by uv (ignored by git)
├── src/                 # Source code directory
├── .env.example         # Template for required environment variables
├── .gitignore           # Git ignore rules for virtualenvs, keys, and cache
├── .python-version      # Pinned Python version
├── First-Day.py         # Baseline script testing Gemini API connectivity
├── pyproject.toml       # Project configuration and dependency specifications
├── README.md            # Session documentation
└── uv.lock              # Deterministic lockfile for project dependencies
