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

## Project structure (`session2-assignments` branch)

```
STS-Academy/
├── src/
│   ├── session1/
│   │   ├── assignments/
│   │   ├── .env.example
│   │   └── First-Day.py
│   └── session2/
│       ├── chunks/
│       ├── embeddings/
│       ├── pdf/
│       ├── session2-assignments/
│       │   ├── sample_pdfs/
│       │   │   ├── devops_ci_cd_architecture.pdf       # sample test document
│       │   │   └── pediatric_bone_age_clinical.pdf     # sample test document
│       │   ├── chroma_db/                              # persistent vector store (created at runtime)
│       │   ├── .env.example                            # template for environment variables
│       │   ├── app.py                                  # Streamlit web interface with document routing
│       │   ├── rag_engine.py                           # PDF extraction, chunking, embeddings & RAG pipeline
│       │   ├── evaluate.py                             # benchmark suite for retrieval recall & generation accuracy
│       │   └── naive_rag_eval_results.json             # saved evaluation benchmark results
│       ├── .env.example
│       ├── embed.py
│       └── searching.py
├── .gitignore
├── .python-version
├── pyproject.toml                                      # project dependencies (used by uv)
├── uv.lock                                             # exact dependency versions (used by uv)
└── README.md
```

## What session 2 assignments does

This branch upgrades the single-PDF pipeline into a multi-document RAG system featuring metadata filtering, relevance distance pruning, an interactive Streamlit UI, and an evaluation suite.

**`rag_engine.py`** handles the core RAG logic:

1. **PDF Ingestion & Cleaning:** Reads uploaded or local PDF files using PyMuPDF (`fitz`), strips redundant headers, digits, and acknowledgement sections, and normalizes characters.
2. **Chunking:** Segments cleaned text into whole sentences (up to 800 characters) with a 150-character sliding overlap.
3. **Embeddings & Vector Storage:** Batches chunks to OpenRouter's embedding endpoint and writes vectors, chunk text, and source document metadata (`source`, `chunk_index`) to a persistent ChromaDB collection.
4. **Distance Filtering & Routing:** Queries ChromaDB using cosine distance, filters chunks by a relevance threshold (`DISTANCE_THRESHOLD = 0.85`), isolates retrieval by source document if selected, and passes the grounded context to a local Ollama LLM.

**`app.py`** provides a Streamlit user interface:

- Upload and index multiple PDFs dynamically.
- Monitor total indexed chunks in ChromaDB.
- Route queries either across all indexed documents or filtered to a specific document.
- Conversational chat interface displaying answers along with source attribution.

**`evaluate.py`** runs quantitative benchmarking:

- Tests retrieval recall (`Recall@4`, `Recall@10`, `MRR`) and answer correctness across answerable and unanswerable questions.
- Enforces strict required-group phrase matching and refusal checks (`I could not find this in the PDF.`).
- Dumps benchmark stats and latencies directly into `naive_rag_eval_results.json`.

## Installation (using uv)

### 1. Requirements

- [uv](https://docs.astral.sh/uv/) (manages virtual environments and Python versions)
- [Ollama](https://ollama.com/) installed and running locally
- An [OpenRouter](https://openrouter.ai/) API key

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
git checkout session2-assignments
```

### 3. Install dependencies

```bash
uv sync
```

This sets up the virtual environment (`.venv`) and installs the required packages defined in `pyproject.toml` and `uv.lock`.

### 4. Set up your `.env` file

Navigate to the assignment directory and copy the environment template:

```bash
cd src/session2/session2-assignments
cp .env.example .env
```

Ensure your `.env` file contains your credentials and local model setup:

```ini
OPENROUTER_API_KEY=your-openrouter-key
OPENROUTER_MODEL=your-embedding-model
OLLAMA_HOST=http://localhost:11434
LOCAL_MODEL=your-ollama-model
```

Never commit `.env` files to git.

### 5. Start Ollama and pull the model

Make sure the Ollama daemon is running, then pull the LLM specified in your `LOCAL_MODEL`:

```bash
ollama pull <your-ollama-model>
```

## Running the project

Run all commands from inside the `src/session2/session2-assignments` directory:

```bash
cd src/session2/session2-assignments
```

### 1. Launch the Streamlit application

```bash
uv run streamlit run app.py
```

- Open the local URL displayed in your terminal (typically `http://localhost:8501`).
- Use the sidebar to upload PDFs and click **Index Uploaded Files**.
- Filter queries by selecting a specific document or keep it as `All`.
- Ask questions in the chat box to retrieve grounded answers.

### 2. Run the evaluation benchmark

```bash
uv run evaluate.py
```

This loads the sample PDFs (`sample_pdfs/`), verifies ChromaDB indexing, tests the retrieval engine across the test suite, outputs performance metrics (Recall, MRR, Accuracy, Latency), and saves the results to `naive_rag_eval_results.json`.
