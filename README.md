# STS-Academy

Course projects for the STS Academy, one branch per session.

## Branches

| Branch     | What it contains                                                    |
|------------|---------------------------------------------------------------------|
| `main`     | Guide to each branch                                                |
| `session1` | Hosting Model on cloud                                              |
| `session1-assignments` | Hosting Model locally and on cloud                      |
| `session2` | PDF embedding and retrieval with ChromaDB and a local Ollama model  |

## Project structure (`session2` branch)

```
STS-Academy/
├── src/
│   ├── session1/
│   │   ├── assignments/
│   │   ├── .env.example
│   │   └── First-Day.py
│   └── session2/
│       ├── pdf/
│       │   └── s12599-025-00945-3.pdf    # the paper used as the knowledge source
│       ├── chunks/
│       │   └── chunks.json               # text chunks (created by embed.py)
│       ├── embeddings/
│       │   └── embeddings.npy            # vectors for each chunk (created by embed.py)
│       ├── chroma_db/                    # vector database (created by embed.py, not pushed)
│       ├── .env.example                  # template for your .env file
│       ├── embed.py                      # step 1: PDF -> chunks -> vectors -> storage
│       └── searching.py                  # step 2: ask questions about the PDF
├── .gitignore
├── .python-version
├── pyproject.toml                        # project dependencies (used by uv)
├── uv.lock                               # exact dependency versions (used by uv)
└── README.md
```

## What session 2 does

Session 2 builds a small RAG (Retrieval-Augmented Generation) system over one PDF.
The answers come only from the text of the PDF.

**`embed.py`** prepares the data:

1. Reads the PDF with PyMuPDF (`fitz`).
2. Cleans the text: removes page numbers, headers and the reference list, fixes broken
   words and special characters.
3. Splits the text into chunks of whole sentences (up to 1000 characters) with an
   overlap, so ideas at the edge of a chunk are not lost.
4. Turns each chunk into a vector (a list of numbers that represents its meaning) using
   an embedding model through the OpenRouter API.
5. Saves the vectors in `embeddings/embeddings.npy`, the chunk text in
   `chunks/chunks.json`, and both together in a ChromaDB database (`chroma_db/`).

**`searching.py`** answers questions:

1. Turns your question into a vector with the same embedding model.
2. Finds the closest chunks in ChromaDB.
3. Sends those chunks and your question to a local Ollama model, which writes the final
   answer using only that context.

## Installation (using uv)

### 1. Requirements

- [uv](https://docs.astral.sh/uv/) (it installs the right Python version for you)
- [Ollama](https://ollama.com/) installed and running locally
- An [OpenRouter](https://openrouter.ai/) API key

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
git checkout session2
```

### 3. Install the dependencies

```bash
uv sync
```

This creates the `.venv` folder and installs everything listed in `pyproject.toml` and
`uv.lock`.

### 4. Set up your `.env` file

Copy the template to the project root and fill in your values:

```bash
cp src/session2/.env.example .env
```

The file should contain:

```
OPENROUTER_API_KEY=your-openrouter-key
OPENROUTER_MODEL=your-embedding-model
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=your-ollama-model
```

Never commit your `.env` file. It is listed in `.gitignore`.

### 5. Download the Ollama model

Make sure Ollama is running, then pull the model you put in `OLLAMA_MODEL`:

```bash
ollama pull <your-ollama-model>
```

## Running the project

Run both scripts from inside the `session2` folder.

```bash
cd src/session2
```

**Step 1: create the embeddings** (run once; the `chroma_db` folder is not in the repo)

```bash
uv run embed.py
```

If you run it again, delete the `chroma_db` folder first, otherwise ChromaDB will
complain that the ids already exist.

**Step 2: ask questions**

```bash
uv run searching.py
```

Type a question about the PDF, for example:

- What is the blinkered chunk effect?
- What is grounding?
- What does RAPTOR stand for?

Type `exit` to quit. To also see which chunks were retrieved, set `SHOW_CHUNKS = True`
at the top of `searching.py`.
