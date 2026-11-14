"""FastAPI backend: Speculative RAG pipeline + PostgreSQL + Redis behind HTTP.

Run:  uvicorn api:app --app-dir scripts/backend --host 0.0.0.0 --port 8000
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
RAG_DIR = HERE.parent / "RAG-Engine"          # Speculative_RAG.py lives here (hyphenated, so not a package)
for path in (HERE, RAG_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cache  # noqa: E402
import database  # noqa: E402
from qa_service import QAService  # noqa: E402

logger = logging.getLogger("backend.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Importing the engine validates the provider settings in .env (it raises if any are
    # missing, exactly as before) and sets up the Ollama / Gemini clients.
    import Speculative_RAG as rag

    database.init_db()                  # logs and carries on if PostgreSQL is not up yet
    try:
        rag.build_index()               # fast no-op when nothing changed
    except Exception:
        logger.exception("build_index() failed; the API still starts")
    app.state.rag = rag
    app.state.qa = QAService(rag)
    logger.info("Backend ready")
    yield


app = FastAPI(title="Speculative RAG API", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: Optional[list[ChatMessage]] = None   # earlier turns of the chat (follow-ups)
    document: Optional[str] = None                # restrict search to one indexed PDF
    top_k: int = Field(default=20, ge=1, le=20)
    n_subsets: int = Field(default=5, ge=2, le=5)


class AskResponse(BaseModel):
    answer: str
    source: Literal["redis", "postgresql", "speculative_rag"]   # where the answer came from
    documents: list[str] = []                                   # PDFs the answer is based on


# --------------------------------------------------------------------------- #
# Endpoints (plain `def`: FastAPI runs them in a thread pool, the pipeline is blocking)
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    """Liveness plus dependency status. Redis/PostgreSQL being down does not fail the probe."""
    return {"status": "ok", "postgres": database.ping(), "redis": cache.ping()}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question must not be empty.")
    history = [m.model_dump() for m in request.history] if request.history else None
    try:
        result = app.state.qa.ask(
            request.question, history=history, document=request.document,
            top_k=request.top_k, n_subsets=request.n_subsets,
        )
    except Exception as error:
        logger.exception("The RAG pipeline failed")
        raise HTTPException(status_code=502, detail=f"The RAG pipeline failed: {error}") from error
    return AskResponse(answer=result.answer, source=result.source, documents=result.documents)


@app.get("/documents")
def list_documents():
    """{file name: chunk count} for everything indexed."""
    return app.state.rag.list_documents()


@app.post("/documents")
def upload_document(file: UploadFile = File(...)):
    """Save an uploaded PDF and index it (same steps the Streamlit sidebar did before)."""
    name = Path(file.filename or "").name
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    rag = app.state.rag
    target = rag.DOCS_DIR / name
    target.write_bytes(file.file.read())
    try:
        status, chunks = rag.ingest_pdf(target, name)
    except Exception as error:
        logger.exception("Failed to index %s", name)
        raise HTTPException(status_code=502, detail=f"Failed to index {name}: {error}") from error
    if status == "added":
        app.state.qa.invalidate_document(name)
    return {"name": name, "status": status, "chunks": chunks}


@app.delete("/documents/{name}")
def delete_document(name: str):
    app.state.rag.delete_document(name)
    app.state.qa.invalidate_document(name)
    return {"deleted": name}
