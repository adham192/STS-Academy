import os
import re
import unicodedata
import requests
import chromadb
import fitz                                    # fitz = PyMuPDF
from dotenv import load_dotenv

load_dotenv()                                  # reads the .env file
API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBED_MODEL = os.getenv("OPENROUTER_MODEL")
OLLAMA_URL = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("LOCAL_MODEL")

CHUNK_SIZE = 800                               # max characters per chunk
OVERLAP = 150                                  # overlap characters between chunks
DISTANCE_THRESHOLD = 0.85                      # max cosine distance allowed for relevance


def get_chroma_collection():
    """Initializes and returns the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_or_create_collection(
        name="multi_pdf_chunks",
        metadata={"hnsw:space": "cosine"}
    )


def extract_and_clean_pdf(pdf_path_or_bytes, filename):
    """Extracts text from PDF bytes or path and applies standard cleaning."""
    if isinstance(pdf_path_or_bytes, bytes):
        doc = fitz.open(stream=pdf_path_or_bytes, filetype="pdf")
    else:
        doc = fitz.open(pdf_path_or_bytes)

    text = ""
    for page in doc:
        for line in page.get_text().split("\n"):
            line = line.strip()
            # skip numbers and headers
            if line.isdigit():
                continue
            text += line + "\n"

    # Clean the text
    if "Acknowledgements" in text:
        text = text.split("Acknowledgements")[0]

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u0002", "")
    text = text.replace("-\n", "")
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


def split_text_to_chunks(text):
    """Splits text into whole sentences with overlap."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    chunks = []
    current = []
    length = 0

    for sentence in sentences:
        if length + len(sentence) > CHUNK_SIZE and current:
            chunks.append(" ".join(current))
            while current and length > OVERLAP:
                length -= len(current[0]) + 1
                current.pop(0)
        current.append(sentence)
        length += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


def get_embeddings(texts):
    """Batch sends texts to OpenRouter embeddings API."""
    vectors = []
    for i in range(0, len(texts), 20):
        batch = texts[i:i + 20]
        response = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": EMBED_MODEL, "input": batch},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()["data"]
        for item in data:
            vectors.append(item["embedding"])
    return vectors


def ingest_document(file_bytes, filename):
    """Parses, chunks, embeds, and indexes a PDF with metadata tags."""
    collection = get_chroma_collection()

    # Avoid duplicate additions for the same filename
    existing = collection.get(where={"source": filename})
    if existing and existing["ids"]:
        collection.delete(where={"source": filename})

    text = extract_and_clean_pdf(file_bytes, filename)
    chunks = split_text_to_chunks(text)
    if not chunks:
        return 0

    vectors = get_embeddings(chunks)

    # Attach document-level metadata to every chunk
    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=chunks,
        metadatas=metadatas
    )
    return len(chunks)


def query_rag(question, selected_source="All"):
    """
    Advanced RAG Retrieval:
    1. Embeds question.
    2. Applies metadata routing if a specific document is chosen.
    3. Filters out chunks beyond the relevance distance threshold.
    4. Prompts local Ollama LLM with strict grounding constraints.
    """
    collection = get_chroma_collection()

    # Get question vector
    response = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": EMBED_MODEL, "input": [question]},
        timeout=60,
    )
    response.raise_for_status()
    question_vector = response.json()["data"][0]["embedding"]

    # Advanced RAG: Apply metadata source filter
    where_filter = None if selected_source == "All" else {"source": selected_source}

    results = collection.query(
        query_embeddings=[question_vector],
        n_results=4,
        where=where_filter
    )

    found_chunks = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results["distances"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    # Score filtering: drop chunks that are not close enough semantically
    filtered_chunks = []
    filtered_sources = []
    for chunk, dist, meta in zip(found_chunks, distances, metadatas):
        if dist <= DISTANCE_THRESHOLD:
            filtered_chunks.append(chunk)
            filtered_sources.append(meta.get("source", "Unknown"))

    if not filtered_chunks:
        return "I could not find this in the PDF.", []

    context = "\n\n".join(filtered_chunks)
    prompt = (
        "You are answering questions about the uploaded documents. "
        "Use ONLY the context below. Do not use any other knowledge. "
        "Write one clear and complete final answer in simple words, "
        "in 3 to 6 sentences. Include the important details from the context. "
        "If the answer is not in the context, say exactly: I could not find this in the PDF.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Final answer:"
    )

    reply = requests.post(
        OLLAMA_URL.rstrip("/") + "/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=300,
    )
    reply.raise_for_status()
    answer = reply.json()["response"].strip()

    return answer, list(set(filtered_sources))