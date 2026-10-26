import os
import re
import json
import unicodedata
import requests
import numpy as np
import chromadb
import fitz                                    # fitz = PyMuPDF
from dotenv import load_dotenv

load_dotenv()                                  # reads the .env file
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL")

PDF_FILE = "pdf/s12599-025-00945-3.pdf"
CHUNK_SIZE = 1000                              # max characters per chunk
OVERLAP = 200                                  # about this many characters are repeated


# Read the PDF
doc = fitz.open(PDF_FILE)
text = ""
for page in doc:
    for line in page.get_text().split("\n"):
        line = line.strip()
        # skip page numbers and the header repeated on every page
        if line.isdigit() or line.startswith("M. Klesel, H. F. Wittmann:"):
            continue
        text += line + "\n"
print("Characters read:", len(text))


# Clean the text
if "Acknowledgements" in text:
    text = text.split("Acknowledgements")[0]   # remove license text and references

text = unicodedata.normalize("NFKC", text)     # turns the joined letters "ﬁ" into "fi"
text = text.replace("\u0002", "")              # remove the unknown symbol
text = text.replace("-\n", "")                 # join words split with a hyphen at line end
text = text.replace("\n", " ")                 # line breaks -> spaces
text = " ".join(text.split())                  # remove repeated spaces
print("Characters after cleaning:", len(text))


# Split into overlapping chunks (whole sentences)
# split after . ! ? when the next sentence starts with a capital letter
sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

chunks = []
current = []                                   # sentences of the chunk being built
length = 0
for sentence in sentences:
    if length + len(sentence) > CHUNK_SIZE and current:
        chunks.append(" ".join(current))       # chunk is full -> save it
        # keep only the last sentences (about OVERLAP characters) = overlap
        while current and length > OVERLAP:
            length -= len(current[0]) + 1
            current.pop(0)
    current.append(sentence)
    length += len(sentence) + 1
if current:
    chunks.append(" ".join(current))           # save the last chunk
print("Number of chunks:", len(chunks))


# Turn chunks into vectors
vectors = []
for i in range(0, len(chunks), 20):            # send 20 chunks at a time
    batch = chunks[i:i + 20]
    response = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={"Authorization": "Bearer " + API_KEY},
        json={"model": MODEL, "input": batch},
    )
    data = response.json()["data"]
    for item in data:
        vectors.append(item["embedding"])
    print("Embedded", len(vectors), "of", len(chunks))


# Save everything
# vectors -> .npy
np.save("embeddings/embeddings.npy", np.array(vectors))

# Chunks -> .json
with open("chunks/chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

# ChromaDB (stores numbers + text together, so you can search later)
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("pdf_chunks")
collection.add(
    ids=[str(i) for i in range(len(chunks))],
    embeddings=vectors,
    documents=chunks,
)