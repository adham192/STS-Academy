import os
import requests
import chromadb
from dotenv import load_dotenv

load_dotenv()                                  # reads the .env file
API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBED_MODEL = os.getenv("OPENROUTER_MODEL")    # must be the SAME model used in embed.py
OLLAMA_URL = os.getenv("OLLAMA_HOST")           # example: http://localhost:11434
OLLAMA_MODEL = os.getenv("LOCAL_MODEL")       

TOP_K = 4                                      # how many chunks to retrieve
SHOW_CHUNKS = False                            # True = also print the retrieved chunks


# Open the database that embed.py created
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("pdf_chunks")
print("Chunks in database:", collection.count())


# Ask questions in a loop
while True:
    question = input("\nAsk a question (or type exit): ")
    if question.strip().lower() == "exit":
        break

    # Turn the question into a vector (same model as the chunks)
    response = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={"Authorization": "Bearer " + API_KEY},
        json={"model": EMBED_MODEL, "input": [question]},
    )
    question_vector = response.json()["data"][0]["embedding"]

    # Find the chunks with the closest vectors
    results = collection.query(query_embeddings=[question_vector], n_results=TOP_K)
    found_chunks = results["documents"][0]
    distances = results["distances"][0]

    # Show what was retrieved (smaller distance = closer match)
    if SHOW_CHUNKS:
        print("\n--- Retrieved chunks ---")
        for i in range(len(found_chunks)):
            print("\n#" + str(i + 1), "distance:", round(distances[i], 3))
            print(found_chunks[i])

    # Build a prompt that only allows the PDF text as the source
    context = "\n\n".join(found_chunks)
    prompt = (
        "You are answering questions about a research paper. "
        "Use ONLY the context below. Do not use any other knowledge. "
        "Write one clear and complete final answer in simple words, "
        "in 3 to 6 sentences. Include the important details from the context. "
        "If the answer is not in the context, say exactly: I could not find this in the PDF.\n\n"
        "Context:\n" + context + "\n\n"
        "Question: " + question + "\n\n"
        "Final answer:"
    )

    # Send the prompt to the local Ollama model
    reply = requests.post(
        OLLAMA_URL.rstrip("/") + "/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=300,
    )
    print("\n=== Final answer ===")
    print(reply.json()["response"].strip())