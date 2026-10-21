import os
from dotenv import load_dotenv
from ollama import Client

# Load environment variables
load_dotenv()

host = os.getenv("OLLAMA_HOST")
model_name = os.getenv("LOCAL_MODEL")

# Initialize Ollama client pointing to your local service
client = Client(host=host)

long_text = """
Artificial intelligence has witnessed rapid evolution over the past decade, 
transitioning from narrow domain-specific models to expansive large language 
models capable of general-purpose comprehension and reasoning. Modern organizations 
rely heavily on these systems to automate workflows, synthesize unstructured data, 
and power conversational agents across industries. However, widespread deployment 
introduces significant technical and infrastructural considerations. Running inference 
on state-of-the-art models demands specialized hardware, robust memory bandwidth, 
and low-latency serving stacks. Consequently, development teams often face trade-offs 
between cloud-hosted APIs that simplify integration and locally hosted instances 
that offer complete data control, cost predictability at scale, and offline utility.
"""

prompt = f"""Summarize the following text in exactly 3 to 4 concise lines:

---
{long_text.strip()}
---
"""

# Query the local Ollama instance
response = client.chat(
    model=model_name,
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
)

print(f"Local Model ({model_name}) summary:\n")
print(response["message"]["content"])