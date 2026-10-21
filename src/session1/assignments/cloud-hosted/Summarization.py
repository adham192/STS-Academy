import os
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env
load_dotenv()

api_key = os.getenv("api-key")
model_name = os.getenv("model")

# Initialize the Gemini client
client = genai.Client(api_key=api_key)

# The long text to summarize
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

# Construct a prompt with explicit length constraints
prompt = f"""Summarize the following text in exactly 3 to 4 concise lines:

---
{long_text.strip()}
---
"""

response = client.models.generate_content(
    model=model_name,
    contents=prompt,
)

print(f"Model ({model_name}) summary:\n")
print(response.text)