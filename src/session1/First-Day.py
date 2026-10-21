import os
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env
load_dotenv()

api_key = os.getenv("api-key")
model_name = os.getenv("model")

# Initialize the Gemini client
client = genai.Client(api_key=api_key)

# Ask a test question
prompt = "List LLM parameters and their descriptions usages and examples."

response = client.models.generate_content(
    model=model_name,
    contents=prompt,
)

print(f"Model ({model_name}) response:\n")
print(response.text)