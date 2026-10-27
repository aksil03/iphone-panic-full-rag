import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialisation du client OpenAI configuré pour Groq
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)