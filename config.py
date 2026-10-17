import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialisation du client
client = OpenAI(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)