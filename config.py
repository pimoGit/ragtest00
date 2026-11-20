import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

_ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_FILE)


def get_genai_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY non impostata. Crea un file .env copiando "
            ".env.example e inserisci la chiave."
        )

    return genai.Client(api_key=api_key)
