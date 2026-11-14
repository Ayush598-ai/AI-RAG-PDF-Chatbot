import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH, override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ==========================================
# GLOBAL MODEL CACHE
# ==========================================

_llm = None


# ==========================================
# LOAD OPENAI MODEL
# ==========================================

def load_llm():

    global _llm

    if _llm is None:

        if not OPENAI_API_KEY:
            raise ValueError(
                "OpenAI API key not found. "
                "Check your .env file."
            )

        print("Loading OpenAI Model...", flush=True)

        _llm = ChatOpenAI(
            model="gpt-4.1-mini",
            api_key=OPENAI_API_KEY,
            temperature=0.3,
            max_retries=1,
            timeout=60
        )

        print(
            "OpenAI Model Loaded Successfully!",
            flush=True
        )

    return _llm