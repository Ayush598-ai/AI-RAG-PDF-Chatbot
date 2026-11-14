import os
from dotenv import load_dotenv

# ==========================================
# PROJECT DIRECTORY
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH, override=True)

# ==========================================
# OPENAI API CONFIGURATION
# ==========================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OpenAI API Key not found. "
        "Please check your .env file."
    )

# ==========================================
# OPENAI MODEL
# ==========================================

OPENAI_MODEL = "gpt-4.1-mini"

# ==========================================
# EMBEDDING MODEL
# ==========================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ==========================================
# PROJECT PATHS
# ==========================================

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

VECTOR_DB_PATH = os.path.join(
    BASE_DIR,
    "vector_db",
    "faiss_index"
)

# ==========================================
# TEXT SPLITTER
# ==========================================

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200

# ==========================================
# RETRIEVAL SETTINGS
# ==========================================

TOP_K = 3