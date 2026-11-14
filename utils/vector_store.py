import os

from langchain_community.vectorstores import FAISS
from config import VECTOR_DB_PATH


# ==========================================
# Cached Vector Database
# ==========================================

_vector_store = None


# ==========================================
# Create Vector Database
# ==========================================

def create_vector_store(chunks, embedding_model):
    """
    Create and save the FAISS database.
    Keep the newly created database in memory.
    """

    global _vector_store

    os.makedirs(VECTOR_DB_PATH, exist_ok=True)

    print("Creating FAISS vector database...")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    vector_store.save_local(VECTOR_DB_PATH)

    # Cache the newly created database
    _vector_store = vector_store

    print("FAISS database created and cached successfully!")

    return _vector_store


# ==========================================
# Load Existing Vector Database
# ==========================================

def load_vector_store(embedding_model):
    """
    Load the FAISS database only when it is not cached.
    """

    global _vector_store

    if _vector_store is not None:
        print("Using cached FAISS database.")
        return _vector_store

    index_file = os.path.join(
        VECTOR_DB_PATH,
        "index.faiss"
    )

    if not os.path.exists(index_file):
        raise FileNotFoundError(
            "FAISS index not found. Please upload and process a PDF first."
        )

    print("Loading FAISS database from disk...")

    _vector_store = FAISS.load_local(
        VECTOR_DB_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    print("FAISS database loaded successfully!")

    return _vector_store


# ==========================================
# Reset Vector Database Cache
# ==========================================

def reset_vector_store():
    """
    Clear the in-memory FAISS cache.
    Does not delete the database saved on disk.
    """

    global _vector_store

    _vector_store = None

    print("FAISS memory cache cleared.")