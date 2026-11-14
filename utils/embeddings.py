from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL

# ==========================================
# Global Embedding Model (Loaded Only Once)
# ==========================================

_embedding_model = None


def load_embeddings():
    """
    Loads the embedding model only once.
    """

    global _embedding_model

    if _embedding_model is None:

        print("Loading Embedding Model...")

        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

    return _embedding_model