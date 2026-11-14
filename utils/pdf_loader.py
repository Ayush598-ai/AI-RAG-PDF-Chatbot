import os

from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_path):
    """
    Load PDF from the given path.
    """

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found:\n{pdf_path}"
        )

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    return documents