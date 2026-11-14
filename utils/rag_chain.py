import time

from langchain_core.prompts import ChatPromptTemplate

from utils.embeddings import load_embeddings
from utils.vector_store import load_vector_store
from utils.openai import load_llm


# ==========================================
# GLOBAL RAG CACHE
# ==========================================

_rag_chain = None


# ==========================================
# INITIALIZE RAG CHAIN
# ==========================================

def load_rag_chain():

    global _rag_chain

    if _rag_chain is None:

        print("Initializing OpenAI RAG Chain...", flush=True)

        llm = load_llm()

        prompt = ChatPromptTemplate.from_template(
            """
You are an AI assistant that answers questions
from an uploaded PDF document.

IMPORTANT RULES:

1. Answer ONLY using the provided context.
2. Do not use outside knowledge.
3. Give clear and simple answers.
4. Avoid unnecessary explanations.
5. If the answer is not available in the context,
   reply exactly: I don't know.

Context:
{context}

Question:
{question}

Answer:
"""
        )

        _rag_chain = prompt | llm

        print(
            "OpenAI RAG Chain initialized successfully!",
            flush=True
        )

    return _rag_chain


# ==========================================
# FALLBACK RESPONSE
# ==========================================

def generate_fallback_answer(documents):

    retrieved_text = "\n\n".join(
        doc.page_content for doc in documents
    )

    return (
        "DEMO MODE - OpenAI API is currently unavailable.\n\n"
        "However, your RAG system successfully retrieved "
        "the following relevant information from your PDF:\n\n"
        + retrieved_text
        + "\n\nNote: This is retrieved PDF content, "
        "not an AI-generated answer."
    )


# ==========================================
# ASK QUESTION
# ==========================================

def ask_question(question):

    total_start = time.perf_counter()

    timings = {}

    print(
        "\n========== PERFORMANCE REPORT ==========",
        flush=True
    )

    # ======================================
    # 1. LOAD EMBEDDING MODEL
    # ======================================

    start = time.perf_counter()

    embedding_model = load_embeddings()

    timings["Embedding Model"] = round(
        time.perf_counter() - start, 2
    )

    # ======================================
    # 2. LOAD FAISS DATABASE
    # ======================================

    start = time.perf_counter()

    vector_store = load_vector_store(
        embedding_model
    )

    timings["FAISS Access"] = round(
        time.perf_counter() - start, 2
    )

    # ======================================
    # 3. RETRIEVE RELEVANT DOCUMENTS
    # ======================================

    start = time.perf_counter()

    results = vector_store.similarity_search_with_score(
        question,
        k=3
    )

    timings["Retrieval"] = round(
        time.perf_counter() - start, 2
    )

    documents = []
    scores = []

    for doc, score in results:

        documents.append(doc)
        scores.append(float(score))

    if not documents:

        return {
            "answer": "I don't know.",
            "pages": [],
            "confidence": 0,
            "timings": timings
        }

    # ======================================
    # 4. PREPARE CONTEXT
    # ======================================

    start = time.perf_counter()

    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    timings["Context Preparation"] = round(
        time.perf_counter() - start, 2
    )

    # ======================================
    # 5. GENERATE ANSWER USING OPENAI
    # ======================================

    answer = None

    try:

        print(
            "Sending question to OpenAI...",
            flush=True
        )

        start = time.perf_counter()

        rag_chain = load_rag_chain()

        response = rag_chain.invoke(
            {
                "context": context,
                "question": question.strip()
            }
        )

        timings["OpenAI Response"] = round(
            time.perf_counter() - start, 2
        )

        content = response.content

        if isinstance(content, str):

            answer = content

        elif isinstance(content, list):

            answer_parts = []

            for item in content:

                if isinstance(item, dict):

                    if item.get("type") == "text":

                        answer_parts.append(
                            item.get("text", "")
                        )

                elif hasattr(item, "text"):

                    answer_parts.append(item.text)

            answer = "\n".join(answer_parts)

        else:

            answer = str(content)

        if not answer.strip():

            answer = "I don't know."

        print(
            "OpenAI answered successfully!",
            flush=True
        )

    except Exception as error:

        print(
            f"OpenAI Error: {error}",
            flush=True
        )

        print(
            "Activating RAG Demo Fallback Mode...",
            flush=True
        )

        start = time.perf_counter()

        answer = generate_fallback_answer(
            documents
        )

        timings["Fallback Response"] = round(
            time.perf_counter() - start, 2
        )

    # ======================================
    # 6. EXTRACT SOURCE PAGES
    # ======================================

    pages = sorted(
        set(
            doc.metadata["page"] + 1
            for doc in documents
            if "page" in doc.metadata
        )
    )

    # ======================================
    # 7. CALCULATE SIMILARITY SCORE
    # ======================================

    if scores:

        avg_score = sum(scores) / len(scores)

        if avg_score <= 0.20:
            confidence = 98

        elif avg_score <= 0.40:
            confidence = 95

        elif avg_score <= 0.60:
            confidence = 90

        elif avg_score <= 0.80:
            confidence = 85

        elif avg_score <= 1.00:
            confidence = 80

        elif avg_score <= 1.50:
            confidence = 70

        else:
            confidence = 60

    else:

        confidence = 0

    # ======================================
    # 8. TOTAL PROCESSING TIME
    # ======================================

    timings["Total Time"] = round(
        time.perf_counter() - total_start, 2
    )

    # ======================================
    # 9. PRINT PERFORMANCE REPORT
    # ======================================

    print(
        "\n========== PERFORMANCE REPORT ==========",
        flush=True
    )

    for name, seconds in timings.items():

        print(
            f"{name}: {seconds} seconds",
            flush=True
        )

    print(
        "========================================\n",
        flush=True
    )

    # ======================================
    # 10. RETURN RESULT
    # ======================================

    return {
        "answer": answer,
        "pages": pages,
        "confidence": confidence,
        "timings": timings
    }