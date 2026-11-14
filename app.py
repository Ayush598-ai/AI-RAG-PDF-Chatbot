import hashlib
import os
import tempfile
import time

import streamlit as st

# Project modules
from utils.pdf_loader import load_pdf
from utils.text_splitter import split_documents
from utils.embeddings import load_embeddings
from utils.vector_store import create_vector_store
from utils.rag_chain import ask_question


# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# CUSTOM CSS - PEACH THEME
# ==========================================

st.markdown(
    """
    <style>

    /* =====================================
       COLOR PALETTE
    ===================================== */

    :root {
        --page: #F8EDEB;
        --sidebar: #FAE1DD;
        --card: #FFFFFF;
        --text: #493B38;
        --muted: #806D68;
        --accent: #FEC5BB;
        --accent-hover: #FCD5CE;
        --border: #F2C9C2;
        --input-bg: #FFF8F6;
    }

    /* =====================================
       MAIN APPLICATION
    ===================================== */

    .stApp {
        background: var(--page) !important;
        color: var(--text) !important;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background: var(--page) !important;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }

    /* =====================================
       SIDEBAR
    ===================================== */

    [data-testid="stSidebar"] {
        background: var(--sidebar) !important;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.4rem;
    }

    /* =====================================
       HEADINGS AND TEXT
    ===================================== */

    h1, h2, h3, h4 {
        color: var(--text) !important;
        font-weight: 700;
    }

    p,
    label,
    [data-testid="stMarkdownContainer"] {
        color: var(--text);
    }

    .stCaption,
    [data-testid="stCaptionContainer"] {
        color: var(--muted) !important;
    }

    /* =====================================
       BUTTONS
    ===================================== */

    .stButton > button {
        width: 100%;
        min-height: 2.7rem;
        border: 0;
        border-radius: 12px;
        background: var(--accent);
        color: var(--text) !important;
        font-weight: 600;
        transition: background 0.2s ease;
    }

    .stButton > button:hover {
        background: var(--accent-hover);
        color: var(--text) !important;
        border: 0;
    }

    .stButton > button:focus {
        box-shadow: 0 0 0 2px var(--page),
                    0 0 0 4px var(--accent);
    }

    /* =====================================
       PDF UPLOADER
    ===================================== */

    [data-testid="stFileUploader"] {
        background: rgba(255, 248, 246, 0.85);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.65rem;
    }

    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzone"] > div {
        background: #FFF8F6 !important;
        border-color: var(--border) !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: var(--accent) !important;
        color: var(--text) !important;
        border: 0 !important;
    }

    /* =====================================
       METRIC CARDS
    ===================================== */

    [data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 16px 18px;
        box-shadow: 0 3px 12px rgba(48, 43, 39, 0.045);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--text) !important;
    }

    /* =====================================
       CHAT MESSAGES
    ===================================== */

    [data-testid="stChatMessage"] {
        background: rgba(255, 248, 246, 0.9) !important;
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.7rem;
        color: var(--text) !important;
    }

    [data-testid="stChatMessageContent"],
    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] li {
        color: var(--text) !important;
    }

    /* =====================================
       CHAT INPUT CONTAINER
    ===================================== */

    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"],
    .stChatFloatingInputContainer {
        background: var(--page) !important;
    }

    [data-testid="stChatInput"] {
        background: var(--input-bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 15px !important;
        box-shadow: 0 3px 12px rgba(73, 59, 56, 0.06) !important;
    }

    /* =====================================
       CHAT INPUT - TYPED TEXT
    ===================================== */

    [data-testid="stChatInput"] textarea {
        color: #F5F5F5 !important;
        -webkit-text-fill-color: #F5F5F5 !important;
        caret-color: #F5F5F5 !important;
        font-size: 17px !important;
        font-weight: 500 !important;
        background: transparent !important;
    }

    /* =====================================
       CHAT INPUT - PLACEHOLDER
    ===================================== */

    [data-testid="stChatInput"] textarea::placeholder {
        color: #806D68 !important;
        -webkit-text-fill-color: #806D68 !important;
        opacity: 1 !important;
    }

    /* =====================================
       CHAT SEND BUTTON
    ===================================== */

    [data-testid="stChatInput"] button {
        background: #FEC5BB !important;
        color: #493B38 !important;
        border: none !important;
        border-radius: 10px !important;
    }

    [data-testid="stChatInput"] button:hover {
        background: #FCD5CE !important;
    }

    /* =====================================
       ALERTS
    ===================================== */

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* =====================================
       PROGRESS BAR
    ===================================== */

    [data-testid="stProgressBar"] > div {
        background: #FCD5CE !important;
    }

    [data-testid="stProgressBar"] > div > div {
        background: #E89B91 !important;
    }

    /* =====================================
       DIVIDERS
    ===================================== */

    hr {
        border-color: var(--border) !important;
    }

    /* =====================================
       STREAMLIT HEADER
    ===================================== */

    [data-testid="stHeader"],
    header[data-testid="stHeader"] {
        background: var(--page) !important;
    }

    [data-testid="stToolbar"] {
        background: transparent !important;
    }

    /* =====================================
       FOOTER
    ===================================== */

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ======================================
# SESSION STATE
# ======================================
def init_state():
    defaults = {
        "messages": [],
        "pdf_ready": False,
        "pdf_name": "",
        "pdf_hash": "",
        "pages": 0,
        "chunks": 0,
        "processing_error": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ======================================
# SIDEBAR — PDF UPLOAD / PROCESSING
# ======================================
with st.sidebar:
    st.markdown("## 🤖 AI PDF Chatbot")
    st.caption("Retrieval-Augmented Generation")
    st.markdown("---")
    st.subheader("📂 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        help="Upload a text-based PDF to ask questions about its contents.",
    )

    if uploaded_file is not None:
        st.success(uploaded_file.name)

        file_bytes = uploaded_file.getvalue()
        current_pdf_hash = hashlib.sha256(file_bytes).hexdigest()

        same_pdf = (
            st.session_state.pdf_ready
            and current_pdf_hash == st.session_state.pdf_hash
        )

        if same_pdf:
            st.caption("This PDF is already processed in this session.")

        if st.button("⚙ Process PDF", disabled=not file_bytes):
            if same_pdf:
                st.info("This PDF is already processed. You can ask questions below.")
            else:
                st.session_state.processing_error = ""
                started = time.perf_counter()

                try:
                    with tempfile.TemporaryDirectory(
                        prefix="ai_pdf_chatbot_"
                    ) as temp_dir:
                        safe_name = os.path.basename(uploaded_file.name) or "document.pdf"
                        pdf_path = os.path.join(temp_dir, safe_name)

                        with open(pdf_path, "wb") as pdf_handle:
                            pdf_handle.write(file_bytes)

                        # 1. Extract PDF text
                        with st.spinner("Processing PDF: extracting text..."):
                            step_start = time.perf_counter()
                            documents = load_pdf(pdf_path)
                            pdf_load_seconds = time.perf_counter() - step_start

                        if not documents:
                            raise ValueError(
                                "No readable text was found in this PDF. "
                                "It may be scanned, image-only, or empty."
                            )

                        # 2. Split into chunks
                        with st.spinner("Splitting document into chunks..."):
                            step_start = time.perf_counter()
                            chunks = split_documents(documents)
                            split_seconds = time.perf_counter() - step_start

                        if not chunks:
                            raise ValueError("No text chunks were created from this PDF.")

                        # 3. Load cached embedding model
                        with st.spinner("Loading embedding model (first run may take longer)..."):
                            step_start = time.perf_counter()
                            embedding_model = load_embeddings()
                            embedding_seconds = time.perf_counter() - step_start

                        # 4. Build and cache FAISS vector store
                        with st.spinner("Creating searchable vector database..."):
                            step_start = time.perf_counter()
                            create_vector_store(chunks, embedding_model)
                            vector_seconds = time.perf_counter() - step_start

                    # Update session state only after successful processing
                    st.session_state.pdf_ready = True
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.pdf_hash = current_pdf_hash
                    st.session_state.pages = len(documents)
                    st.session_state.chunks = len(chunks)
                    st.session_state.messages = []
                    st.session_state.processing_error = ""

                    total_seconds = time.perf_counter() - started
                    st.success(
                        f"PDF processed successfully in {total_seconds:.1f} seconds."
                    )

                    with st.expander("Processing timings"):
                        st.write(f"PDF extraction: {pdf_load_seconds:.2f} seconds")
                        st.write(f"Text splitting: {split_seconds:.2f} seconds")
                        st.write(f"Embedding model loading: {embedding_seconds:.2f} seconds")
                        st.write(f"Vector database creation: {vector_seconds:.2f} seconds")
                        st.caption(
                            "The first model load and embedding generation may take longer."
                        )

                except Exception as exc:
                    st.session_state.processing_error = str(exc)
                    st.error("PDF processing failed. See the error details below.")
                    st.exception(exc)

    st.markdown("---")
    st.subheader("📘 PDF Information")

    if st.session_state.pdf_ready:
        st.success("Ready")
        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.metric("Pages", st.session_state.pages)

        with info_col2:
            st.metric("Chunks", st.session_state.chunks)

        st.info(f"📄 {st.session_state.pdf_name}")
        st.caption("Powered by OpenAI + FAISS")
    else:
        st.warning("Upload a PDF to begin.")

    st.markdown("---")
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ======================================
# MAIN HEADER / WELCOME
# ======================================
st.markdown(
    """
    <div style="text-align:center; padding: 0.3rem 0 0.8rem 0;">
        <h1 style="margin-bottom:0.35rem;">🤖 AI PDF Chatbot</h1>
        <p style="font-size:17px; color:#766D64; margin-top:0;">
            Retrieval-Augmented Generation using LangChain, FAISS &amp; OpenAI
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

if not st.session_state.pdf_ready:
    st.info("👋 Upload a PDF and click **Process PDF** to start chatting.")
elif not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            f"### Welcome 👋\n\n"
            f"Your PDF **{st.session_state.pdf_name}** has been processed successfully.\n\n"
            "You can now ask questions related to the document."
        )


# ======================================
# CHAT HISTORY
# ======================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ======================================
# QUESTION ANSWERING
# ======================================
if st.session_state.pdf_ready:
    question = st.chat_input("💬 Ask anything from your PDF...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer_started = time.perf_counter()
                    result = ask_question(question)
                    answer_seconds = time.perf_counter() - answer_started

                    answer = result.get("answer", "No answer was returned.")
                    pages = result.get("pages", [])
                    confidence = result.get("confidence", 0)
                    timings = result.get("timings", {})

                    with st.expander("⚡ Performance Diagnostics"):  
                     st.json(timings)

                    if not isinstance(answer, str):
                        answer = str(answer)

                    st.markdown(answer)
                    st.caption(f"Response time: {answer_seconds:.2f} seconds")

                    source_col, confidence_col = st.columns(2)

                    with source_col:
                        st.subheader("📄 Source Pages")
                        if pages:
                            st.write(", ".join(map(str, pages)))
                        else:
                            st.write("No pages found.")

                    with confidence_col:
                        st.subheader("📊 Confidence")
                        try:
                            safe_confidence = max(0, min(100, int(confidence)))
                        except (TypeError, ValueError):
                            safe_confidence = 0

                        st.progress(safe_confidence / 100)

                        if safe_confidence >= 90:
                            st.success(f"{safe_confidence}%")
                        elif safe_confidence >= 75:
                            st.info(f"{safe_confidence}%")
                        elif safe_confidence >= 60:
                            st.warning(f"{safe_confidence}%")
                        else:
                            st.error(f"{safe_confidence}%")

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception as exc:
                    error_text = (
                        "Sorry, something went wrong while answering: "
                        f"{exc}"
                    )
                    st.error(error_text)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_text}
                    )
else:
    st.chat_input("Upload and process a PDF first...", disabled=True)


# ======================================
# FOOTER / TECHNOLOGY CARDS
# ======================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.metric("🤖 LLM", "GPT-4.1-mini")

with footer_col2:
    st.metric("🗂 Vector DB", "FAISS")

with footer_col3:
    st.metric("🧠 Framework", "LangChain")

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#766D64;
                font-size:14px; padding:0.4rem 0 0.8rem 0;">
        Built with ❤️ using<br>
        <strong>LangChain</strong> •
        <strong>OpenAI GPT-4.1-mini</strong> •
        <strong>FAISS</strong> •
        <strong>HuggingFace Embeddings</strong> •
        <strong>Streamlit</strong>
    </div>
    """,
    unsafe_allow_html=True,
)