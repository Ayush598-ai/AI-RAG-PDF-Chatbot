# 🤖 AI RAG PDF Chatbot

An AI-powered PDF question-answering application built using **Retrieval-Augmented Generation (RAG)**. It allows users to upload PDF documents and ask questions based on their content.

## 📌 Project Overview

The AI RAG PDF Chatbot extracts text from uploaded PDF documents, divides it into smaller chunks, converts those chunks into numerical embeddings, and stores them in a FAISS vector database.

When a user asks a question, the application retrieves the most relevant information from the PDF and uses it as context to generate an answer.

## ✨ Features

- Upload PDF documents.
- Extract text from PDF files.
- Split documents into manageable text chunks.
- Generate embeddings using HuggingFace.
- Store and search document embeddings using FAISS.
- Ask questions related to uploaded PDFs.
- Retrieve relevant source pages.
- Display response time and performance diagnostics.
- Interactive web interface built with Streamlit.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Streamlit | User interface |
| LangChain | RAG application framework |
| OpenAI GPT-4.1-mini | Language model |
| HuggingFace | Text embeddings |
| FAISS | Vector database |
| PyPDF | PDF text extraction |

## 🏗️ Project Structure

```text
AI-RAG-PDF-Chatbot/
│
├── utils/
│   ├── embeddings.py
│   ├── openai.py
│   ├── pdf_loader.py
│   ├── rag_chain.py
│   ├── text_splitter.py
│   └── vector_store.py
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env.example
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Ayush598-ai/AI-RAG-PDF-Chatbot.git
```

### 2. Open the project folder

```bash
cd AI-RAG-PDF-Chatbot
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file in the project root directory.

Add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Replace the placeholder with your own API key.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## 🚀 How to Use

1. Launch the application.
2. Upload a PDF document.
3. Click **Process PDF**.
4. Wait for document processing to finish.
5. Enter a question related to the uploaded document.
6. View the answer and relevant source pages.

## 🔐 Security

- API keys are stored in a local `.env` file.
- The `.env` file is excluded from GitHub using `.gitignore`.
- Users must provide their own OpenAI API key.

## 👨‍💻 Author

**Ayush Sharma**

GitHub: [Ayush598-ai](https://github.com/Ayush598-ai)

---

⭐ If you find this project useful, consider giving it a star!
