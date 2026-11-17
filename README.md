# ⚡ Enterprise Document Intelligence & RAG Assistant

An industry-grade, production-level Retrieval-Augmented Generation (RAG) system built with Python, LangChain, FAISS, BM25 Hybrid Search, and Google Gemini (`gemini-1.5-flash`), paired with a sleek, modern enterprise-grade Streamlit web interface.

---

## 🚀 Key Features

* **Advanced Semantic & Recursive Chunking:** Intelligent document parsing that preserves structural context and splits text cleanly using overlap.
* **Hybrid Search (FAISS + BM25 Ensemble):** Combines dense vector similarity search (conceptual understanding) with sparse keyword retrieval (exact entity matching) using LangChain's EnsembleRetriever.
* **Google Gemini Integration:** Utilizes `gemini-1.5-flash` for high-speed, grounded answer generation with built-in exponential backoff rate-limit handling (HTTP 429 retries).
* **Precise Source Attributions:** Every response dynamically cites its source document, page number, and chunk ID, rendered via expandable UI cards.
* **Modern Enterprise UI:** A high-end dark-mode web dashboard featuring gradient branding, live telemetry system status, and an intuitive sidebar document manager.

---

## 🛠️ Tech Stack

* **Core Language:** Python 3.10+
* **LLM & Embeddings:** Google Generative AI (`gemini-1.5-flash`), Sentence Transformers (`all-MiniLM-L6-v2`)
* **Vector Store & Retrieval:** FAISS, BM25, LangChain
* **Frontend & UI:** Streamlit with custom CSS styling

---

## 📂 Project Architecture

```text
enterprise-rag-assistant/
│
├── data/
│   └── documents/          # Secure storage for uploaded PDFs and text files
├── vectorstore/
│   └── faiss_index/        # Local persistent FAISS vector database
├── src/
│   ├── chunking.py         # Document loading and recursive text splitting
│   ├── embeddings.py       # Embedding model initialization
│   ├── retriever.py        # Hybrid search (FAISS + BM25) and retrieval logic
│   ├── rag_pipeline.py     # Gemini LLM generation and retry rate-limiting logic
│   └── evaluation.py       # Retrieval accuracy and RAG evaluation tools
├── app.py                  # High-end Streamlit web interface
├── requirements.txt        # Project dependencies
├── .env.example            # Environment variables template
└── README.md               # Project documentation
