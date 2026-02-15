# 📑 InsightPDF Pro: AI-Powered Document Intelligence
**InsightPDF Pro** is a production-ready Retrieval-Augmented Generation (RAG) application that allows users to have natural conversations with multiple PDF documents simultaneously.
[![Streamlit App] https://insightpdf-pro-bmtgqefrqsdjjkg6bu9jjw.streamlit.app/

## 🚀 Features
* **Multi-PDF Support:** Upload and sync multiple documents into a single knowledge base.
* **Instant Inference:** Powered by **Groq (Llama 3.3-70B)** for near-zero latency responses.
* **Smart Retrieval:** Uses **MMR (Maximum Marginal Relevance)** to ensure diverse and non-redundant context retrieval.
* **High-Contrast UI:** Custom CSS-themed interface for professional-grade readability.

## 🛠️ Tech Architecture
I engineered this system to handle common cloud-deployment bottlenecks:
1. **Document Processing:** Uses `PyPDFLoader` and `RecursiveCharacterTextSplitter` to maintain semantic context within 1000-character chunks.
2. **Vector Store (FAISS):** Migrated from ChromaDB to **FAISS (Facebook AI Similarity Search)**. This decision was critical to resolve SQLite versioning conflicts in serverless environments, resulting in faster in-memory lookups.
3. **Embeddings:** Leverages `all-MiniLM-L6-v2` via HuggingFace for a balanced trade-off between vector precision and memory efficiency.
4. **Logic Layer:** Built with **LangChain's Expression Language (LCEL)** for a modular and scalable RAG pipeline.

## ⚡ Technical Challenges Overcome
* **The SQLite Conflict:** Encountered a common deployment issue where Streamlit Cloud's SQLite version was incompatible with ChromaDB's Rust-based bindings. Solved this by pivoting the architecture to FAISS, which uses C++-optimized in-memory indexing, removing the system-level dependency.
* **Context Overflow:** Fine-tuned the `chunk_overlap` to 150 characters to ensure the LLM doesn't lose data "between" chunks.

## 📦 Local Setup
1. Clone the repo: `git clone https://github.com/singhshreya425/InsightPDF-Pro.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Set up `.env`: Add your `GROQ_API_KEY`.
4. Run: `streamlit run app.py`
