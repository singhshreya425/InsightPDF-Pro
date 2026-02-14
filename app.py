try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
import streamlit as st
import os
import shutil
import gc
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Configuration & Styling
load_dotenv()
DB_DIR = "vector_db"
st.set_page_config(page_title="InsightPDF Pro", page_icon="📑", layout="wide")

# High Contrast UI
st.markdown("""
    <style>
    /* Main Background */
    .stApp { 
        background-color: #0d1117; 
    }
    
    /* Sidebar - darker to create depth */
    section[data-testid="stSidebar"] {
        background-color: #010409 !important;
        border-right: 1px solid #30363d;
    }

    /* Chat Messages - High Contrast */
    [data-testid="stChatMessage"] {
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #30363d;
    }

    /* User Message - Slate Blue */
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: #161b22;
        color: #ffffff !important;
    }

    /* Assistant Message - Dark Grey/Black */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #21262d;
        color: #f0f6fc !important;
    }

    /* Force all text inside bubbles to be bright white/off-white */
    [data-testid="stChatMessage"] p, 
    [data-testid="stChatMessage"] li, 
    [data-testid="stChatMessage"] div {
        color: #f0f6fc !important;
        font-size: 16px !important;
        font-weight: 400 !important;
    }

    /* Input box text color */
    .stChatInput textarea {
        color: #ffffff !important;
    }

    /* Headers and Titles */
    h1, h2, h3 {
        color: #58a6ff !important;
    }

    /* Buttons */
    .stButton>button { 
        background-color: #238636; 
        color: white;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

class RAGEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # SAFE KEY RETRIEVAL: Prevents crash if st.secrets is missing locally
        api_key = None
        try:
            if "GROQ_API_KEY" in st.secrets:
                api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
            
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            st.error("🔑 Groq API Key not found! Add it to .env or Streamlit Secrets.")
            st.stop()
            
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key)

    def process_pdfs(self, uploaded_files):
        if st.session_state.vectorstore is not None:
            st.session_state.vectorstore = None
            gc.collect() 
        
        if os.path.exists(DB_DIR):
            try:
                shutil.rmtree(DB_DIR)
            except Exception:
                pass
        
        all_docs = []
        for uploaded_file in uploaded_files:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader(temp_path)
            all_docs.extend(loader.load())
            if os.path.exists(temp_path): os.remove(temp_path)
            
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ".", " "])
        chunks = splitter.split_documents(all_docs)
        return Chroma.from_documents(chunks, self.embeddings, persist_directory=DB_DIR)

# --- INITIALIZATION ---
if "messages" not in st.session_state: st.session_state.messages = []
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
if "engine" not in st.session_state: st.session_state.engine = RAGEngine()

# Sidebar
with st.sidebar:
    st.title("📑 InsightPDF Pro")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files and st.button("🚀 Sync Knowledge Base"):
        with st.spinner("Processing documents..."):
            st.session_state.vectorstore = st.session_state.engine.process_pdfs(uploaded_files)
        st.success("Fresh Knowledge Base Ready!")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # --- FEATURE: DOWNLOAD REPORT ---
    if st.session_state.messages:
        st.markdown("---")
        full_chat = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
        st.download_button(
            label="📄 Download Report",
            data=full_chat,
            file_name="Insight_Report.txt",
            mime="text/plain"
        )

# Chat Area
st.title("💬 PDF Intelligence Assistant")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if st.session_state.vectorstore:
    if user_input := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)

        with st.chat_message("assistant"):
            retriever = st.session_state.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6, 'fetch_k': 20})
            context_docs = retriever.invoke(user_input)
            
            prompt = ChatPromptTemplate.from_template("""
            Analyze the context and answer professionally. 
            Context: {context}
            Question: {question}
            Answer:""")
            
            chain = ({"context": lambda x: context_docs, "question": RunnablePassthrough()} 
                     | prompt | st.session_state.engine.llm | StrOutputParser())

            placeholder = st.empty()
            full_response = ""
            for chunk in chain.stream(user_input):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
            
            with st.expander("📌 View Citations"):
                for i, doc in enumerate(context_docs):
                    st.markdown(f"**Source {i+1}**")
                    st.caption(doc.page_content[:200] + "...")
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.rerun() 
else:
    st.info("👈 Please upload your PDFs to begin.")