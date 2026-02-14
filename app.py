import streamlit as st
import os
import shutil
import gc
from dotenv import load_dotenv

# 1. Imports - Using FAISS for cloud stability (No SQLite required)
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
load_dotenv()
st.set_page_config(page_title="InsightPDF Pro", page_icon="📑", layout="wide")

# High Contrast UI Styling - Updated for consistent text color during streaming
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #0d1117; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { 
        background-color: #010409 !important; 
        border-right: 1px solid #30363d; 
    }

    /* Chat Messages Container */
    [data-testid="stChatMessage"] { 
        border-radius: 15px; 
        padding: 1.5rem; 
        margin-bottom: 1rem; 
        border: 1px solid #30363d; 
    }

    /* Message Bubbles - Alternating Colors */
    [data-testid="stChatMessage"]:nth-child(even) { background-color: #161b22; }
    [data-testid="stChatMessage"]:nth-child(odd) { background-color: #21262d; }

    /* UNIVERSAL TEXT COLOR FIX */
    /* Targets paragraphs, spans (streaming text), and markdown divs */
    [data-testid="stChatMessage"] p, 
    [data-testid="stChatMessage"] span, 
    [data-testid="stChatMessage"] div,
    .stMarkdown p { 
        color: #f0f6fc !important; 
        font-size: 16px !important; 
        opacity: 1 !important;
        line-height: 1.6;
    }

    /* Headers and Titles */
    h1, h2, h3 { color: #58a6ff !important; }
    
    /* Sidebar Buttons */
    .stButton>button { 
        background-color: #238636; 
        color: white; 
        border: none; 
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

class RAGEngine:
    def __init__(self):
        # Reliable open-source embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # API Key Handling (Cloud Secrets + Local .env support)
        api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("🔑 Groq API Key not found! Add it to Streamlit Secrets.")
            st.stop()
            
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key)

    def process_pdfs(self, uploaded_files):
        # Clear memory and old vectorstore
        if st.session_state.vectorstore is not None:
            st.session_state.vectorstore = None
            gc.collect() 
        
        all_docs = []
        for uploaded_file in uploaded_files:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                loader = PyPDFLoader(temp_path)
                all_docs.extend(loader.load())
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150, 
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_documents(all_docs)
        
        # FAISS initialization (In-memory, perfect for Streamlit Cloud)
        return FAISS.from_documents(chunks, self.embeddings)

# --- INITIALIZATION ---
if "messages" not in st.session_state: st.session_state.messages = []
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
if "engine" not in st.session_state: st.session_state.engine = RAGEngine()

# --- SIDEBAR ---
with st.sidebar:
    st.title("📑 InsightPDF Pro")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if uploaded_files and st.button("🚀 Sync Knowledge Base"):
        with st.spinner("Analyzing PDF content..."):
            st.session_state.vectorstore = st.session_state.engine.process_pdfs(uploaded_files)
        st.success("Knowledge Base Ready!")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT AREA ---
st.title("💬 PDF Intelligence Assistant")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Interaction
if st.session_state.vectorstore:
    if user_input := st.chat_input("Ask about your PDFs..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # MMR Retrieval for diverse context
            retriever = st.session_state.vectorstore.as_retriever(
                search_type="mmr", 
                search_kwargs={"k": 5}
            )
            context_docs = retriever.invoke(user_input)
            
            prompt = ChatPromptTemplate.from_template("""
            Analyze the context and answer the question professionally.
            Context: {context}
            Question: {question}
            Answer:""")
            
            chain = ({"context": lambda x: context_docs, "question": RunnablePassthrough()} 
                     | prompt | st.session_state.engine.llm | StrOutputParser())

            # Use native stream for better UI experience
            full_response = st.write_stream(chain.stream(user_input))
            
            with st.expander("📌 View Sources"):
                for i, doc in enumerate(context_docs):
                    st.markdown(f"**Source {i+1}:** {doc.page_content[:250]}...")
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.info("👈 Upload and Sync PDFs in the sidebar to start chatting.")