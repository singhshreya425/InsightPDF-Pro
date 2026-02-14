import streamlit as st
import os
import shutil
import gc
from dotenv import load_dotenv

# 1. Imports - Note the switch to FAISS
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # Swapped from Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
load_dotenv()
st.set_page_config(page_title="InsightPDF Pro", page_icon="📑", layout="wide")

# High Contrast UI Styling
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; }
    section[data-testid="stSidebar"] { background-color: #010409 !important; border-right: 1px solid #30363d; }
    [data-testid="stChatMessage"] { border-radius: 15px; padding: 1.5rem; margin-bottom: 1rem; border: 1px solid #30363d; }
    [data-testid="stChatMessage"]:nth-child(even) { background-color: #161b22; color: #ffffff !important; }
    [data-testid="stChatMessage"]:nth-child(odd) { background-color: #21262d; color: #f0f6fc !important; }
    [data-testid="stChatMessage"] p { color: #f0f6fc !important; font-size: 16px !important; }
    h1, h2, h3 { color: #58a6ff !important; }
    .stButton>button { background-color: #238636; color: white; border: none; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

class RAGEngine:
    def __init__(self):
        # Using a reliable open-source embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # API Key Handling
        api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("🔑 Groq API Key not found! Add it to Streamlit Secrets.")
            st.stop()
            
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key)

    def process_pdfs(self, uploaded_files):
        # Memory Management
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
        
        # FAISS initialization (No SQLite required!)
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

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.vectorstore:
    if user_input := st.chat_input("Ask about your PDFs..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # Retrieval logic using FAISS
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

            full_response = st.write_stream(chain.stream(user_input))
            
            with st.expander("📌 View Sources"):
                for i, doc in enumerate(context_docs):
                    st.markdown(f"**Chunk {i+1}:** {doc.page_content[:200]}...")
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.info("👈 Upload and Sync PDFs in the sidebar to start chatting.")