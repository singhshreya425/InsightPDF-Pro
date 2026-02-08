import streamlit as st
import os
import shutil
import gc  # NEW: For memory management/releasing file locks
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

# Modern UI Styling with High Contrast
st.markdown("""
    <style>
    /* Main Background */
    .stApp { 
        background-color: #0d1117; 
        color: #c9d1d9; 
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }

    /* Fix for Top Right Icons (Streamlit Menu/Buttons) */
    .stApp header {
        background-color: rgba(0,0,0,0);
        color: white !important;
    }
    button[kind="header"] {
        color: white !important;
    }

    /* Chat Bubble Styling - High Contrast */
    [data-testid="stChatMessage"] {
        background-color: #21262d; 
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        color: #f0f6fc !important;
    }

    /* Specific fix for text readability in bubbles */
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {
        color: #f0f6fc !important;
        line-height: 1.6;
    }

    /* Input Box Styling */
    .stChatInputContainer {
        padding-bottom: 2rem;
    }

    /* Success/Info Message Contrast */
    .stAlert {
        background-color: #161b22;
        border: 1px solid #30363d;
        color: white;
    }
    
    /* Button Styling */
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        background-color: #238636; 
        color: white;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

class RAGEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile")

    def process_pdfs(self, uploaded_files):
        # 1. Release active ChromaDB connection to avoid Windows PermissionError
        if st.session_state.vectorstore is not None:
            # Explicitly set to None and trigger garbage collection to close file handles
            st.session_state.vectorstore = None
            gc.collect() 
        
        # 2. Safely attempt to delete the old database folder
        if os.path.exists(DB_DIR):
            try:
                shutil.rmtree(DB_DIR, ignore_errors=False)
            except PermissionError:
                # Fallback: If still locked, we proceed but notify the user
                st.sidebar.warning("⚠️ Database busy. Restart app if data from previous session persists.")
        
        all_docs = []
        for uploaded_file in uploaded_files:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader(temp_path)
            all_docs.extend(loader.load())
            if os.path.exists(temp_path): os.remove(temp_path)
            
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(all_docs)
        
        # Create a fresh vectorstore from scratch
        return Chroma.from_documents(chunks, self.embeddings, persist_directory=DB_DIR)

# --- UI INITIALIZATION ---
if "messages" not in st.session_state: st.session_state.messages = []
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
if "engine" not in st.session_state: st.session_state.engine = RAGEngine()

# Sidebar
with st.sidebar:
    st.title("📑 InsightPDF Pro")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files and st.button("🚀 Sync Knowledge Base"):
        with st.spinner("Releasing file locks and indexing fresh data..."):
            st.session_state.vectorstore = st.session_state.engine.process_pdfs(uploaded_files)
        st.success("Fresh Knowledge Base Ready!")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Chat Area
st.title("💬 PDF Intelligence Assistant")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if st.session_state.vectorstore:
    if user_input := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # 1. Retrieve context
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 6})
            context_docs = retriever.invoke(user_input)
            
            # 2. Universal Professional Prompt
            prompt = ChatPromptTemplate.from_template("""
            You are a Professional Research Assistant. 
            Analyze the provided context and answer the question accurately. 
            Adopt a tone appropriate for the subject matter (e.g., technical for manuals, analytical for reports).

            If the context doesn't contain the answer, state that clearly, but offer to answer based on your general knowledge if relevant.

            Context: {context}
            Question: {question}
            Answer:""")
            
            chain = (
                {"context": lambda x: context_docs, "question": RunnablePassthrough()}
                | prompt | st.session_state.engine.llm | StrOutputParser()
            )

            # 3. Stream response
            placeholder = st.empty()
            full_response = ""
            for chunk in chain.stream(user_input):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
            
            # 4. Citation Feature
            with st.expander("📌 View Citations"):
                for i, doc in enumerate(context_docs):
                    page_num = doc.metadata.get('page', 'Unknown')
                    st.markdown(f"**Source {i+1} (Page {page_num})**")
                    st.caption(doc.page_content[:200] + "...")
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.info("👈 Please upload your PDFs in the sidebar to begin.")