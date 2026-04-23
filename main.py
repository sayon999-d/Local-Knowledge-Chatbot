import os
import sys
import time
import hashlib
import socket
import re
import ipaddress
from urllib.parse import urlparse
from datetime import datetime

import streamlit as st

os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_PROJECT'] = 'local-rag'
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

import requests
import logging
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader
from langchain.docstore.document import Document
from dotenv import load_dotenv
from hf_embeddings import HuggingFaceAPIEmbeddings

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_QUESTION_LENGTH = 1000       
MAX_GROQ_TOKENS = 512            
MAX_QUESTIONS_PER_SESSION = 30   
MAX_FILE_UPLOAD_MB = 10          
MAX_FILE_CONTENT_CHARS = 50000   


def get_secret(key, default=None):
    val = os.getenv(key)
    if val:
        return val
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


GROQ_API_KEY = get_secret("GROQ_API_KEY")
GROQ_MODEL = get_secret("GROQ_MODEL", "llama-3.1-8b-instant")  
HUGGINGFACE_API_KEY = get_secret("HUGGINGFACE_API_KEY")
PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
PINECONE_INDEX_NAME = get_secret("PINECONE_INDEX_NAME", "rag-chatbot")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(
    page_title="AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* 1. Main Background - Soft Grey (Easier on eyes) */
    .stApp {
        background-color: #f4f6f9;
    }
    
    /* 2. Text Visibility - Force Dark Grey */
    h1, h2, h3, p, div, span, li {
        color: #2c3e50 !important;
    }
    
    /* 3. Sidebar - Dark Mode with White Text */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] p {
        color: #ffffff !important;
    }
    
    /* 4. Buttons - Modern & High Contrast */
    .stButton button {
        background-color: #4CAF50; /* Green Pop */
        color: white !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    
    /* 5. Chat Bubbles - Card Style */
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 10px;
        border-left: 4px solid #4CAF50;
    }
    
    /* 6. Input Box */
    .stTextInput input {
        border-radius: 10px;
        border: 1px solid #ced4da;
    }

    /* 7. Chat Input - keep the message box visible */
    [data-testid="stChatInputContainer"],
    [data-testid="stChatInput"] {
        background: rgba(244, 246, 249, 0.96);
        border-top: 1px solid #d8dee6;
        padding-top: 0.75rem;
    }
    [data-testid="stChatInput"] > div {
        background-color: #ffffff !important;
        border: 2px solid #b8c4d1 !important;
        border-radius: 14px !important;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08) !important;
    }
    [data-testid="stChatInputContainer"] *,
    [data-testid="stChatInput"] * {
        color: #1f2937 !important;
    }
    textarea[data-testid="stChatInputTextArea"],
    [data-testid="stChatInput"] textarea {
        background-color: #ffffff !important;
        color: #1f2937 !important;
        -webkit-text-fill-color: #1f2937 !important;
        caret-color: #1f2937 !important;
        opacity: 1 !important;
        text-shadow: none !important;
        filter: none !important;
        mix-blend-mode: normal !important;
        font-size: 1rem !important;
        font-weight: 500;
    }
    textarea[data-testid="stChatInputTextArea"]::placeholder,
    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
        opacity: 1;
    }
    textarea[data-testid="stChatInputTextArea"]::-webkit-input-placeholder,
    [data-testid="stChatInput"] textarea::-webkit-input-placeholder {
        color: #64748b !important;
    }
    textarea[data-testid="stChatInputTextArea"]::-moz-placeholder,
    [data-testid="stChatInput"] textarea::-moz-placeholder {
        color: #64748b !important;
        opacity: 1;
    }
    textarea[data-testid="stChatInputTextArea"]::selection,
    [data-testid="stChatInput"] textarea::selection {
        background: rgba(76, 175, 80, 0.25);
        color: #111827 !important;
    }
    [data-testid="stChatInput"]:focus-within > div,
    [data-testid="stChatInputContainer"] textarea:focus,
    textarea[data-testid="stChatInputTextArea"]:focus,
    [data-testid="stChatInput"] textarea:focus {
        border-color: #4CAF50 !important;
        box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.18) !important;
        outline: none !important;
    }
    [data-testid="stChatInputContainer"] button,
    [data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"],
    [data-testid="stChatInput"] button[data-testid="stChatInputMicButton"] {
        background-color: #4CAF50 !important;
        color: #ffffff !important;
        border-radius: 12px !important;
    }
    [data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"]:disabled,
    [data-testid="stChatInput"] button[data-testid="stChatInputMicButton"]:disabled {
        background-color: #94a3b8 !important;
        color: #ffffff !important;
    }
    [data-testid="stChatInputContainer"] button svg,
    [data-testid="stChatInput"] button svg {
        fill: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

def clean_filename(filename):
    filename = os.path.basename(filename)
    filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    if not filename or filename.startswith('.'):
        filename = f"upload_{int(time.time())}.txt"
    return filename

def validate_url(url):
    try:
        parsed = urlparse(url)
        if not parsed.scheme or parsed.scheme not in ['http', 'https']:
            return False, "Invalid URL scheme"
        hostname = parsed.hostname
        if not hostname: return False, "Invalid hostname"
        try:
            ip = socket.gethostbyname(hostname)
        except socket.error:
            return False, "Could not resolve hostname"
        ip_obj = ipaddress.ip_address(ip)
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local:
            return False, "Restricted local/private IP"
        return True, "Valid"
    except Exception as e:
        return False, f"Validation error: {e}"

def init_session_state():
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0
    if "total_tokens_used" not in st.session_state:
        st.session_state.total_tokens_used = 0
    if "response_cache" not in st.session_state:
        st.session_state.response_cache = {}


def check_rate_limit():
    if st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION:
        return False
    return True


def get_cache_key(question):
    return hashlib.md5(question.strip().lower().encode()).hexdigest()

class RAGSystem:
    def __init__(self):
        self.qa_chain = None
        self.embeddings = None
        self.vectorstore = None

    def get_embeddings(self):
        if not HUGGINGFACE_API_KEY:
            st.error("HUGGINGFACE_API_KEY environment variable is not set!")
            st.stop()
        return HuggingFaceAPIEmbeddings(
            api_key=HUGGINGFACE_API_KEY,
            model_name=EMBEDDING_MODEL
        )

    def initialize(self):
        try:
            if not GROQ_API_KEY:
                st.error("GROQ_API_KEY environment variable is not set! Please set it to deploy.")
                return False
            if not PINECONE_API_KEY:
                st.error("PINECONE_API_KEY environment variable is not set! Please set it to deploy.")
                return False

            self.embeddings = self.get_embeddings()

            logger.info("Connecting to Pinecone vector store...")
            os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
            self.vectorstore = PineconeVectorStore(
                index_name=PINECONE_INDEX_NAME,
                embedding=self.embeddings
            )

            self.update_chain()
            logger.info("Connected to Pinecone successfully. Ready to serve.")
            return True

        except Exception as e:
            st.error(f"Initialization error: {str(e)}")
            logger.error(f"Initialization failed: {str(e)}", exc_info=True)
            return False

    def update_chain(self):
        llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.1,
            max_tokens=MAX_GROQ_TOKENS,       
            request_timeout=15,                
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True
        )

    def add_content(self, source_name, text_content):
        try:
            text_content = text_content[:MAX_FILE_CONTENT_CHARS]
            doc = Document(page_content=text_content, metadata={"source": source_name})
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_docs = text_splitter.split_documents([doc])
            self.vectorstore.add_documents(split_docs)
            return True, f"Added {len(split_docs)} chunks successfully"
        except Exception as e:
            return False, str(e)

    def add_url(self, url):
        is_valid, msg = validate_url(url)
        if not is_valid:
            return False, f"Security Warning: {msg}"
        try:
            loader = WebBaseLoader(url)
            new_docs = loader.load()
            if not new_docs: return False, "No content found."
            return self.add_content(url, new_docs[0].page_content)
        except Exception as e:
            return False, str(e)

    def ask(self, question):
        if not self.qa_chain:
            return {"answer": "Initializing...", "sources": [], "cached": False}

        question = question[:MAX_QUESTION_LENGTH]

        cache_key = get_cache_key(question)
        if cache_key in st.session_state.response_cache:
            cached = st.session_state.response_cache[cache_key]
            cached["cached"] = True
            return cached

        try:
            res = self.qa_chain.invoke({"query": question})
            response = {
                "answer": res["result"],
                "sources": res.get("source_documents", []),
                "cached": False
            }
            st.session_state.response_cache[cache_key] = response
            st.session_state.total_tokens_used += len(question) // 4 + len(res["result"]) // 4
            return response
        except Exception as e:
            return {"answer": f"Error: {str(e)}", "sources": [], "cached": False}

@st.cache_resource
def get_engine():
    sys = RAGSystem()
    if sys.initialize():
        return sys
    return None

engine = get_engine()
init_session_state()

with st.sidebar:
    st.title("Dashboard")
    st.markdown("---")

    st.subheader("Usage")
    remaining = MAX_QUESTIONS_PER_SESSION - st.session_state.question_count
    st.progress(st.session_state.question_count / MAX_QUESTIONS_PER_SESSION)
    st.caption(f"Questions: {st.session_state.question_count}/{MAX_QUESTIONS_PER_SESSION} | ~{st.session_state.total_tokens_used} tokens used")

    st.markdown("---")
    st.subheader("Scrape & Learn")
    st.caption("Add new knowledge to the AI instantly.")
    
    add_mode = st.radio("Source Type:", ["Web URL", "File Upload"], label_visibility="collapsed")
    
    if add_mode == "Web URL":
        new_url = st.text_input("Paste Website Link:")
        if st.button("Scrape URL"):
            if engine and new_url:
                with st.spinner("Scraping & Learning..."):
                    ok, msg = engine.add_url(new_url)
                    if ok: 
                        st.success("Added!")
                        time.sleep(1)
                        st.rerun()
                    else: 
                        st.error(msg)
                    
    elif add_mode == "File Upload":
        uploaded_file = st.file_uploader(f"Upload Text (Max {MAX_FILE_UPLOAD_MB}MB)", type=['txt'], help="Supports text files")
        if uploaded_file and st.button("📂 Process File"):
            if uploaded_file.size > MAX_FILE_UPLOAD_MB * 1024 * 1024:
                st.error(f"File too large. Maximum {MAX_FILE_UPLOAD_MB}MB.")
            elif engine:
                with st.spinner("Reading File..."):
                    try:
                        content = uploaded_file.read().decode("utf-8")
                        ok, msg = engine.add_content(uploaded_file.name, content)
                        if ok: 
                            st.success(msg)
                            time.sleep(1)
                            st.rerun()
                        else: 
                            st.error(msg)
                    except Exception as e:
                        st.error(f"Error: {e}")

    st.markdown("---")
    st.caption(f"Model: {GROQ_MODEL} | Max tokens: {MAX_GROQ_TOKENS}")

st.markdown("""
    <h2 style="color: #2c3e50;">AI Assistant</h2>
    <p style="color: #555;">
        Powered by <strong>Groq ({groq_model})</strong> | <strong>Fast Mode (k=2)</strong>
    </p>
""".format(groq_model=GROQ_MODEL), unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("View Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")

if prompt := st.chat_input("Ask me anything..."):
    if len(prompt) > MAX_QUESTION_LENGTH:
        st.warning(f"Question truncated to {MAX_QUESTION_LENGTH} characters to save tokens.")
        prompt = prompt[:MAX_QUESTION_LENGTH]

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not engine:
        st.error("System is not ready. Check if GROQ_API_KEY is set.")
    elif not check_rate_limit():
        st.warning(f"Rate limit reached ({MAX_QUESTIONS_PER_SESSION} questions/session). Refresh the page to reset.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("AI is thinking..."):
                response = engine.ask(prompt)
                st.session_state.question_count += 1

                st.markdown(response["answer"])
                if response.get("cached"):
                    st.caption("⚡ Served from cache (no API tokens used)")

                sources_clean = list(set([d.metadata.get("source", "Unknown") for d in response["sources"]]))
                if sources_clean:
                    with st.expander("View Sources"):
                        for src in sources_clean:
                            st.markdown(f"- [{src}]({src})")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": sources_clean
                })
