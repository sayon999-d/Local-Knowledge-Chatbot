import os
import sys
import time
import socket
from datetime import datetime

import streamlit as st
from streamlit.web.server.websocket_headers import _get_websocket_headers

def main():
    st.set_page_config(
        page_title="RAG Chatbot",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
if __name__ == "__main__":
    import os
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    main()

hostname = socket.gethostname()
container_ip = socket.gethostbyname(hostname)

st.config.set_option('server.address', '0.0.0.0')
st.config.set_option('server.port', 8501)
st.config.set_option('server.headless', True)
st.config.set_option('server.enableCORS', True)
st.config.set_option('server.enableXsrfProtection', True)

os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
os.environ['STREAMLIT_SERVER_PORT'] = '8501'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = "YOUR_LANGCHAIN_API_KEY"
os.environ['LANGCHAIN_PROJECT'] = 'local-rag'

os.environ['GRADIO_OFFLINE_MODE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_EVALUATE_OFFLINE'] = '1'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

import requests
import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import WebBaseLoader
from langchain.docstore.document import Document
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ST_PORT = 8501
CHROMA_DIR = "./chroma_db_data"
MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=MODEL_PATH, model_kwargs={'device': 'cpu'})
MODEL_NAME = "all-MiniLM-L6-v2"
SAVE_DIR = "./saved_articles"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = "mistral:instruct"

os.makedirs(SAVE_DIR, exist_ok=True)

URL_LIST = [
    # AMD Radeon (Official Product & Tech Pages)
    "https://www.amd.com/en/products/graphics/desktops/radeon.html",
    "https://www.amd.com/en/products/graphics/desktops/radeon/7000-series.html",
    "https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xtx.html",
    "https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900-xt.html",
    "https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7800-xt.html",
    "https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7700-xt.html",
    "https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7600-xt.html",
    "https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7600.html",
    "https://www.amd.com/en/support",
    "https://www.amd.com/en/technologies/rdna3",
    "https://pg.asrock.com/Graphics-Card/AMD/Radeon%20RX%207900%20XTX%20Phantom%20Gaming%2024GB%20OC/index.asp",

    # Microsoft DirectML (Windows AI Acceleration)
    "https://learn.microsoft.com/en-us/windows/ai/directml/dml",
    "https://learn.microsoft.com/en-us/windows/ai/directml/dml-get-started",
    "https://learn.microsoft.com/en-us/windows/ai/directml/dml-ops",

    # ONNX Runtime
    "https://onnxruntime.ai/docs/",
    "https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html",

    # ROCm
    "https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html",

    # Stable Diffusion on AMD / DirectML
    "https://github.com/microsoft/Stable-Diffusion-WebUI-DirectML",

    # Developer Docs
    "https://learn.microsoft.com/en-us/windows/ai/windows-ml/",
    "https://www.club386.com/nvidia-geforce-rtx-5080-vs-amd-radeon-rx-7900-xtx/",
    "https://www.tomshardware.com/reviews/gpu-hierarchy,4388.html",
    "https://www.sapphiretech.com/en/consumer/nitro-radeon-rx-7900-xtx-vaporx-24g-gddr6",
    "https://llm-tracker.info/_TOORG/RTX-3090-vs-7900-XTX-Comparison",
    "https://www.pugetsystems.com/labs/articles/2025-consumer-gpu-content-creation-roundup/",
    "https://www.pugetsystems.com/labs/articles/amd-radeon-rx-9070-xt-content-creation-review/",
    "https://www.byteplus.com/en/topic/376338?title=radeon-rx-7900-xtx-deepseek-benchmark-comprehensive-ai-performance-analysis-for-2025",
    "https://www.gigabyte.com/Graphics-Card/GV-R79XTXAORUS-E-24GD",
    "https://pcpartpicker.com/forums/topic/443648-4080-super-vs-7900-xtx-for-editing",
]

st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
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
    </style>
""", unsafe_allow_html=True)

def save_text_to_file(source_name, content):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_name = source_name.split("//")[-1].replace("/", "_")[:30]
        filename = f"{clean_name}_{timestamp}.txt"
        filepath = os.path.join(SAVE_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Source: {source_name}\n\n{content}")
        return filename
    except:
        return None

class RAGSystem:
    def __init__(self):
        self.qa_chain = None
        self.embeddings = None
        self.vectorstore = None
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    def check_internet(self):
        try:
            requests.get("https://www.google.com", timeout=1)
            return True
        except:
            return False

    def get_embeddings(self):
        
        Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    
        if os.path.exists(MODEL_PATH) and len(os.listdir(MODEL_PATH)) > 0:
            return HuggingFaceEmbeddings(
                model_name=MODEL_PATH,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

        if not self.check_internet():
            st.error("OFFLINE ERROR: First run requires internet to download AI models.")
            st.stop()

        with st.spinner("Downloading AI Models (One-time setup)..."):
            model = SentenceTransformer(MODEL_NAME)
            model.save(MODEL_PATH)  
        return HuggingFaceEmbeddings(
            model_name=MODEL_PATH,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def initialize(self):
        try:
            # Get embeddings
            self.embeddings = self.get_embeddings()
        
            # Check if we have existing data
            if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
                logger.info("Loading existing vector store...")
                self.vectorstore = Chroma(
                    persist_directory=CHROMA_DIR,
                    embedding_function=self.embeddings,
                    collection_metadata={"hnsw:space": "cosine"}
                )
            else:
                logger.info("No existing vector store found. Building knowledge base...")
                # Initialize with empty collection
                self.vectorstore = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=CHROMA_DIR,
                    collection_metadata={"hnsw:space": "cosine"}
                )
            
            # Add initial content if needed
            with st.spinner("Building Knowledge Base..."):
                loader = WebBaseLoader(URL_LIST)
                raw_docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=100
                )
                split_docs = text_splitter.split_documents(raw_docs)
                self.vectorstore.add_documents(split_docs)
                self.vectorstore.persist()
        
            # Initialize the QA chain
            self.update_chain()
            return True
        
        except Exception as e:
            st.error(f"Initialization error: {str(e)}")
            logger.error(f"Initialization failed: {str(e)}", exc_info=True)
            return False

    def update_chain(self):
        llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            # OPTIMIZATION: k=2 makes it faster (reads less data)
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True
        )

    def add_content(self, source_name, text_content):
        try:
            save_text_to_file(source_name, text_content)
            doc = Document(page_content=text_content, metadata={"source": source_name})
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_docs = text_splitter.split_documents([doc])
            self.vectorstore.add_documents(split_docs)
            self.update_chain()
            return True, "Knowledge Added Successfully"
        except Exception as e:
            return False, str(e)

    def add_url(self, url):
        if not self.check_internet():
            return False, "Internet required for URLs."
        try:
            loader = WebBaseLoader(url)
            new_docs = loader.load()
            if not new_docs: return False, "No content found."
            return self.add_content(url, new_docs[0].page_content)
        except Exception as e:
            return False, str(e)

    def ask(self, question):
        if not self.qa_chain:
            return {"answer": "Initializing...", "sources": []}
        try:
            res = self.qa_chain.invoke({"query": question})
            return {"answer": res["result"], "sources": res.get("source_documents", [])}
        except Exception as e:
            return {"answer": f"Error: {str(e)}", "sources": []}

@st.cache_resource
def get_engine():
    sys = RAGSystem()
    if sys.initialize():
        return sys
    return None

engine = get_engine()

with st.sidebar:
    st.title("Dashboard")
    st.markdown("---")
    
    st.subheader("Scrape & Learn")
    st.caption("Add new knowledge to the AI instantly.")
    
    # Toggle for Online/Offline Input
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
        # 1GB Upload Limit Notice
        uploaded_file = st.file_uploader("Upload Text (Max 1GB)", type=['txt'], help="Supports large text files up to 1GB")
        if uploaded_file and st.button("📂 Process File"):
            if engine:
                with st.spinner("Reading File..."):
                    try:
                        content = uploaded_file.read().decode("utf-8")
                        ok, msg = engine.add_content(uploaded_file.name, content)
                        if ok: 
                            st.success("Added!")
                            time.sleep(1)
                            st.rerun()
                        else: 
                            st.error(msg)
                    except Exception as e:
                        st.error(f"Error: {e}")

    st.markdown("---")
    st.caption(f"Status: {'Online' if engine and engine.check_internet() else 'Offline Mode'}")

# --- MAIN CHAT AREA ---
st.markdown("""
    <h2 style="color: #2c3e50;">🤖 Local AI Assistant</h2>
    <p style="color: #555;">
        Running on <strong>Localhost</strong> | <strong>Fast Mode (k=2)</strong>
    </p>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("View Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")

# Input Area
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if engine:
        with st.chat_message("assistant"):
            with st.spinner("AI is thinking..."):
                response = engine.ask(prompt)
                
                st.markdown(response["answer"])
                
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
    else:
        st.error("System is not ready. Check if Ollama is running.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "streamlit.web.cli:main",
        host="0.0.0.0",
        port=8501,
        log_level="info",
        reload=False,
        workers=1
    )
