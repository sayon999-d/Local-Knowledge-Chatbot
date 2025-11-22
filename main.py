import os
import sys
import time
import traceback
from datetime import datetime

# ==========================================
# 1. CONFIGURATION & SETTINGS
# ==========================================
# Disable Gradio analytics and set User Agent
os.environ['GRADIO_OFFLINE_MODE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ["USER_AGENT"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

import streamlit as st
import socket
import requests
import logging

from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import WebBaseLoader
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Load Environment
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 2. CONSTANTS
# ==========================================
ST_PORT = 8501
CHROMA_DIR = "./chroma_db_data"
MODEL_PATH = "./local_embeddings/mxbai-embed-large"
MODEL_NAME = "mxbai-embed-large"
SAVE_DIR = "./saved_articles"
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://host.docker.internal:11434')
OLLAMA_MODEL = "mistral:instruct"
EMBEDDING_MODEL = "mxbai-embed-large"

# Ensure save directory exists
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

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def save_text_to_file(source_name, content):
    """Saves raw text content to the local folder"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_name = source_name.split("//")[-1].replace("/", "_").replace(".", "_")[:30]
        filename = f"{clean_name}_{timestamp}.txt"
        filepath = os.path.join(SAVE_DIR, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Source: {source_name}\n")
            f.write("="*50 + "\n\n")
            f.write(content)
        return filename
    except Exception as e:
        return None

def check_internet():
    try:
        requests.get("https://www.google.com", timeout=1)
        return True
    except:
        return False

def get_smart_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.168.1.1", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

# ==========================================
# 4. THE RAG ENGINE
# ==========================================
class RAGSystem:
    def __init__(self):
        self.qa_chain = None
        self.embeddings = None
        self.vectorstore = None
    
    # Ensure Chroma DB directory exists
    os.makedirs(CHROMA_DIR, exist_ok=True)
        
    # Only create directory if EMBEDDING_MODEL has a directory path
    if os.path.dirname(EMBEDDING_MODEL):
        os.makedirs(os.path.dirname(EMBEDDING_MODEL), exist_ok=True)
        
    # Ensure Chroma DB directory exists
    os.makedirs(CHROMA_DIR, exist_ok=True)
    def get_smart_embeddings(self):
        try:
            logger.info("Using local Ollama model for embeddings")
            return OllamaEmbeddings(
            model=MODEL_NAME,
            base_url=OLLAMA_BASE_URL
        )

        except Exception as e:
            logger.warning(f"Could not initialize local model: {str(e)}")

            # If local model fails, try to download
            if not check_internet():
                raise ConnectionError(
                    "No internet connection and no valid local model found.\n"
                    "Please either:\n"
                    "1. Connect to the internet to download the model\n"
                    f"2. Place the model files in: {os.path.abspath(MODEL_PATH)}"
                )

            logger.info(f"Downloading model: {MODEL_NAME}...")
            os.makedirs(MODEL_PATH, exist_ok=True)
            
            try:
                model = SentenceTransformer(MODEL_NAME)
                model.save(MODEL_PATH)
                logger.info(f"Model successfully downloaded and saved to: {os.path.abspath(MODEL_PATH)}")
                return HuggingFaceEmbeddings(model_name=MODEL_PATH)
            except Exception as e:
                logger.error(f"Failed to download model: {str(e)}")
                if os.path.exists(MODEL_PATH):
                    logger.info(f"Cleaning up partially downloaded model at: {MODEL_PATH}")
                    import shutil
                    shutil.rmtree(MODEL_PATH)
                raise
                
        except Exception as e:
            logger.error(f"Failed to initialize Ollama embeddings: {str(e)}")
            logger.info("\nPlease ensure Ollama is running and the model is downloaded.")
            logger.info(f"Run 'ollama pull {MODEL_NAME}' to download the model.")
            raise

    def initialize(self):
        try:
            # Initialize embeddings
            self.embeddings = self.get_smart_embeddings()
            
            # Check if we already have a vector store
            if os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0:
                try:
                    logger.info("Loading database from disk...")
                    self.vectorstore = Chroma(
                        persist_directory=CHROMA_DIR,
                        embedding_function=self.embeddings
                    )
                    logger.info("Database loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading database: {str(e)}")
                    logger.info("Creating a new database...")
                    self._create_new_database()
            else:
                logger.info("No existing database found. Creating a new one...")
                self._create_new_database()
            
            # Initialize the QA chain
            self.update_chain()
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def _create_new_database(self):
        """Helper method to create a new vector database"""
        try:
            # Load documents
            logger.info("Loading documents...")
            loader = WebBaseLoader(URL_LIST)
            documents = loader.load()
            
            # Split documents
            logger.info("Processing documents...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            texts = text_splitter.split_documents(documents)
            
            # Create and store the vector store
            logger.info("Creating vector store...")
            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory=CHROMA_DIR
            )
            logger.info("Vector database created and saved to disk.")
            
        except Exception as e:
            logger.error(f"Failed to create database: {str(e)}")
            raise

    def update_chain(self):
        """Refreshes the QA chain"""
        try:
            llm = Ollama(
                model=OLLAMA_MODEL,  # Changed from MODEL_NAME to OLLAMA_MODEL
                    base_url=OLLAMA_BASE_URL,
                    temperature=0.1
                )
                
            if self.vectorstore:
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True
                )
        except Exception as e:
            logger.error(f"Failed to initialize QA chain: {str(e)}")
            raise

    def add_content(self, source_name, text_content):
        try:
            filename = save_text_to_file(source_name, text_content)
            doc = Document(page_content=text_content, metadata={"source": source_name})
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
            split_docs = text_splitter.split_documents([doc])
            self.vectorstore.add_documents(split_docs)
            self.update_chain()
            return True, f"Added content from '{source_name}' and saved file."
        except Exception as e:
            return False, str(e)

    def add_url(self, url):
        if not check_internet():
            return False, "Internet Required to scrape URLs. Use File Upload instead."
        try:
            loader = WebBaseLoader(url)
            new_docs = loader.load()
            if not new_docs: return False, "No content found."
            return self.add_content(url, new_docs[0].page_content)
        except Exception as e:
            return False, str(e)

    def ask(self, question: str):
        if not self.qa_chain:
            return {"answer": "System is initializing...", "sources": []}
        try:
            response = self.qa_chain.invoke({"query": question})
            return {
                "answer": response["result"],
                "sources": response.get("source_documents", [])
            }
        except Exception as e:
            return {"answer": f"Error: {str(e)}", "sources": []}

@st.cache_resource
def get_engine():
    system = RAGSystem()
    try:
        system.initialize()
        return system
    except Exception as e:
        st.error(str(e))
        return None

engine = get_engine()

# ==========================================
# 5. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Local RAG System", layout="wide")

LOCAL_IP = get_smart_ip()
IS_ONLINE = check_internet()
STATUS_TEXT = "Online Mode" if IS_ONLINE else "Offline Mode"

st.markdown("""
    <style>
    .status-badge { 
        display: inline-block; 
        padding: 5px 12px; 
        background-color: #f0f2f6; 
        border-radius: 5px; 
        border: 1px solid #ccc; 
        font-family: sans-serif;
    }
    .link-box { 
        padding: 15px; 
        border: 1px solid #ddd; 
        border-radius: 5px; 
        background-color: #f9f9f9; 
        margin-bottom: 15px; 
    }
    .stButton button { width: 100%; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR (Add Knowledge) ---
with st.sidebar:
    st.header("Add Knowledge")
    
    add_mode = st.radio("Source Type:", ["Web URL", "File Upload"])
    
    if add_mode == "Web URL":
        new_url = st.text_input("Enter Website URL:")
        if st.button("Scrape and Learn"):
            if not new_url:
                st.warning("Please enter a URL.")
            elif not engine:
                st.error("Engine not running.")
            else:
                with st.spinner("Processing..."):
                    success, msg = engine.add_url(new_url)
                    if success:
                        st.success(msg)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(msg)
                        
    elif add_mode == "File Upload":
        uploaded_file = st.file_uploader(
            "Upload Text File (Max 1GB)", 
            type=['txt'],
            help="Ensure you run the app with --server.maxUploadSize=1024"
        )
        if uploaded_file and st.button("Read and Learn"):
            if not engine:
                st.error("Engine not running.")
            else:
                with st.spinner("Reading File..."):
                    text_content = uploaded_file.read().decode("utf-8")
                    success, msg = engine.add_content(uploaded_file.name, text_content)
                    if success:
                        st.success(msg)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(msg)

# --- MAIN PAGE ---
st.title("Local RAG System")
st.markdown(f'<div class="status-badge">Status: {STATUS_TEXT}</div>', unsafe_allow_html=True)

# Network Info
net_msg = f"Local Network: {LOCAL_IP}" if LOCAL_IP != "127.0.0.1" else "Localhost Only"
st.markdown(f"""
<div class="link-box">
    <strong>Connection Info:</strong><br>
    {net_msg}<br>
    Access Link: <a href="http://{LOCAL_IP}:{ST_PORT}" target="_blank">http://{LOCAL_IP}:{ST_PORT}</a>
</div>
""", unsafe_allow_html=True)

# Chat Interface
if "messages" not in st.session_state: st.session_state.messages = []

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Sources and Retrieval"):
                for i, src_url in enumerate(msg["sources"]):
                    st.markdown(f"**{i+1}.** [{src_url}]({src_url})")

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if engine:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                data = engine.ask(prompt)
                st.markdown(data["answer"])
                
                sources_list = [d.metadata.get("source", "#") for d in data["sources"]]
                if sources_list:
                    with st.expander("Sources and Retrieval"):
                        for src in sources_list:
                            st.markdown(f"[{src}]({src})")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": data["answer"], 
                    "sources": sources_list
                })
    else:
        st.error("Engine failed to start.")