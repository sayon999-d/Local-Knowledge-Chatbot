import os
import logging
import time
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from pinecone import Pinecone, ServerlessSpec
from hf_embeddings import HuggingFaceAPIEmbeddings

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-chatbot")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
UPLOAD_BATCH_SIZE = int(os.getenv("UPLOAD_BATCH_SIZE", "40"))
EMBEDDING_WORKERS = int(os.getenv("EMBEDDING_WORKERS", "4"))
RETRY_DELAY_SECONDS = float(os.getenv("UPLOAD_RETRY_DELAY_SECONDS", "5"))

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

    # Microsoft DirectML
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

    # Developer Docs & Benchmarks
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


def main():
    if not HUGGINGFACE_API_KEY:
        raise ValueError("HUGGINGFACE_API_KEY not set in .env")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not set in .env")

    logger.info("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        logger.info(f"Creating index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384, 
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        logger.info("Index created. Waiting for it to be ready...")
    else:
        logger.info(f"Index '{PINECONE_INDEX_NAME}' already exists.")

    logger.info("Setting up embeddings...")
    embeddings = HuggingFaceAPIEmbeddings(
        api_key=HUGGINGFACE_API_KEY,
        model_name=EMBEDDING_MODEL,
        max_workers=EMBEDDING_WORKERS,
    )

    logger.info(f"Scraping {len(URL_LIST)} URLs...")
    loader = WebBaseLoader(URL_LIST)
    raw_docs = loader.load()
    logger.info(f"Loaded {len(raw_docs)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(raw_docs)
    logger.info(f"Split into {len(split_docs)} chunks.")

    logger.info("Uploading to Pinecone in batches...")
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )

    total = len(split_docs)
    uploaded = 0

    logger.info(
        "Using upload batch size %s with %s embedding workers.",
        UPLOAD_BATCH_SIZE,
        EMBEDDING_WORKERS,
    )

    for i in range(0, total, UPLOAD_BATCH_SIZE):
        batch = split_docs[i:i + UPLOAD_BATCH_SIZE]
        try:
            vectorstore.add_documents(
                batch,
                batch_size=UPLOAD_BATCH_SIZE,
                embedding_chunk_size=UPLOAD_BATCH_SIZE,
                async_req=True,
            )
            uploaded += len(batch)
            logger.info(f"  Uploaded {uploaded}/{total} chunks")
        except Exception as e:
            logger.error(f"  Error on batch {i}-{i+len(batch)}: {e}")
            logger.info("  Retrying after %.1fs...", RETRY_DELAY_SECONDS)
            time.sleep(RETRY_DELAY_SECONDS)
            try:
                vectorstore.add_documents(
                    batch,
                    batch_size=UPLOAD_BATCH_SIZE,
                    embedding_chunk_size=UPLOAD_BATCH_SIZE,
                    async_req=True,
                )
                uploaded += len(batch)
                logger.info(f"  Retry success. {uploaded}/{total} chunks")
            except Exception as e2:
                logger.error(f"  Skipping batch: {e2}")

    logger.info(f"Done! {uploaded}/{total} chunks uploaded to '{PINECONE_INDEX_NAME}'.")
    logger.info("Your Pinecone index is now populated. You can deploy the main app.")


if __name__ == "__main__":
    main()
