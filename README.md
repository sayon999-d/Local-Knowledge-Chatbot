# Local Knowledge Chatbot

A powerful, locally-running RAG (Retrieval-Augmented Generation) chatbot system that allows you to build a knowledge base from web URLs and documents, then query it using local LLMs via Ollama.

## Features

- **Local-First Architecture**: Runs entirely on your machine with no external API dependencies
- **Web Scraping**: Automatically scrapes and indexes content from web URLs
- **File Upload**: Add knowledge by uploading text files
- **Vector Database**: Uses ChromaDB for efficient semantic search
- **Ollama Integration**: Leverages local LLMs (Mistral, Llama, etc.) via Ollama
- **Streamlit UI**: Beautiful, interactive web interface
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **Offline Capable**: Works without internet connection once models are downloaded

## Prerequisites

- Python 3.9 or higher
- [Ollama](https://ollama.ai/) installed and running
- Docker and Docker Compose (optional, for containerized deployment)

## Installation

### Option 1: Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sayon999-d/Local-Knowledge-Chatbot.git
   cd Local-Knowledge-Chatbot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Ollama:**
   ```bash
   # Install Ollama from https://ollama.ai/
   # Pull the required model
   ollama pull mistral:instruct
   ```

4. **Run the application:**
   ```bash
   streamlit run main.py
   ```

### Option 2: Docker Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sayon999-d/Local-Knowledge-Chatbot.git
   cd Local-Knowledge-Chatbot
   ```

2. **Start with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

3. **Access the application:**
   - Open your browser and navigate to `http://localhost:8080`

## Configuration

### Environment Variables

Create a `.env` file in the project root (optional):

```env
OLLAMA_BASE_URL=http://localhost:11434
```

For Docker, the default is `http://host.docker.internal:11434` to connect to Ollama running on the host machine.

### Model Configuration

The default configuration uses:
- **LLM Model**: `mistral:instruct` (via Ollama)
- **Embedding Model**: `mxbai-embed-large` (via Ollama)

You can modify these in `main.py`:
```python
OLLAMA_MODEL = "mistral:instruct"
EMBEDDING_MODEL = "mxbai-embed-large"
```

## Usage

### Adding Knowledge

1. **Via Web URL:**
   - Click on "Add Knowledge" in the sidebar
   - Select "Web URL"
   - Enter a URL and click "Scrape and Learn"

2. **Via File Upload:**
   - Click on "Add Knowledge" in the sidebar
   - Select "File Upload"
   - Upload a `.txt` file and click "Read and Learn"

### Querying the Knowledge Base

1. Type your question in the chat input at the bottom
2. The system will:
   - Search the vector database for relevant context
   - Generate an answer using the local LLM
   - Display sources used for the answer

## Architecture

```
┌─────────────────┐
│  Streamlit UI   │
└────────┬────────┘
         │
┌────────▼────────┐
│   RAG System    │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼──────┐
│ChromaDB│ │ Ollama  │
│Vector  │ │  LLM    │
│Store   │ │         │
└────────┘ └─────────┘
```

## Project Structure

```
Local-Knowledge-Chatbot/
├── main.py                 # Main application file
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── README.md             # This file
├── LICENSE               # MIT License
├── chroma_db_data/       # Vector database (created automatically)
├── local_embeddings/     # Local embedding models (optional)
└── saved_articles/       # Saved scraped content (created automatically)
```

## Customization

### Adding Default URLs

Edit the `URL_LIST` in `main.py` to include URLs that will be indexed on first run:

```python
URL_LIST = [
    "https://example.com/page1",
    "https://example.com/page2",
    # Add your URLs here
]
```

### Changing Port

For local installation:
```bash
streamlit run main.py --server.port 8501
```

For Docker, edit `docker-compose.yml`:
```yaml
ports:
  - "8080:8080"  # Change 8080 to your desired port
```

## Troubleshooting

### Ollama Connection Issues

- Ensure Ollama is running: `ollama serve`
- Check the `OLLAMA_BASE_URL` in your configuration
- For Docker, ensure `host.docker.internal` resolves correctly

### Model Not Found

- Pull the required model: `ollama pull mistral:instruct`
- Verify the model name matches in `main.py`

### Port Already in Use

- Change the port in `docker-compose.yml` or use a different port for Streamlit
- Check what's using the port: `lsof -i :8080`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.

## Acknowledgments

- [LangChain](https://www.langchain.com/) for the RAG framework
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Ollama](https://ollama.ai/) for local LLM inference
- [Streamlit](https://streamlit.io/) for the UI framework
