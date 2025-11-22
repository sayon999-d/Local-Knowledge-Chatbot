
FROM python:3.9-slim


WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt


COPY main.py .


RUN mkdir -p /app/chroma_db_data /app/local_embeddings


EXPOSE 8080


CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.enableCORS=false", "--browser.serverAddress=0.0.0.0"]
