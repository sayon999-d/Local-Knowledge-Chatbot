FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update --fix-missing && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install --default-timeout=1000 --no-cache-dir \
    torch \
    sentence-transformers \
    chromadb

RUN mkdir -p /app/local_embeddings
COPY local_models/all-MiniLM-L6-v2 /app/local_embeddings/all-MiniLM-L6-v2
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Create a non-root user
RUN useradd -m -u 1000 appuser

COPY --chown=appuser:appuser . .

# Set permissions for the working directory
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501"]
