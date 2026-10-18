# Official lightweight Python 3.13 image
FROM python:3.13-slim

# System configuration: unbuffered stdout, no .pyc generation
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CHROMA_PERSIST_DIR=/app/data/.chroma_data

# Install build tools, libgomp1 (required by onnxruntime/chromadb), and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency requirements and install cleanly
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create volume mount point for persistent ChromaDB storage
RUN mkdir -p /app/data/.chroma_data

# Copy Streamlit theme configuration, core scripts, and entrypoint
COPY .streamlit/ .streamlit/
COPY agent.py rag_engine.py app.py entrypoint.sh ./
RUN chmod +x /app/entrypoint.sh

# Expose Streamlit default port
EXPOSE 8501

# Entrypoint script starts Streamlit bound to Railway's assigned $PORT
ENTRYPOINT ["/bin/sh", "/app/entrypoint.sh"]
