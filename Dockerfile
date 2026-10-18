# Official lightweight Python 3.13 image
FROM python:3.13-slim

# System configuration: unbuffered stdout, no .pyc generation
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501 \
    CHROMA_PERSIST_DIR=/app/data/.chroma_data

# Install necessary build tools and curl for container health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency requirements and install cleanly
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create volume mount point for persistent ChromaDB storage
RUN mkdir -p /app/data/.chroma_data

# Copy Streamlit theme configuration and core application scripts
COPY .streamlit/ .streamlit/
COPY agent.py rag_engine.py app.py ./

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Start Streamlit dynamically bound to Railway's assigned $PORT
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false"]
