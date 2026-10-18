#!/bin/sh
set -e

# Railway dynamically provides $PORT. Fall back to 8501 if unset.
PORT="${PORT:-8501}"

echo "[INFO] Starting Streamlit server on 0.0.0.0:${PORT}..."

exec streamlit run app.py \
    --server.port="${PORT}" \
    --server.address="0.0.0.0" \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
