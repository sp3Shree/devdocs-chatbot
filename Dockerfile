# ===== Builder: install deps and cache wheels =====
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (faiss wheel needs libgomp; build tools for some libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Leverage Docker layer caching for dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ===== Runtime: small image with only what we need ===== \
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Runtime libs needed for faiss, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install wheels
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy source code
COPY app ./app

# Copy your data directory structure (kept small; mounted in compose for local dev)
# If you have prebuilt indexes you want baked in, keep this COPY; otherwise omit and rely on volume mount.
COPY data ./data

COPY app/ingest ./ingest

# (Optional) pre-warm the sentence-transformer to reduce cold start
# Comment this out if you prefer to download model at first request.
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
PY

# Expose API port
EXPOSE 8000

# Default command (Render and others set $PORT automatically; for local compse we map 8000:8000)
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]