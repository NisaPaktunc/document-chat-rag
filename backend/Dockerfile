# ── Stage: Production ─────────────────────────────────────────────
FROM python:3.11-slim AS production

# Sistem bağımlılıkları (PyMuPDF ve ChromaDB için)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini
WORKDIR /app

# Bağımlılıkları önce kopyala (Docker cache optimizasyonu)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu kopyala
COPY . .

# ChromaDB kalıcı veri dizini
RUN mkdir -p /app/chroma_db
VOLUME /app/chroma_db

# Port
EXPOSE 8000

# Ortam değişkenleri
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Sağlık kontrolü
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Başlatma komutu
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
