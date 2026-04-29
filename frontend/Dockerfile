# ── Stage: Production ─────────────────────────────────────────────
FROM python:3.11-slim AS production

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini
WORKDIR /app

# Frontend bağımlılıklarını kur
RUN pip install --no-cache-dir \
    gradio==5.25.2 \
    httpx==0.28.1

# Uygulama kodunu kopyala
COPY . .

# Port
EXPOSE 7860

# Ortam değişkenleri
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BACKEND_URL=http://backend:8000

# Sağlık kontrolü
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Başlatma komutu
CMD ["python", "app.py"]
