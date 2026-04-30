"""
config.py — Uygulama Yapılandırması
====================================
Tüm ortam değişkenlerini (.env dosyasından) yükler ve uygulama genelinde
kullanılan sabitleri tek bir noktada toplar.

Sorumlulukları:
  • GOOGLE_API_KEY gibi gizli anahtarları güvenli biçimde okumak
  • ChromaDB koleksiyon adı, chunk boyutu, örtüşme (overlap) gibi
    parametreleri merkezi olarak tanımlamak
  • Embedding ve LLM model adlarını belirlemek
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── .env dosyasını yükle ────────────────────────────────────────────
# Proje kök dizinindeki .env dosyasını arar
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

# ── Google API ──────────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# ── Embedding Ayarları ──────────────────────────────────────────────
EMBEDDING_MODEL: str = "models/gemini-embedding-001"
EMBEDDING_DIMENSION: int = 3072        # gemini-embedding-001 çıktı boyutu

# ── LLM (Gemini) Ayarları ──────────────────────────────────────────
LLM_MODEL: str = "gemini-1.5-flash"
LLM_TEMPERATURE: float = 0.3           # Düşük = daha tutarlı yanıtlar
LLM_MAX_OUTPUT_TOKENS: int = 2048

# ── ChromaDB Ayarları ──────────────────────────────────────────────
CHROMA_COLLECTION_NAME: str = "documents"
CHROMA_PERSIST_DIR: str = str(
    Path(__file__).resolve().parent.parent / "chroma_data"
)

# ── Doküman İşleme Ayarları ────────────────────────────────────────
CHUNK_SIZE: int = 2000          # ~500 token ≈ 2000 karakter
CHUNK_OVERLAP: int = 400        # %20 overlap (CHUNK_SIZE * 0.20)

# ── Sunucu Ayarları ────────────────────────────────────────────────
BACKEND_HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))

# ── Doğrulama ──────────────────────────────────────────────────────
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY ortam değişkeni tanımlı değil. "
        "Lütfen .env dosyanızı kontrol edin."
    )
