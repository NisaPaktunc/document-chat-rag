"""
main.py — FastAPI Ana Uygulama
================================
REST API endpoint'lerini tanımlar. Doküman yükleme, soru-cevap (RAG),
doküman özetleme, streaming ve sohbet işlevlerini HTTP üzerinden sunar.

Endpoint'ler:
  POST   /upload      → Birden fazla dosya yükleme (TXT, PDF, DOC, DOCX)
  POST   /ask         → Soru-cevap (RAG zinciri ile)
  POST   /chat        → Sohbet geçmişi ile soru-cevap
  POST   /summarize   → Doküman özetleme (belirli veya tümü)
  POST   /stream      → Streaming yanıt (SSE, token token)
  GET    /documents   → Yüklü doküman listesi
  DELETE /documents   → Tüm koleksiyonu silme
  GET    /health      → Sistem sağlık kontrolü

Genel özellikler:
  • CORS middleware (frontend erişimi)
  • Dosya boyutu limiti: 10 MB / dosya
  • Geçici dosya yönetimi (işlem sonrası otomatik silme)
  • Yapılandırılmış loglama
  • HTTPException ile ayrıntılı hata mesajları
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import BACKEND_HOST, BACKEND_PORT
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_chain import RAGChain

# ── Loglama ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ── Sabitler ────────────────────────────────────────────────────────
MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS: set = {".txt", ".pdf", ".doc", ".docx"}

# ── FastAPI Uygulaması ──────────────────────────────────────────────

app = FastAPI(
    title="Document Chat — RAG API",
    description=(
        "TXT, PDF, DOC ve DOCX dokümanlarınızı yükleyin ve "
        "içerikleri hakkında Türkçe sorular sorun. "
        "Yanıtlar Google Gemini tarafından, dokümanlarınızdaki "
        "bilgilere dayanarak üretilir."
    ),
    version="3.0.0",
)

# ── CORS Middleware ─────────────────────────────────────────────────
# Frontend (Gradio veya herhangi bir istemci) farklı porttan
# bağlanabilsin diye tüm origin'lere izin veriyoruz.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Uygulama Durumu (Singleton) ────────────────────────────────────
doc_processor = DocumentProcessor()
vector_store = VectorStore()
rag_chain = RAGChain(vector_store=vector_store)


# =====================================================================
# PYDANTIC MODELLERİ
# =====================================================================

class AskRequest(BaseModel):
    """POST /ask isteği için gövde modeli."""
    question: str = Field(..., min_length=1, description="Sorulacak soru")
    k: int = Field(default=5, ge=1, le=20, description="Getirilecek chunk sayısı")


class AskResponse(BaseModel):
    """POST /ask yanıt modeli."""
    answer: str
    sources: List[str]
    chunks_used: int


class ChatRequest(BaseModel):
    """POST /chat isteği için gövde modeli."""
    question: str = Field(..., min_length=1)
    history: List[dict] = Field(default_factory=list)
    k: int = Field(default=5, ge=1, le=20)


class ChatResponse(BaseModel):
    """POST /chat yanıt modeli."""
    answer: str
    sources: List[str]
    history: List[dict]


class SummarizeRequest(BaseModel):
    """POST /summarize isteği için gövde modeli."""
    document_name: Optional[str] = Field(
        default=None,
        description="Özetlenecek doküman adı. Boş bırakılırsa tüm dokümanlar özetlenir.",
    )


class UploadResponse(BaseModel):
    """POST /upload yanıt modeli."""
    status: str
    files_processed: int
    chunks_added: int


class HealthResponse(BaseModel):
    """GET /health yanıt modeli."""
    status: str
    total_documents: int
    total_chunks: int
    uptime_seconds: float


# ── Uygulama başlangıç zamanı (uptime hesabı için) ─────────────────
_start_time = time.time()


# =====================================================================
# YARDIMCI FONKSİYONLAR
# =====================================================================

def _validate_file(file: UploadFile) -> str:
    """
    Yüklenen dosyanın uzantısını ve boyutunu doğrular.

    Args:
        file: FastAPI UploadFile nesnesi.

    Returns:
        Dosya uzantısı (küçük harfle, ör. ".pdf").

    Raises:
        HTTPException 400: Uzantı desteklenmiyorsa.
        HTTPException 413: Dosya boyutu 10 MB'ı aşıyorsa.
    """
    # Uzantı kontrolü
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Desteklenmeyen dosya formatı: '{ext}'. "
                f"Kabul edilen formatlar: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    # Boyut kontrolü — dosyayı okuyup boyutunu kontrol et
    file.file.seek(0, 2)  # Dosya sonuna git
    file_size = file.file.tell()  # Mevcut konum = dosya boyutu
    file.file.seek(0)  # Başa geri dön

    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=(
                f"Dosya boyutu çok büyük: {size_mb:.1f} MB. "
                f"Maksimum dosya boyutu: {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
            ),
        )

    return ext


def _save_temp_file(file: UploadFile, ext: str) -> str:
    """
    UploadFile'ı geçici bir dosyaya kaydeder.

    Returns:
        Geçici dosyanın tam yolu.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        return tmp.name


def _cleanup_temp_files(paths: List[str]) -> None:
    """Geçici dosyaları güvenli biçimde siler."""
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.unlink(path)
        except OSError as e:
            logger.warning("Geçici dosya silinemedi: %s — %s", path, e)


def _get_chunks_for_document(document_name: str) -> List[dict]:
    """
    Belirli bir dokümanın chunk'larını ChromaDB'den getirir.

    Args:
        document_name: Filtrelenecek doküman adı.

    Returns:
        {"text": ..., "metadata": ...} formatında dict listesi.
    """
    all_data = vector_store.collection.get(
        where={"source": document_name},
        include=["documents", "metadatas"],
    )

    chunks = []
    for doc, meta in zip(
        all_data.get("documents", []),
        all_data.get("metadatas", []),
    ):
        chunks.append({"text": doc, "metadata": meta or {}})

    return chunks


def _get_all_chunks() -> List[dict]:
    """
    Tüm chunk'ları ChromaDB'den getirir.

    Returns:
        {"text": ..., "metadata": ...} formatında dict listesi.
    """
    all_data = vector_store.collection.get(
        include=["documents", "metadatas"],
    )

    chunks = []
    for doc, meta in zip(
        all_data.get("documents", []),
        all_data.get("metadatas", []),
    ):
        chunks.append({"text": doc, "metadata": meta or {}})

    return chunks


# =====================================================================
# ENDPOINT'LER
# =====================================================================

# ── 1) POST /upload — Birden Fazla Dosya Yükleme ───────────────────

@app.post(
    "/upload",
    response_model=UploadResponse,
    tags=["Doküman"],
    summary="Birden fazla dosya yükle",
)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Bir veya birden fazla dosyayı yükler, işler ve vektör veritabanına kaydeder.

    **Desteklenen formatlar:** .txt, .pdf, .doc, .docx
    **Maksimum dosya boyutu:** 10 MB / dosya

    Her dosya şu adımlardan geçer:
    1. Uzantı ve boyut doğrulaması
    2. Geçici dosyaya kaydetme
    3. DocumentProcessor ile metin çıkarma ve chunk'lama
    4. VectorStore ile embedding üretme ve ChromaDB'ye kaydetme
    5. Geçici dosyanın silinmesi
    """
    if not files:
        raise HTTPException(status_code=400, detail="En az bir dosya yüklemelisiniz.")

    logger.info("Dosya yükleme başlatıldı: %d dosya", len(files))

    temp_paths: List[str] = []
    total_chunks_added = 0
    files_processed = 0
    errors: List[str] = []

    try:
        for file in files:
            filename = file.filename or "bilinmeyen_dosya"

            # ── Doğrulama ──
            try:
                ext = _validate_file(file)
            except HTTPException as e:
                errors.append(f"'{filename}': {e.detail}")
                logger.warning("Dosya reddedildi — %s: %s", filename, e.detail)
                continue

            # ── Geçici dosyaya kaydet ──
            try:
                tmp_path = _save_temp_file(file, ext)
                temp_paths.append(tmp_path)
            except Exception as e:
                errors.append(f"'{filename}': Dosya kaydedilemedi — {e}")
                logger.error("Dosya kayıt hatası — %s: %s", filename, e)
                continue

            # ── Dokümanı işle ──
            try:
                chunks = doc_processor.process_file(tmp_path)

                # source metadata'sını orijinal dosya adıyla güncelle
                for chunk in chunks:
                    chunk["source"] = filename

                added = vector_store.add_documents(chunks)
                total_chunks_added += added
                files_processed += 1

                logger.info(
                    "✅ '%s' işlendi: %d chunk eklendi", filename, added
                )
            except Exception as e:
                errors.append(f"'{filename}': İşleme hatası — {e}")
                logger.error("İşleme hatası — %s: %s", filename, e, exc_info=True)

    finally:
        # ── Geçici dosyaları temizle ──
        _cleanup_temp_files(temp_paths)

    # ── Sonuç ──
    if files_processed == 0 and errors:
        raise HTTPException(
            status_code=400,
            detail=f"Hiçbir dosya işlenemedi. Hatalar: {'; '.join(errors)}",
        )

    if errors:
        logger.warning(
            "Kısmi başarı: %d/%d dosya işlendi. Hatalar: %s",
            files_processed, len(files), "; ".join(errors),
        )

    logger.info(
        "Yükleme tamamlandı: %d dosya, %d chunk eklendi",
        files_processed, total_chunks_added,
    )

    return UploadResponse(
        status="success",
        files_processed=files_processed,
        chunks_added=total_chunks_added,
    )


# ── 2) POST /ask — Soru-Cevap (RAG) ───────────────────────────────

@app.post(
    "/ask",
    response_model=AskResponse,
    tags=["RAG"],
    summary="Soru sor (RAG)",
)
async def ask_question(request: AskRequest):
    """
    Kullanıcı sorusunu RAG zinciri ile yanıtlar.

    1. Vektör veritabanında semantik arama yapılır (k adet chunk).
    2. Bulunan chunk'lar bağlam olarak Gemini'ye gönderilir.
    3. Yanıt ve kaynak referansları döndürülür.
    """
    if vector_store.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="Henüz hiç doküman yüklenmemiş. Önce /upload ile doküman yükleyin.",
        )

    logger.info("Soru alındı: '%s' (k=%d)", request.question[:80], request.k)

    try:
        # Vektör arama
        retrieved_chunks = vector_store.similarity_search(
            query=request.question,
            k=request.k,
        )

        # RAG ile yanıt üret
        result = rag_chain.generate_answer(
            question=request.question,
            retrieved_chunks=retrieved_chunks,
        )

        return AskResponse(
            answer=result["answer"],
            sources=result["sources"],
            chunks_used=len(retrieved_chunks),
        )
    except Exception as e:
        logger.error("Soru yanıtlama hatası: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Yanıt üretilemedi: {e}")


# ── 3) POST /chat — Sohbet Geçmişi ile Soru-Cevap ─────────────────

@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["RAG"],
    summary="Sohbet geçmişi ile soru sor",
)
async def chat_with_history(request: ChatRequest):
    """
    Sohbet geçmişini dikkate alarak soruyu yanıtlar.
    Önceki sorularla tutarlı cevap üretir.
    """
    if vector_store.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="Henüz hiç doküman yüklenmemiş. Önce /upload ile doküman yükleyin.",
        )

    logger.info(
        "Sohbet sorusu: '%s' (geçmiş: %d mesaj)",
        request.question[:80], len(request.history),
    )

    try:
        chunks = vector_store.similarity_search(
            query=request.question,
            k=request.k,
        )
        result = rag_chain.chat_with_history(
            question=request.question,
            history=request.history,
            retrieved_chunks=chunks,
        )
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            history=result["history"],
        )
    except Exception as e:
        logger.error("Sohbet hatası: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Yanıt üretilemedi: {e}")


# ── 4) POST /summarize — Doküman Özetleme ──────────────────────────

@app.post(
    "/summarize",
    tags=["RAG"],
    summary="Doküman özetle",
)
async def summarize_document(request: SummarizeRequest):
    """
    Belirtilen dokümanı veya tüm dokümanları özetler.

    - `document_name` verilirse yalnızca o doküman özetlenir.
    - `document_name` boş bırakılırsa tüm yüklü dokümanlar özetlenir.
    """
    if vector_store.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="Henüz hiç doküman yüklenmemiş. Önce /upload ile doküman yükleyin.",
        )

    try:
        if request.document_name:
            # ── Belirli dokümanı özetle ──
            logger.info("Doküman özetleniyor: '%s'", request.document_name)

            # Dokümanın var olup olmadığını kontrol et
            available_docs = vector_store.get_document_list()
            if request.document_name not in available_docs:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"'{request.document_name}' adlı doküman bulunamadı. "
                        f"Yüklü dokümanlar: {', '.join(available_docs)}"
                    ),
                )

            chunks = _get_chunks_for_document(request.document_name)
        else:
            # ── Tüm dokümanları özetle ──
            logger.info("Tüm dokümanlar özetleniyor")
            chunks = _get_all_chunks()

        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="Özetlenecek chunk bulunamadı.",
            )

        summary = rag_chain.summarize_document(chunks)
        return {"summary": summary}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Özetleme hatası: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Özet oluşturulamadı: {e}")


# ── 5) POST /stream — Streaming Yanıt ──────────────────────────────

@app.post(
    "/stream",
    tags=["RAG"],
    summary="Streaming yanıt",
)
async def stream_answer(request: AskRequest):
    """
    Soruyu yanıtlar ve yanıtı token token döndürür (Server-Sent Events).

    Bu endpoint özellikle frontend'de gerçek zamanlı metin gösterimi
    için kullanılır.
    """
    if vector_store.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="Henüz hiç doküman yüklenmemiş.",
        )

    logger.info("Streaming yanıt başlıyor: '%s'", request.question[:80])

    chunks = vector_store.similarity_search(
        query=request.question,
        k=request.k,
    )

    def generate():
        for token in rag_chain.stream_answer(request.question, chunks):
            yield token

    return StreamingResponse(generate(), media_type="text/plain")


# ── 6) GET /documents — Yüklü Doküman Listesi ─────────────────────

@app.get(
    "/documents",
    tags=["Doküman"],
    summary="Yüklü doküman listesi",
)
async def list_documents():
    """
    Vektör veritabanına yüklenmiş dokümanların benzersiz adlarını döndürür.
    """
    documents = vector_store.get_document_list()

    logger.info("Doküman listesi sorgulandı: %d doküman", len(documents))

    return {
        "documents": documents,
        "total_documents": len(documents),
        "total_chunks": vector_store.count(),
    }


# ── 7) DELETE /documents — Koleksiyonu Temizleme ──────────────────

@app.delete(
    "/documents",
    tags=["Doküman"],
    summary="Tüm dokümanları sil",
)
async def delete_all_documents():
    """
    Vektör veritabanındaki tüm dokümanları ve chunk'ları kalıcı olarak siler.

    ⚠️ Bu işlem geri alınamaz!
    """
    chunk_count = vector_store.count()
    doc_count = len(vector_store.get_document_list())

    vector_store.delete_collection()

    logger.warning(
        "🗑️ Koleksiyon silindi: %d doküman, %d chunk",
        doc_count, chunk_count,
    )

    return {
        "status": "success",
        "message": "Tüm dokümanlar başarıyla silindi.",
        "deleted_documents": doc_count,
        "deleted_chunks": chunk_count,
    }


# ── 8) GET /health — Sistem Sağlık Kontrolü ────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Sistem"],
    summary="Sağlık kontrolü",
)
async def health_check():
    """
    API'nin çalışır durumda olduğunu, veritabanı bağlantısının
    sağlıklı olduğunu ve temel metrikleri döndürür.
    """
    uptime = time.time() - _start_time

    try:
        total_chunks = vector_store.count()
        total_documents = len(vector_store.get_document_list())
        status = "healthy"
    except Exception as e:
        logger.error("Sağlık kontrolü başarısız: %s", e)
        status = "unhealthy"
        total_chunks = 0
        total_documents = 0

    return HealthResponse(
        status=status,
        total_documents=total_documents,
        total_chunks=total_chunks,
        uptime_seconds=round(uptime, 2),
    )


# =====================================================================
# UYGULAMA BAŞLATMA
# =====================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info(
        "🚀 Document Chat API başlatılıyor: http://%s:%d",
        BACKEND_HOST, BACKEND_PORT,
    )

    uvicorn.run(
        "main:app",
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        reload=True,
    )
