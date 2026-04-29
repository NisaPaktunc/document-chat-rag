"""
vector_store.py — Vektör Veritabanı Yönetimi (ChromaDB + Google Embedding)
===========================================================================
Google text-embedding-004 modeli ile metin embedding'leri üretir ve
ChromaDB'de kalıcı olarak saklar / sorgular.

Sorumlulukları:
  • Google Generative AI SDK üzerinden embedding vektörleri oluşturmak
  • Rate limiting hatalarında exponential backoff ile yeniden denemek
  • ChromaDB koleksiyonunu başlatmak ve kalıcı dizine yazmak
  • Doküman chunk'larını embedding'leri ve metadata'ları ile kaydetmek
  • Kullanıcı sorgusuna en benzer chunk'ları kosinüs benzerliği ile bulmak
  • Koleksiyon temizleme ve yüklü doküman listeleme

Sınıf: VectorStore
  __init__           → ChromaDB bağlantısı ve koleksiyon oluşturma
  add_documents      → Chunk'ları embed edip kaydetme
  similarity_search  → Semantik arama (sorgu → en benzer k chunk)
  delete_collection  → Koleksiyonu silme (yeni yükleme öncesi)
  get_document_list  → Yüklü benzersiz doküman adlarını listeleme
  count              → Toplam chunk sayısı
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Dict, List, Optional

import chromadb
import google.generativeai as genai

from config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
)

# ── Logger ──────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Google API Yapılandırması ───────────────────────────────────────
genai.configure(api_key=GOOGLE_API_KEY)

# ── Sabitler ────────────────────────────────────────────────────────
_MAX_RETRIES: int = 5                # Maksimum yeniden deneme sayısı
_BASE_DELAY: float = 1.0            # İlk bekleme süresi (saniye)
_MAX_DELAY: float = 60.0            # Maksimum bekleme süresi (saniye)
_EMBEDDING_BATCH_SIZE: int = 100    # Google API'ye tek seferde gönderilecek metin sayısı


# =====================================================================
# EMBEDDING FONKSİYONLARI (Retry Mekanizmalı)
# =====================================================================

def _embed_with_retry(
    content,
    task_type: str = "retrieval_document",
    max_retries: int = _MAX_RETRIES,
    base_delay: float = _BASE_DELAY,
) -> list:
    """
    Google Embedding API'sine istek gönderir; hata durumunda
    exponential backoff ile yeniden dener.

    Exponential backoff stratejisi:
      Deneme 1 → 1s bekle
      Deneme 2 → 2s bekle
      Deneme 3 → 4s bekle
      Deneme 4 → 8s bekle
      Deneme 5 → 16s bekle (maks. 60s ile sınırlı)

    Args:
        content:     Embed edilecek metin veya metin listesi.
        task_type:   "retrieval_document" (doküman ekleme) veya
                     "retrieval_query" (arama sorgusu).
        max_retries: Maksimum yeniden deneme sayısı.
        base_delay:  İlk bekleme süresi (saniye).

    Returns:
        Tek metin için: List[float] (tek vektör)
        Metin listesi için: List[List[float]] (vektör listesi)

    Raises:
        Exception: Tüm denemeler tükendikten sonra son hatayı fırlatır.
    """
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=content,
                task_type=task_type,
            )
            return result["embedding"]

        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()

            # Rate limit veya geçici sunucu hatalarında yeniden dene
            is_retryable = any(keyword in error_msg for keyword in (
                "429", "rate", "limit", "quota", "resource_exhausted",
                "503", "unavailable", "deadline", "timeout", "500",
                "internal",
            ))

            if not is_retryable:
                # Kalıcı hata — yeniden deneme anlamsız
                logger.error(
                    "Embedding hatası (yeniden denenMEyecek): %s", e
                )
                raise

            # Exponential backoff: delay = base * 2^(attempt-1)
            delay = min(base_delay * (2 ** (attempt - 1)), _MAX_DELAY)
            logger.warning(
                "Embedding API hatası (deneme %d/%d). "
                "%.1f saniye sonra yeniden denenecek. Hata: %s",
                attempt, max_retries, delay, e,
            )
            time.sleep(delay)

    # Tüm denemeler tükendi
    logger.error(
        "Embedding API %d denemeden sonra başarısız oldu. Son hata: %s",
        max_retries, last_exception,
    )
    raise last_exception  # type: ignore[misc]


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Metin listesi için embedding vektörleri üretir (doküman ekleme amaçlı).

    Büyük listeleri _EMBEDDING_BATCH_SIZE'lık gruplara bölerek
    API'ye gönderir. Her grup için retry mekanizması aktiftir.

    Args:
        texts: Embedding'i hesaplanacak metin listesi.

    Returns:
        Her metin için float vektörlerin listesi.
        Sıralama girdi listesiyle birebir eşleşir.
    """
    if not texts:
        return []

    all_embeddings: List[List[float]] = []

    # Büyük listeleri batch'lere böl
    for i in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + _EMBEDDING_BATCH_SIZE]
        logger.debug(
            "Embedding batch %d-%d / %d işleniyor…",
            i + 1, min(i + _EMBEDDING_BATCH_SIZE, len(texts)), len(texts),
        )
        batch_embeddings = _embed_with_retry(
            content=batch,
            task_type="retrieval_document",
        )
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def get_query_embedding(query: str) -> List[float]:
    """
    Tek bir sorgu metni için embedding vektörü üretir.

    task_type olarak 'retrieval_query' kullanır; bu, Google'ın
    embedding modelinde arama sorgularına özel optimize edilmiş
    vektör üretimini aktifleştirir.

    Args:
        query: Embedding'i hesaplanacak sorgu metni.

    Returns:
        Float vektör (768 boyutlu — text-embedding-004 için).
    """
    return _embed_with_retry(
        content=query,
        task_type="retrieval_query",
    )


# =====================================================================
# VectorStore SINIFI
# =====================================================================

class VectorStore:
    """
    ChromaDB üzerinde doküman ekleme, semantik arama ve koleksiyon
    yönetimi işlemlerini saran sınıf.

    ChromaDB, PersistentClient ile çalışır ve veriler persist_dir
    altında disk üzerinde kalıcı olarak saklanır. Uygulama yeniden
    başlatıldığında mevcut veriler korunur.

    Attributes:
        client:     ChromaDB PersistentClient örneği.
        collection: Aktif ChromaDB koleksiyonu.

    Kullanım:
        >>> store = VectorStore(persist_dir="./chroma_db")
        >>> store.add_documents(chunks)
        >>> results = store.similarity_search("yapay zeka nedir?", k=3)
        >>> for r in results:
        ...     print(r["text"][:80], r["similarity_score"])
    """

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR) -> None:
        """
        ChromaDB bağlantısını kurar ve koleksiyonu oluşturur/açar.

        PersistentClient kullanılır — veriler persist_dir altında disk
        üzerinde kalıcı olarak saklanır. Uygulama kapansa bile veriler
        kaybolmaz.

        Koleksiyon kosinüs benzerliği (cosine similarity) ile yapılandırılır.
        Kosinüs mesafesi, metin embedding'leri için en yaygın benzerlik
        metriğidir; vektör büyüklüğünden bağımsız olarak yönsel benzerliği
        ölçer.

        Args:
            persist_dir: ChromaDB verilerinin yazılacağı dizin yolu.
                         Varsayılan: config.py'deki CHROMA_PERSIST_DIR.

        Örnekler:
            >>> store = VectorStore()                     # varsayılan dizin
            >>> store = VectorStore("./my_chroma_data")   # özel dizin
        """
        logger.info("ChromaDB başlatılıyor: %s", persist_dir)

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "ChromaDB hazır. Koleksiyon: '%s', mevcut chunk sayısı: %d",
            CHROMA_COLLECTION_NAME, self.collection.count(),
        )

    # ================================================================
    # 1) DOKÜMAN EKLEME
    # ================================================================

    def add_documents(self, chunks: List[Dict]) -> int:
        """
        DocumentProcessor'dan gelen chunk listesini embed eder ve
        ChromaDB'ye kaydeder.

        Her chunk dict'i en az şu anahtarları içermelidir:
          - "text" (str): Chunk'ın metin içeriği (zorunlu)

        Ek metadata alanları (varsa) otomatik olarak ChromaDB'ye
        kaydedilir:
          - "source"   (str): Kaynak dosya adı (ör. "rapor.pdf")
          - "page"     (int): Sayfa numarası (yalnızca PDF'ler için)
          - "chunk_id" (str): Benzersiz tanımlayıcı (yoksa UUID üretilir)

        Büyük chunk listeleri batch'lere bölünerek işlenir. Her batch
        için rate limiting koruması aktiftir.

        Args:
            chunks: DocumentProcessor.process_file() çıktısı formatında
                    dict listesi.

        Returns:
            Başarıyla eklenen chunk sayısı.

        Raises:
            ValueError: chunks listesi boşsa veya "text" anahtarı eksikse.
            Exception:  Embedding API tüm retry'lardan sonra başarısız olursa.

        Örnekler:
            >>> store = VectorStore()
            >>> chunks = [
            ...     {"text": "Yapay zeka...", "source": "rapor.pdf", "page": 1},
            ...     {"text": "Derin öğrenme...", "source": "rapor.pdf", "page": 2},
            ... ]
            >>> added = store.add_documents(chunks)
            >>> print(f"{added} chunk eklendi")
        """
        if not chunks:
            logger.warning("add_documents: Boş chunk listesi, işlem atlandı.")
            return 0

        # ── Verileri hazırla ──
        texts: List[str] = []
        ids: List[str] = []
        metadatas: List[Dict] = []

        for chunk in chunks:
            text = chunk.get("text", "")
            if not text or not text.strip():
                continue  # Boş chunk'ları atla

            texts.append(text)
            ids.append(chunk.get("chunk_id", str(uuid.uuid4())))

            # text ve chunk_id dışındaki tüm alanlar metadata olur
            meta: Dict = {}
            for key, value in chunk.items():
                if key in ("text", "chunk_id"):
                    continue
                # ChromaDB metadata'da yalnızca str, int, float, bool kabul eder
                if isinstance(value, (str, int, float, bool)):
                    meta[key] = value
                else:
                    meta[key] = str(value)
            metadatas.append(meta)

        if not texts:
            logger.warning("add_documents: Geçerli metin içeren chunk yok.")
            return 0

        # ── Embedding'leri üret (batch + retry) ──
        logger.info("%d chunk için embedding üretiliyor…", len(texts))
        embeddings = get_embeddings(texts)

        # ── ChromaDB'ye kaydet ──
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            "%d chunk başarıyla ChromaDB'ye eklendi. Toplam: %d",
            len(texts), self.collection.count(),
        )
        return len(texts)

    # ================================================================
    # 2) SEMANTİK ARAMA
    # ================================================================

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Kullanıcı sorgusunu embed eder ve ChromaDB'de kosinüs benzerliği
        ile en yakın k chunk'ı döndürür.

        ChromaDB kosinüs mesafesi döndürür (0 = tamamen benzer, 2 = tamamen
        zıt). Bu metot mesafeyi benzerlik skoruna çevirir:
            similarity_score = 1 - (distance / 2)
        Böylece 1.0 = mükemmel eşleşme, 0.0 = hiç ilgisiz.

        Args:
            query: Kullanıcının sorduğu soru / arama metni.
            k:     Döndürülecek maksimum sonuç sayısı (varsayılan: 5).
            where: Opsiyonel metadata filtresi.
                   Örnek: {"source": "rapor.pdf"}
                   Örnek: {"page": 3}

        Returns:
            Her biri şu anahtarları içeren dict listesi (benzerlik skoruna
            göre azalan sırada):
              - "text"             (str):   Chunk'ın metin içeriği
              - "metadata"         (dict):  source, page vb. bilgiler
              - "similarity_score" (float): 0.0 – 1.0 arası benzerlik skoru

            Sonuç bulunamazsa boş liste döner.

        Raises:
            Exception: Embedding API tüm retry'lardan sonra başarısız olursa.

        Örnekler:
            >>> store = VectorStore()
            >>> results = store.similarity_search("yapay zeka nedir?", k=3)
            >>> for r in results:
            ...     print(f"[{r['similarity_score']:.2f}] {r['text'][:60]}…")
            ...     print(f"  Kaynak: {r['metadata'].get('source')}")
        """
        if self.collection.count() == 0:
            logger.warning("similarity_search: Koleksiyon boş, sonuç yok.")
            return []

        # ── Sorguyu embed et ──
        logger.debug("Sorgu embed ediliyor: '%s'", query[:80])
        query_embedding = get_query_embedding(query)

        # ── ChromaDB'de ara ──
        search_params: Dict = {
            "query_embeddings": [query_embedding],
            "n_results": min(k, self.collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            search_params["where"] = where

        results = self.collection.query(**search_params)

        # ── Sonuçları yapılandır ──
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        formatted_results: List[Dict] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            # Kosinüs mesafesini benzerlik skoruna çevir
            # ChromaDB cosine distance: 0 (aynı) → 2 (zıt)
            # Benzerlik skoru: 1.0 (aynı) → 0.0 (zıt)
            similarity_score = 1.0 - (distance / 2.0)

            formatted_results.append({
                "text": text,
                "metadata": metadata or {},
                "similarity_score": round(similarity_score, 4),
            })

        logger.info(
            "Arama tamamlandı: '%s' → %d sonuç bulundu",
            query[:50], len(formatted_results),
        )
        return formatted_results

    # ================================================================
    # 3) KOLEKSİYON SİLME
    # ================================================================

    def delete_collection(self) -> None:
        """
        Mevcut koleksiyonu tamamen siler ve aynı isimle yeniden oluşturur.

        Bu metot genellikle yeni bir doküman seti yüklemeden önce
        çağrılır — eski verilerle karışmaması için veritabanını
        temizler.

        Silme işlemi geri alınamaz. Tüm chunk'lar, embedding'ler ve
        metadata'lar kalıcı olarak silinir.

        Örnekler:
            >>> store = VectorStore()
            >>> print(f"Silme öncesi: {store.count()} chunk")
            >>> store.delete_collection()
            >>> print(f"Silme sonrası: {store.count()} chunk")  # → 0
        """
        logger.warning(
            "Koleksiyon siliniyor: '%s' (%d chunk)",
            CHROMA_COLLECTION_NAME, self.collection.count(),
        )

        self.client.delete_collection(CHROMA_COLLECTION_NAME)

        # Aynı isim ve ayarlarla yeniden oluştur
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info("Koleksiyon sıfırlandı: '%s'", CHROMA_COLLECTION_NAME)

    # ================================================================
    # 4) YÜKLÜ DOKÜMAN LİSTESİ
    # ================================================================

    def get_document_list(self) -> List[str]:
        """
        ChromaDB'de kayıtlı tüm benzersiz doküman adlarını döndürür.

        Metadata'daki "source" alanından benzersiz dosya adlarını
        toplar ve alfabetik sırayla döndürür.

        Returns:
            Benzersiz doküman adlarının sıralı listesi.
            Hiç doküman yoksa boş liste döner.

        Örnekler:
            >>> store = VectorStore()
            >>> docs = store.get_document_list()
            >>> print(docs)
            ['analiz.docx', 'rapor.pdf', 'notlar.txt']
        """
        total = self.collection.count()
        if total == 0:
            return []

        # Tüm metadata'ları çek (yalnızca "metadatas" alanı yeterli)
        all_data = self.collection.get(include=["metadatas"])
        metadatas = all_data.get("metadatas", [])

        # "source" alanlarından benzersiz isimleri topla
        sources: set = set()
        for meta in metadatas:
            if meta and "source" in meta:
                sources.add(meta["source"])

        sorted_sources = sorted(sources)
        logger.debug(
            "Yüklü dokümanlar (%d adet): %s",
            len(sorted_sources), sorted_sources,
        )
        return sorted_sources

    # ================================================================
    # 5) YARDIMCI METODLAR
    # ================================================================

    def count(self) -> int:
        """
        Koleksiyondaki toplam chunk sayısını döndürür.

        Returns:
            Kayıtlı chunk sayısı (int).

        Örnekler:
            >>> store = VectorStore()
            >>> print(f"Toplam: {store.count()} chunk")
        """
        return self.collection.count()
