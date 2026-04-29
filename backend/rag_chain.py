"""
rag_chain.py — RAG Zinciri (Retrieval-Augmented Generation)
============================================================
Kullanıcı sorusunu alır, vektör veritabanından çekilen bağlam parçalarını
kullanarak Google Gemini 1.5 Flash modeli ile yanıt üretir.

Sınıf: RAGChain
  Dört temel işlev sunar:
    1. generate_answer      → Tek soru-cevap (kaynak gösterimli)
    2. summarize_document   → Doküman özetleme (madde madde)
    3. stream_answer        → Streaming yanıt (token token yield)
    4. chat_with_history    → Sohbet geçmişi ile tutarlı yanıt

Kaynak Gösterme (Source Citation):
  Her yanıtta kullanılan chunk'ların kaynak dosya adı ve sayfa numarası
  otomatik olarak eklenir. Gemini'ye verilen system prompt da modelden
  cevap içinde kaynakları belirtmesini ister.
"""

from __future__ import annotations

import logging
from typing import Dict, Generator, List, Optional

import google.generativeai as genai

from config import (
    GOOGLE_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_OUTPUT_TOKENS,
)
from vector_store import VectorStore

# ── Logger ──────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Google API Yapılandırması ───────────────────────────────────────
genai.configure(api_key=GOOGLE_API_KEY)


# =====================================================================
# PROMPT ŞABLONLARI
# =====================================================================

SYSTEM_PROMPT = """\
Sen yardımcı bir asistansın. Yalnızca sağlanan bağlam içeriğini kullanarak \
soruları yanıtla. Eğer bağlamda cevap yoksa 'Bu bilgi yüklenen dokümanlarda \
bulunamadı' de. Cevabında hangi dokümandan bilgi aldığını belirt.

Kurallar:
1. Yalnızca verilen bağlam bilgilerini kullan, dışarıdan bilgi ekleme.
2. Bağlamda yanıt bulamazsan bunu açıkça ve dürüstçe belirt.
3. Yanıtlarını her zaman Türkçe ver.
4. Cevabının sonunda kullandığın kaynak belge adını ve sayfa numarasını belirt.
5. Birden fazla kaynaktan bilgi kullandıysan hepsini listele.
"""

QA_PROMPT_TEMPLATE = """\
Aşağıdaki bağlam bilgilerini kullanarak soruyu yanıtla.

Bağlam:
{context}

---

Soru: {question}
"""

SUMMARY_PROMPT_TEMPLATE = """\
Aşağıdaki doküman içeriğini Türkçe olarak özetle.

Özet kuralları:
1. Madde madde (bullet point) formatında yaz.
2. Ana başlıkları ve alt konuları belirt.
3. En önemli bilgileri öne çıkar.
4. Özet, orijinal içeriğin %20-30'u kadar uzunlukta olsun.
5. Hangi kaynak dosyadan alındığını belirt.

Doküman İçeriği:
{content}
"""

CHAT_PROMPT_TEMPLATE = """\
Aşağıdaki bağlam bilgilerini ve konuşma geçmişini kullanarak soruyu yanıtla.
Önceki sorularla tutarlı ol.

Bağlam:
{context}

---

Konuşma Geçmişi:
{history}

---

Son Soru: {question}
"""


# =====================================================================
# RAGChain SINIFI
# =====================================================================

class RAGChain:
    """
    Retrieval-Augmented Generation zincirini yöneten ana sınıf.

    Google Gemini 1.5 Flash modelini kullanarak, VectorStore'dan
    getirilen bağlam chunk'ları üzerinden yanıt üretir.

    Dört temel operasyon:
      • generate_answer:    Tek soru → bağlam → yanıt + kaynaklar
      • summarize_document: Chunk listesi → madde madde özet
      • stream_answer:      Tek soru → bağlam → streaming yanıt (generator)
      • chat_with_history:  Sohbet geçmişi + soru → tutarlı yanıt

    Attributes:
        vector_store: VectorStore örneği (similarity_search için).
        model:        Gemini GenerativeModel örneği.

    Kullanım:
        >>> chain = RAGChain(vector_store=my_store)
        >>> result = chain.generate_answer("Yapay zeka nedir?", chunks)
        >>> print(result["answer"])
    """

    def __init__(self, vector_store: Optional[VectorStore] = None) -> None:
        """
        RAGChain'i başlatır.

        Args:
            vector_store: Kullanılacak VectorStore örneği.
                          None ise varsayılan ayarlarla yeni bir tane oluşturulur.
        """
        self.vector_store = vector_store or VectorStore()

        self.model = genai.GenerativeModel(
            model_name=LLM_MODEL,
            generation_config=genai.GenerationConfig(
                temperature=LLM_TEMPERATURE,
                max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
            ),
            system_instruction=SYSTEM_PROMPT,
        )

        logger.info("RAGChain başlatıldı. Model: %s", LLM_MODEL)

    # ================================================================
    # YARDIMCI METODLAR
    # ================================================================

    @staticmethod
    def _build_context(retrieved_chunks: List[Dict]) -> str:
        """
        Retrieve edilen chunk listesinden numaralı bağlam metni oluşturur.

        Her chunk'ın kaynak bilgisi (dosya adı + sayfa) başlık olarak
        eklenir. Bu sayede hem Gemini kaynağı görebilir hem de
        kullanıcıya referans verilebilir.

        Args:
            retrieved_chunks: VectorStore.similarity_search() çıktısı
                              formatında dict listesi. Her dict:
                              {"text": ..., "metadata": {...}, "similarity_score": ...}

        Returns:
            Numaralı, kaynakları işaretlenmiş bağlam metni.
        """
        if not retrieved_chunks:
            return ""

        parts: List[str] = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            meta = chunk.get("metadata", {})
            source = meta.get("source", "Bilinmiyor")
            page = meta.get("page")

            header = f"[Kaynak {i}: {source}"
            if page:
                header += f", Sayfa {page}"
            header += "]"

            parts.append(f"{header}\n{chunk['text']}")

        return "\n\n".join(parts)

    @staticmethod
    def _extract_sources(retrieved_chunks: List[Dict]) -> List[str]:
        """
        Chunk listesinden benzersiz kaynak referanslarını çıkarır.

        Aynı kaynak+sayfa kombinasyonu yalnızca bir kez eklenir.
        Çıktı formatı: "dosya_adi.pdf (Sayfa 3)" veya "dosya_adi.docx"

        Args:
            retrieved_chunks: Retrieve edilmiş chunk listesi.

        Returns:
            Benzersiz kaynak referans string'lerinin listesi.
        """
        sources: List[str] = []
        seen: set = set()

        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {})
            source = meta.get("source", "Bilinmiyor")
            page = meta.get("page")

            if page:
                ref = f"{source} (Sayfa {page})"
            else:
                ref = source

            if ref not in seen:
                seen.add(ref)
                sources.append(ref)

        return sources

    @staticmethod
    def _format_history(history: List[Dict]) -> str:
        """
        Sohbet geçmişini Gemini'ye gönderilebilecek düz metin formatına
        dönüştürür.

        Args:
            history: Her biri {"role": "user"|"assistant", "content": "..."}
                     formatında dict listesi.

        Returns:
            Formatlı sohbet geçmişi metni.
        """
        if not history:
            return "(Önceki konuşma yok)"

        lines: List[str] = []
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prefix = "Kullanıcı" if role == "user" else "Asistan"
            lines.append(f"{prefix}: {content}")

        return "\n".join(lines)

    # ================================================================
    # 1) SORU-CEVAP (generate_answer)
    # ================================================================

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict],
    ) -> Dict:
        """
        Retrieve edilen chunk'ları bağlam olarak kullanarak soruyu yanıtlar.

        İşlem akışı:
          1. Chunk'lardan bağlam metni oluştur (_build_context)
          2. QA prompt şablonunu doldur
          3. Gemini modeline gönder
          4. Yanıtı ve kaynakları dict olarak döndür

        Args:
            question:          Kullanıcının sorduğu soru.
            retrieved_chunks:  VectorStore.similarity_search() çıktısı.
                               Her dict: {"text", "metadata", "similarity_score"}

        Returns:
            Şu anahtarları içeren dict:
              - "answer"  (str):       Gemini'nin ürettiği Türkçe yanıt.
              - "sources" (List[str]): Kullanılan kaynakların listesi.
                                       Örn: ["rapor.pdf (Sayfa 3)", "notlar.docx"]

        Raises:
            Exception: Gemini API hatası durumunda.

        Örnekler:
            >>> chain = RAGChain()
            >>> chunks = store.similarity_search("yapay zeka", k=5)
            >>> result = chain.generate_answer("Yapay zeka nedir?", chunks)
            >>> print(result["answer"])
            >>> print(result["sources"])
        """
        # ── Bağlam boşsa erken dön ──
        if not retrieved_chunks:
            logger.warning("generate_answer: Bağlam chunk'ı yok.")
            return {
                "answer": "Bu bilgi yüklenen dokümanlarda bulunamadı. "
                          "Lütfen önce bir doküman yükleyin.",
                "sources": [],
            }

        # ── Bağlam ve prompt oluştur ──
        context = self._build_context(retrieved_chunks)
        prompt = QA_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        # ── Gemini'ye gönder ──
        logger.info("Gemini'ye soru gönderiliyor: '%s'", question[:80])
        response = self.model.generate_content(prompt)
        answer = response.text

        # ── Kaynakları topla ──
        sources = self._extract_sources(retrieved_chunks)

        logger.info(
            "Yanıt üretildi (%d karakter, %d kaynak)",
            len(answer), len(sources),
        )

        return {
            "answer": answer,
            "sources": sources,
        }

    # ================================================================
    # 2) DOKÜMAN ÖZETLEME (summarize_document)
    # ================================================================

    def summarize_document(self, chunks: List[Dict]) -> str:
        """
        Verilen chunk listesindeki dokümanı madde madde özetler.

        Tüm chunk'lar tek bir metin olarak birleştirilir ve Gemini'ye
        özet talebi gönderilir. Uzun dokümanlar için chunk'lar
        sırayla birleştirilir.

        Özet formatı:
          • Ana başlıklar bold olarak belirtilir
          • Alt konular madde işaretleri ile listelenir
          • Hangi kaynak dosyadan alındığı belirtilir
          • Özet, orijinal içeriğin ~%20-30'u kadar uzunlukta olur

        Args:
            chunks: Özetlenecek dokümanın chunk'ları.
                    Her dict en az "text" anahtarı içermelidir.
                    Opsiyonel: "metadata" → {"source", "page"}

        Returns:
            Türkçe, madde madde formatlı özet metni.
            Chunk listesi boşsa bilgilendirme mesajı döner.

        Raises:
            Exception: Gemini API hatası durumunda.

        Örnekler:
            >>> chain = RAGChain()
            >>> chunks = processor.process_file("rapor.pdf")
            >>> # chunk'ları VectorStore formatına dönüştür
            >>> formatted = [{"text": c["text"], "metadata": c} for c in chunks]
            >>> summary = chain.summarize_document(formatted)
            >>> print(summary)
        """
        if not chunks:
            return "⚠️ Özetlenecek doküman içeriği bulunamadı."

        # ── Kaynak bilgilerini topla ──
        sources = self._extract_sources(chunks)
        source_info = ", ".join(sources) if sources else "Bilinmiyor"

        # ── Tüm chunk metinlerini birleştir ──
        content_parts: List[str] = []
        for chunk in chunks:
            text = chunk.get("text", "")
            if text.strip():
                content_parts.append(text)

        full_content = "\n\n".join(content_parts)

        if not full_content.strip():
            return "⚠️ Doküman içeriği boş, özet oluşturulamadı."

        # ── Çok uzun içerikleri kırp (Gemini token limiti) ──
        # Gemini 1.5 Flash geniş context window'a sahip (~1M token)
        # ama güvenli sınır olarak ~100K karakter ile sınırla
        max_chars = 100_000
        if len(full_content) > max_chars:
            full_content = full_content[:max_chars]
            logger.warning(
                "Doküman içeriği %d karaktere kırpıldı (orijinal: %d)",
                max_chars, len(full_content),
            )

        # ── Prompt oluştur ve gönder ──
        prompt = SUMMARY_PROMPT_TEMPLATE.format(content=full_content)

        logger.info(
            "Doküman özetleniyor: %s (%d chunk, %d karakter)",
            source_info, len(chunks), len(full_content),
        )
        response = self.model.generate_content(prompt)
        summary = response.text

        # ── Kaynak bilgisini sonuna ekle ──
        summary += f"\n\n📄 **Kaynak:** {source_info}"

        logger.info("Özet üretildi (%d karakter)", len(summary))
        return summary

    # ================================================================
    # 3) STREAMING YANIT (stream_answer)
    # ================================================================

    def stream_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict],
    ) -> Generator[str, None, None]:
        """
        Soruyu yanıtlar ve yanıtı token token yield eder (streaming).

        Bu metot bir Python generator fonksiyonudur. FastAPI'nin
        StreamingResponse'u veya Gradio'nun streaming özelliği ile
        kullanılmak üzere tasarlanmıştır.

        Her yield çağrısında yanıtın bir parçası (genellikle birkaç
        kelime veya bir cümle parçası) döndürülür. Bu sayede kullanıcı
        yanıtın tamamını beklemeden ilk kelimeleri görmeye başlar.

        Akış tamamlandığında kaynak bilgileri son parça olarak yield
        edilir.

        Args:
            question:          Kullanıcının sorduğu soru.
            retrieved_chunks:  VectorStore.similarity_search() çıktısı.

        Yields:
            str: Yanıtın bir parçası (token veya kelime grubu).
                 Son yield kaynak bilgilerini içerir.

        Raises:
            Exception: Gemini API streaming hatası durumunda.

        Örnekler:
            >>> chain = RAGChain()
            >>> chunks = store.similarity_search("yapay zeka", k=5)
            >>> for token in chain.stream_answer("Yapay zeka nedir?", chunks):
            ...     print(token, end="", flush=True)
        """
        # ── Bağlam boşsa ──
        if not retrieved_chunks:
            yield "Bu bilgi yüklenen dokümanlarda bulunamadı."
            return

        # ── Bağlam ve prompt oluştur ──
        context = self._build_context(retrieved_chunks)
        prompt = QA_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        # ── Gemini'ye streaming istek gönder ──
        logger.info("Streaming yanıt başlıyor: '%s'", question[:80])
        response = self.model.generate_content(prompt, stream=True)

        for chunk in response:
            if chunk.text:
                yield chunk.text

        # ── Kaynakları son parça olarak ekle ──
        sources = self._extract_sources(retrieved_chunks)
        if sources:
            yield "\n\n📚 **Kaynaklar:**\n"
            for src in sources:
                yield f"- {src}\n"

        logger.info("Streaming yanıt tamamlandı.")

    # ================================================================
    # 4) SOHBET GEÇMİŞİ İLE YANITLAMA (chat_with_history)
    # ================================================================

    def chat_with_history(
        self,
        question: str,
        history: List[Dict],
        retrieved_chunks: List[Dict],
    ) -> Dict:
        """
        Sohbet geçmişini dikkate alarak soruyu yanıtlar.

        Önceki soru-cevaplar prompt'a eklenir, böylece Gemini:
          • Önceki sorularla tutarlı cevap verir
          • "O ne demek?", "Biraz daha açıklar mısın?" gibi
            bağlam gerektiren takip sorularını anlayabilir
          • Aynı bilgiyi tekrarlamaktan kaçınır

        Args:
            question:          Kullanıcının son sorusu.
            history:           Konuşma geçmişi. Her eleman:
                               {"role": "user"|"assistant", "content": "..."}
                               Kronolojik sırada olmalıdır (eskiden yeniye).
            retrieved_chunks:  VectorStore.similarity_search() çıktısı.

        Returns:
            Şu anahtarları içeren dict:
              - "answer"  (str):       Gemini'nin ürettiği yanıt.
              - "sources" (List[str]): Kullanılan kaynaklar.
              - "history" (List[Dict]): Güncellenmiş sohbet geçmişi
                                        (son soru ve yanıt eklenmiş).

        Raises:
            Exception: Gemini API hatası durumunda.

        Örnekler:
            >>> chain = RAGChain()
            >>> history = [
            ...     {"role": "user", "content": "Yapay zeka nedir?"},
            ...     {"role": "assistant", "content": "Yapay zeka, ..."},
            ... ]
            >>> chunks = store.similarity_search("derin öğrenme", k=5)
            >>> result = chain.chat_with_history(
            ...     "Derin öğrenme ile ne farkı var?", history, chunks
            ... )
            >>> print(result["answer"])
        """
        # ── Bağlam boşsa ──
        if not retrieved_chunks:
            no_context_answer = (
                "Bu bilgi yüklenen dokümanlarda bulunamadı. "
                "Lütfen sorunuzu farklı şekilde sormayı deneyin "
                "veya ilgili dokümanı yükleyin."
            )
            updated_history = list(history or [])
            updated_history.append({"role": "user", "content": question})
            updated_history.append({"role": "assistant", "content": no_context_answer})
            return {
                "answer": no_context_answer,
                "sources": [],
                "history": updated_history,
            }

        # ── Bağlam, geçmiş ve prompt oluştur ──
        context = self._build_context(retrieved_chunks)
        formatted_history = self._format_history(history or [])

        prompt = CHAT_PROMPT_TEMPLATE.format(
            context=context,
            history=formatted_history,
            question=question,
        )

        # ── Gemini'ye gönder ──
        logger.info(
            "Sohbet geçmişi ile soru gönderiliyor: '%s' (geçmiş: %d mesaj)",
            question[:80], len(history or []),
        )
        response = self.model.generate_content(prompt)
        answer = response.text

        # ── Kaynakları topla ──
        sources = self._extract_sources(retrieved_chunks)

        # ── Geçmişi güncelle ──
        updated_history = list(history or [])
        updated_history.append({"role": "user", "content": question})
        updated_history.append({"role": "assistant", "content": answer})

        logger.info(
            "Sohbet yanıtı üretildi (%d karakter, %d kaynak, toplam %d mesaj)",
            len(answer), len(sources), len(updated_history),
        )

        return {
            "answer": answer,
            "sources": sources,
            "history": updated_history,
        }

    # ================================================================
    # 5) TAM RAG AKIŞI (query — geriye uyumluluk)
    # ================================================================

    def query(self, question: str, n_results: int = 5) -> Dict:
        """
        Tam RAG akışını çalıştırır: vektör arama + yanıt üretimi.

        Bu metot generate_answer'ın üzerine vektör arama adımını
        ekler. Dışarıdan chunk geçmek yerine soruyu doğrudan sorar.

        Args:
            question:  Kullanıcının sorduğu soru.
            n_results: Vektör aramada getirilecek chunk sayısı.

        Returns:
            generate_answer ile aynı formatta dict:
              - "answer"  (str)
              - "sources" (List[str])
        """
        # 1 — Retrieval
        retrieved_chunks = self.vector_store.similarity_search(
            query=question,
            k=n_results,
        )

        # 2 — Generation
        return self.generate_answer(question, retrieved_chunks)
