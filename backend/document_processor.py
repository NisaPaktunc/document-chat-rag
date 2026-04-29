"""
document_processor.py — Doküman Okuma, Temizleme ve Parçalama
==============================================================
Yüklenen TXT, PDF, DOC ve DOCX dosyalarını okur, ham metni temizler
ve RAG pipeline'ına beslenecek küçük metin parçalarına (chunk) böler.

Desteklenen formatlar:
  • .txt  — Düz metin dosyaları
  • .pdf  — PyMuPDF (fitz) ile sayfa sayfa okuma
  • .doc  — PyMuPDF (fitz) ile okuma (ikili Word formatı)
  • .docx — python-docx ile paragraf paragraf okuma

Sınıf: DocumentProcessor
  Tek giriş noktası olarak tüm doküman işleme adımlarını yönetir.
  extract_text → clean_text → chunk_text → process_file → process_multiple_files

Chunking stratejisi:
  • Maks. ~500 token ≈ 2000 karakter
  • %20 overlap (400 karakter)
  • Cümle sınırlarına duyarlı bölme (nokta, soru/ünlem işareti, satır sonu)
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from typing import Dict, List, Optional

import fitz  # PyMuPDF — PDF ve DOC okuma
from docx import Document as DocxDocument  # DOCX okuma

from config import CHUNK_SIZE, CHUNK_OVERLAP

# ── Logger ──────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Desteklenen Uzantılar ───────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".doc", ".docx"}


# ── Özel Hatalar ────────────────────────────────────────────────────

class UnsupportedFormatError(Exception):
    """Desteklenmeyen dosya formatı yüklendiğinde fırlatılır."""
    pass


class DocumentReadError(Exception):
    """Dosya okunurken bir hata oluştuğunda fırlatılır."""
    pass


# ── DocumentProcessor Sınıfı ───────────────────────────────────────

class DocumentProcessor:
    """
    Doküman işleme pipeline'ını yöneten ana sınıf.

    Tek bir dosyayı veya birden fazla dosyayı okur, temizler ve
    metadata zenginleştirilmiş chunk'lara böler.

    Attributes:
        chunk_size:    Her chunk'ın maksimum karakter uzunluğu (varsayılan: 2000).
        chunk_overlap: Ardışık chunk'lar arasındaki örtüşen karakter sayısı
                       (varsayılan: chunk_size * 0.20 = 400).

    Kullanım:
        >>> processor = DocumentProcessor()
        >>> chunks = processor.process_file("rapor.pdf")
        >>> for c in chunks:
        ...     print(c["chunk_id"], c["text"][:80])
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ================================================================
    # 1) METIN ÇIKARMA
    # ================================================================

    def extract_text(self, file_path: str) -> str:
        """
        Dosya uzantısına göre uygun okuyucuyu seçer ve ham metni döndürür.

        Desteklenen formatlar: .txt, .pdf, .doc, .docx

        Args:
            file_path: Okunacak dosyanın tam veya göreli yolu.

        Returns:
            Dosyanın tamamından çıkarılan ham metin (temizlenmemiş).

        Raises:
            FileNotFoundError:       Dosya bulunamazsa.
            UnsupportedFormatError:  Uzantı desteklenmiyorsa.
            DocumentReadError:       Okuma sırasında beklenmeyen hata olursa.

        Örnekler:
            >>> dp = DocumentProcessor()
            >>> text = dp.extract_text("ornek.pdf")
            >>> print(text[:100])
        """
        # ── Dosya var mı? ──
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"Dosya bulunamadı: '{file_path}'. "
                "Lütfen dosya yolunu kontrol edin."
            )

        ext = os.path.splitext(file_path)[1].lower()

        # ── Uzantı destekleniyor mu? ──
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Desteklenmeyen dosya formatı: '{ext}'. "
                f"Desteklenen formatlar: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        try:
            if ext == ".txt":
                return self._read_txt(file_path)
            elif ext == ".pdf":
                return self._read_pdf(file_path)
            elif ext == ".doc":
                return self._read_doc(file_path)
            elif ext == ".docx":
                return self._read_docx(file_path)
        except (UnsupportedFormatError, FileNotFoundError):
            raise
        except Exception as e:
            raise DocumentReadError(
                f"'{file_path}' dosyası okunurken hata oluştu: {e}"
            ) from e

        # Buraya düşmemeli ama type-checker için:
        return ""  # pragma: no cover

    # ── Format-Specific Okuyucular ──────────────────────────────────

    @staticmethod
    def _read_txt(file_path: str) -> str:
        """
        Düz metin (.txt) dosyasını okur.

        UTF-8 → Latin-1 sırasıyla encoding dener.
        """
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1254"):
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        # Hiçbir encoding işlemediyse binary modda dene
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    @staticmethod
    def _read_pdf(file_path: str) -> str:
        """
        PDF dosyasını PyMuPDF ile sayfa sayfa okur ve tüm metni birleştirir.

        Her sayfanın metni araya çift satır sonu ile ayrılır, böylece
        chunk_text aşamasında sayfa geçişleri doğal bölme noktası olur.
        """
        pages_text: List[str] = []
        with fitz.open(file_path) as doc:
            for page in doc:
                text = page.get_text("text")
                if text.strip():
                    pages_text.append(text)
        return "\n\n".join(pages_text)

    @staticmethod
    def _read_doc(file_path: str) -> str:
        """
        Eski Word (.doc) formatını PyMuPDF ile okur.

        PyMuPDF 1.23+ sürümleri .doc dosyalarını da açabilir.
        Eğer açılamazsa anlamlı bir hata mesajı verir.
        """
        try:
            with fitz.open(file_path) as doc:
                pages_text: List[str] = []
                for page in doc:
                    text = page.get_text("text")
                    if text.strip():
                        pages_text.append(text)
            return "\n\n".join(pages_text)
        except Exception as e:
            raise DocumentReadError(
                f".doc dosyası okunamadı: {e}. "
                "Bu format sınırlı destek sunar; mümkünse dosyayı "
                ".docx veya .pdf olarak dönüştürmeyi deneyin."
            ) from e

    @staticmethod
    def _read_docx(file_path: str) -> str:
        """
        DOCX dosyasını python-docx ile paragraf paragraf okur.

        Boş paragraflar atlanır, geri kalanlar satır sonu ile birleştirilir.
        """
        doc = DocxDocument(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)

    # ================================================================
    # 2) METİN TEMİZLEME
    # ================================================================

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Ham metni normalize eder ve gereksiz gürültüyü temizler.

        Uygulanan adımlar (sırasıyla):
          1. Unicode NBSP (\\xa0) ve benzeri boşlukları normal boşluğa çevir
          2. Satır içi fazla boşlukları tek boşluğa indir
          3. Üç veya daha fazla ardışık satır sonunu ikiye indir
          4. Tamamen boş satırları kaldır
          5. Baş ve sondaki boşlukları temizle

        Args:
            text: Temizlenecek ham metin.

        Returns:
            Temizlenmiş metin.

        Örnekler:
            >>> DocumentProcessor.clean_text("  Merhaba   dünya  \\n\\n\\n  ")
            'Merhaba dünya'
        """
        if not text:
            return ""

        # 1 — Non-breaking space ve benzeri özel boşlukları temizle
        text = text.replace("\xa0", " ")
        text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

        # 2 — Satır içi çoklu boşlukları tek boşluğa indir
        text = re.sub(r"[^\S\n]+", " ", text)

        # 3 — 3+ ardışık satır sonunu 2'ye indir (paragraf ayracı olarak kalsın)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 4 — Tamamen boş (sadece whitespace) satırları kaldır
        lines = text.splitlines()
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        text = "\n".join(cleaned_lines)

        # 5 — Baş ve sondaki boşluklar
        return text.strip()

    # ================================================================
    # 3) CHUNK'LARA AYIRMA
    # ================================================================

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Temizlenmiş metni akıllı biçimde chunk'lara böler.

        Bölme stratejisi:
          • chunk_size (2000 karakter) sınırına yaklaşınca metni doğrudan
            kesmek yerine en yakın cümle sonu aranır.
          • Cümle sonu olarak şunlar kabul edilir: . ? ! ve satır sonu (\\n)
          • Eğer chunk_size içinde hiç cümle sonu bulunamazsa, son boşluk
            karakterinde bölünür (kelime ortasında bölmeyi önler).
          • Ardışık chunk'lar arasında %20 overlap uygulanır.

        Args:
            text:     Chunk'lara bölünecek temizlenmiş metin.
            metadata: Her chunk'a eklenecek temel metadata (ör. {"source": "rapor.pdf"}).
                      Her chunk'a otomatik olarak chunk_id eklenir.

        Returns:
            Her biri şu anahtarları içeren dict listesi:
              - chunk_id  (str):  Benzersiz UUID
              - text      (str):  Chunk metni
              - source    (str):  Kaynak dosya adı (metadata'dan)
              - page      (int):  Sayfa numarası — yalnızca PDF için (metadata'dan)
              - …diğer metadata alanları

        Örnekler:
            >>> dp = DocumentProcessor(chunk_size=100, chunk_overlap=20)
            >>> chunks = dp.chunk_text("Kısa bir metin.", {"source": "test.txt"})
            >>> len(chunks)
            1
        """
        if not text or not text.strip():
            return []

        base_metadata = metadata or {}
        chunks: List[Dict] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)

            # ── Tam chunk ise direkt ekle ──
            if end >= text_length:
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append(self._build_chunk(chunk_text, base_metadata))
                break

            # ── Akıllı bölme: cümle sonunu ara ──
            chunk_candidate = text[start:end]
            split_pos = self._find_best_split(chunk_candidate)

            if split_pos > 0:
                # Cümle sonu bulundu — oradan böl
                actual_end = start + split_pos
            else:
                # Cümle sonu yok — kelime sınırında böl
                actual_end = end

            chunk_text = text[start:actual_end].strip()
            if chunk_text:
                chunks.append(self._build_chunk(chunk_text, base_metadata))

            # ── Sonraki başlangıç: overlap kadar geri git ──
            step = actual_end - start
            start += max(step - self.chunk_overlap, 1)

        return chunks

    # ── Yardımcı: En iyi bölme noktasını bul ───────────────────────

    @staticmethod
    def _find_best_split(text: str) -> int:
        """
        Chunk metninin sonuna yakın en uygun bölme noktasını bulur.

        Öncelik sırası:
          1. Son paragraf sonu (çift satır sonu)
          2. Son cümle sonu (. ? !)
          3. Son satır sonu (tek \\n)
          4. Bulunamazsa 0 döner (çağıran metot kelime sınırında böler)

        Yalnızca metnin son %50'sinde arar, çok geriye gitmemek için.
        """
        search_start = len(text) // 2  # Metnin ikinci yarısında ara

        # 1 — Paragraf sonu
        pos = text.rfind("\n\n", search_start)
        if pos > search_start:
            return pos + 2  # "\n\n" sonrasına konumlan

        # 2 — Cümle sonu: "." "?" "!" ve ardından boşluk veya metin sonu
        #     Kısaltmalarda yanlış bölmeyi azaltmak için en az 2 karakter sonrasına bak
        for pattern in (". ", "? ", "! ", ".\n", "?\n", "!\n"):
            pos = text.rfind(pattern, search_start)
            if pos > search_start:
                return pos + len(pattern)

        # 3 — Tek satır sonu
        pos = text.rfind("\n", search_start)
        if pos > search_start:
            return pos + 1

        # 4 — Son çare: boşluk karakteri (kelime ortasında bölmeyi önle)
        pos = text.rfind(" ", search_start)
        if pos > search_start:
            return pos + 1

        return 0  # Hiçbir uygun nokta bulunamadı

    # ── Yardımcı: Chunk dict'i oluştur ─────────────────────────────

    @staticmethod
    def _build_chunk(text: str, base_metadata: Dict) -> Dict:
        """Chunk metni ve metadata'yı tek bir dict'te birleştirir."""
        chunk = {
            "chunk_id": str(uuid.uuid4()),
            "text": text,
        }
        chunk.update(base_metadata)
        return chunk

    # ================================================================
    # 4) TEK DOSYA İŞLEME
    # ================================================================

    def process_file(self, file_path: str) -> List[Dict]:
        """
        Tek bir dosyayı baştan sona işler: oku → temizle → chunk'la.

        PDF dosyaları için her sayfaya ayrı sayfa numarası atanır.
        Diğer formatlar tek parça olarak işlenir.

        Args:
            file_path: İşlenecek dosyanın tam yolu.

        Returns:
            Metadata zenginleştirilmiş chunk dict'lerinin listesi.
            Her dict şunları içerir:
              - chunk_id  (str)
              - text      (str)
              - source    (str) — dosya adı
              - page      (int) — yalnızca PDF dosyaları için

        Raises:
            FileNotFoundError:      Dosya bulunamazsa.
            UnsupportedFormatError: Format desteklenmiyorsa.
            DocumentReadError:      Okuma hatası olursa.

        Örnekler:
            >>> dp = DocumentProcessor()
            >>> chunks = dp.process_file("sunum.pdf")
            >>> print(f"{len(chunks)} chunk üretildi")
        """
        filename = os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        logger.info("Dosya işleniyor: %s", filename)

        # ── PDF: Sayfa bazında metadata ekle ──
        if ext == ".pdf":
            return self._process_pdf_with_pages(file_path, filename)

        # ── Diğer formatlar: Tek parça olarak işle ──
        raw_text = self.extract_text(file_path)
        cleaned = self.clean_text(raw_text)

        if not cleaned:
            logger.warning("Dosyadan metin çıkarılamadı: %s", filename)
            return []

        metadata = {"source": filename}
        chunks = self.chunk_text(cleaned, metadata)

        logger.info(
            "%s → %d chunk üretildi (toplam %d karakter)",
            filename, len(chunks), len(cleaned),
        )
        return chunks

    def _process_pdf_with_pages(
        self, file_path: str, filename: str
    ) -> List[Dict]:
        """
        PDF'i sayfa sayfa işler ve her chunk'a sayfa numarası ekler.

        Bu sayede RAG yanıtlarında "Sayfa 5'te şöyle yazıyor…" gibi
        kesin kaynak referansları verilebilir.
        """
        all_chunks: List[Dict] = []

        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text("text")
                cleaned = self.clean_text(page_text)

                if not cleaned:
                    continue

                metadata = {
                    "source": filename,
                    "page": page_num,
                }
                page_chunks = self.chunk_text(cleaned, metadata)
                all_chunks.extend(page_chunks)

        logger.info(
            "%s (PDF, %d sayfa) → %d chunk üretildi",
            filename,
            len(list(fitz.open(file_path))),
            len(all_chunks),
        )
        return all_chunks

    # ================================================================
    # 5) ÇOKLU DOSYA İŞLEME
    # ================================================================

    def process_multiple_files(self, file_paths: List[str]) -> List[Dict]:
        """
        Birden fazla dosyayı sırayla işler ve tüm chunk'ları birleştirir.

        Bir dosyada hata olursa o dosya atlanır, diğer dosyalar işlenmeye
        devam eder. Hatalar loglanır ve sonuç listesine eklenmez.

        Args:
            file_paths: İşlenecek dosya yollarının listesi.

        Returns:
            Tüm dosyalardan üretilen chunk'ların birleşik listesi.

        Örnekler:
            >>> dp = DocumentProcessor()
            >>> files = ["rapor.pdf", "notlar.docx", "ozet.txt"]
            >>> all_chunks = dp.process_multiple_files(files)
            >>> print(f"Toplam: {len(all_chunks)} chunk")
        """
        all_chunks: List[Dict] = []
        success_count = 0
        error_count = 0

        for file_path in file_paths:
            try:
                chunks = self.process_file(file_path)
                all_chunks.extend(chunks)
                success_count += 1
            except (FileNotFoundError, UnsupportedFormatError) as e:
                logger.error("Dosya atlandı — %s: %s", file_path, e)
                error_count += 1
            except DocumentReadError as e:
                logger.error("Okuma hatası — %s: %s", file_path, e)
                error_count += 1
            except Exception as e:
                logger.error(
                    "Beklenmeyen hata — %s: %s", file_path, e, exc_info=True
                )
                error_count += 1

        logger.info(
            "Çoklu dosya işleme tamamlandı: %d başarılı, %d hatalı, "
            "toplam %d chunk üretildi",
            success_count, error_count, len(all_chunks),
        )
        return all_chunks
