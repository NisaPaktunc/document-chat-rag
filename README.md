# 📚 Document Chat — Dokümanlarınla Sohbet Et

PDF, DOCX, DOC ve TXT dokümanlarınızı yükleyin, içerikleri hakkında Türkçe
sorular sorun. Yanıtlar **Google Gemini 1.5 Flash** modeli tarafından,
dokümanlarınızdaki bilgilere dayanarak üretilir.

**RAG (Retrieval-Augmented Generation)** mimarisi ile çalışır:
dokümanlarınız parçalara ayrılır, vektör veritabanında saklanır ve
her soruda en ilgili parçalar bulunarak yapay zekaya bağlam olarak verilir.

---

## 🏗️ Mimari

```
┌─────────────────────────────────────────────────────────────────┐
│                        KULLANICI                                │
│                    (Tarayıcı: 7860)                             │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│              FRONTEND — Gradio (Python)                         │
│                                                                  │
│  📤 Dosya Yükleme  │ 💬 Sohbet │ 📝 Özet │ 📂 Doküman Listesi  │
│                                                                  │
│  HTTP istekleri → http://backend:8000                            │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│              BACKEND — FastAPI (Python)                          │
│                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────────┐  │
│  │ DocumentProc.  │  │  VectorStore   │  │    RAGChain       │  │
│  │ ─────────────  │  │  ────────────  │  │  ──────────────── │  │
│  │ • PDF (PyMuPDF)│  │ • ChromaDB     │  │ • generate_answer │  │
│  │ • DOCX         │  │ • Embedding    │  │ • summarize_doc   │  │
│  │ • TXT / DOC    │  │ • Similarity   │  │ • stream_answer   │  │
│  │ • Chunking     │  │   Search       │  │ • chat_w_history  │  │
│  └────────┬───────┘  └───────┬────────┘  └──────┬────────────┘  │
│           │                  │                   │               │
│           │                  ▼                   ▼               │
│           │          ┌──────────────┐    ┌──────────────┐       │
│           │          │  ChromaDB    │    │ Google Gemini│       │
│           │          │  (Kalıcı)    │    │  1.5 Flash   │       │
│           │          │  /chroma_db  │    │  + Embedding │       │
│           │          └──────────────┘    └──────────────┘       │
│           │                                      │               │
│           └──────────────────────────────────────┘               │
└──────────────────────────────────────────────────────────────────┘
```

## 📁 Proje Yapısı

```
document-chat/
├── backend/
│   ├── main.py                # FastAPI endpoint'leri (8 adet)
│   ├── config.py              # Yapılandırma & ortam değişkenleri
│   ├── document_processor.py  # PDF/DOCX/TXT okuma ve chunking
│   ├── vector_store.py        # ChromaDB & embedding (retry mekanizmalı)
│   ├── rag_chain.py           # RAG zinciri (4 mod: QA, özet, stream, chat)
│   ├── requirements.txt       # Python bağımlılıkları
│   └── Dockerfile             # Backend Docker image'ı
├── frontend/
│   ├── app.py                 # Gradio arayüzü (4 sekmeli)
│   └── Dockerfile             # Frontend Docker image'ı
├── .env.example               # Ortam değişkenleri şablonu
├── docker-compose.yml         # Docker Compose yapılandırması
└── README.md                  # Bu dosya
```

---

## 🚀 Kurulum

### Ön Gereksinimler

- **Python 3.11+** (manuel kurulum için)
- **Docker & Docker Compose** (Docker kurulumu için)
- **Google API Key** — [Google AI Studio](https://aistudio.google.com/app/apikey) adresinden alın

### 1. Ortam Değişkenlerini Ayarlayın

```bash
# .env.example dosyasını kopyalayın
cp .env.example .env

# .env dosyasını açıp GOOGLE_API_KEY değerini girin
```

> ⚠️ **GOOGLE_API_KEY olmadan uygulama çalışmaz!**

---

### 🐳 Docker ile Kurulum (Önerilen)

En hızlı ve temiz kurulum yöntemi:

```bash
# 1. Projeyi klonlayın
git clone <repo-url>
cd document-chat

# 2. .env dosyasını oluşturun
cp .env.example .env
# → GOOGLE_API_KEY değerini girin

# 3. Docker ile başlatın
docker-compose up --build

# Arka planda çalıştırmak için:
docker-compose up --build -d
```

Uygulama hazır olduğunda:
- 🌐 **Frontend:** http://localhost:7860
- 📡 **API Docs:** http://localhost:8000/docs
- 💚 **Health:** http://localhost:8000/health

```bash
# Durdurma
docker-compose down

# Durdurma + verileri silme (dikkat!)
docker-compose down -v

# Logları takip etme
docker-compose logs -f backend
docker-compose logs -f frontend
```

---

### 🖥️ Manuel Kurulum

```bash
# 1. Sanal ortam oluşturun (önerilen)
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

# 2. Bağımlılıkları yükleyin
pip install -r backend/requirements.txt

# 3. Backend'i başlatın
cd backend
python main.py
# → http://localhost:8000 adresinde çalışır

# 4. Frontend'i başlatın (yeni terminal)
cd frontend
python app.py
# → http://localhost:7860 adresinde çalışır
```

---

## 📡 API Endpoint'leri

Tüm endpoint'ler http://localhost:8000 adresinden erişilebilir.
Swagger UI: http://localhost:8000/docs

| Metot    | Yol          | Açıklama                                    | Request Body                              |
|----------|--------------|---------------------------------------------|-------------------------------------------|
| `POST`   | `/upload`    | Birden fazla dosya yükle (TXT/PDF/DOC/DOCX) | `multipart/form-data` (files)             |
| `POST`   | `/ask`       | Soru sor (RAG ile yanıt üret)               | `{"question": str, "k": 5}`              |
| `POST`   | `/chat`      | Sohbet geçmişi ile soru sor                 | `{"question": str, "history": [], "k": 5}`|
| `POST`   | `/summarize` | Doküman özetle (belirli veya tümü)          | `{"document_name": str \| null}`          |
| `POST`   | `/stream`    | Streaming yanıt (token token)               | `{"question": str, "k": 5}`              |
| `GET`    | `/documents` | Yüklü doküman listesi                       | —                                         |
| `DELETE` | `/documents` | Tüm dokümanları sil                         | —                                         |
| `GET`    | `/health`    | Sistem sağlık kontrolü                      | —                                         |

---

## 💡 Örnek Kullanım

### Doküman Yükleme (cURL)

```bash
# Tek dosya yükle
curl -X POST http://localhost:8000/upload \
  -F "files=@rapor.pdf"

# Birden fazla dosya yükle
curl -X POST http://localhost:8000/upload \
  -F "files=@rapor.pdf" \
  -F "files=@notlar.docx" \
  -F "files=@ozet.txt"
```

**Yanıt:**
```json
{
  "status": "success",
  "files_processed": 3,
  "chunks_added": 47
}
```

### Soru Sorma

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Raporun ana konusu nedir?", "k": 5}'
```

**Yanıt:**
```json
{
  "answer": "Raporun ana konusu...",
  "sources": ["rapor.pdf (Sayfa 1)", "rapor.pdf (Sayfa 3)"],
  "chunks_used": 5
}
```

### Sohbet Geçmişi ile Soru

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Bu konuyu biraz daha açıklar mısın?",
    "history": [
      {"role": "user", "content": "Raporun konusu ne?"},
      {"role": "assistant", "content": "Raporun ana konusu..."}
    ],
    "k": 5
  }'
```

### Doküman Özetleme

```bash
# Belirli dokümanı özetle
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"document_name": "rapor.pdf"}'

# Tüm dokümanları özetle
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Doküman Listesi & Silme

```bash
# Yüklü dokümanları listele
curl http://localhost:8000/documents

# Tüm dokümanları sil
curl -X DELETE http://localhost:8000/documents

# Sağlık kontrolü
curl http://localhost:8000/health
```

---

## 🛠️ Teknolojiler

| Katman        | Teknoloji                  | Versiyon  |
|---------------|----------------------------|-----------|
| Backend       | FastAPI + Uvicorn          | 0.115.12  |
| LLM           | Google Gemini 1.5 Flash    | —         |
| Embedding     | Google text-embedding-004  | 768 boyut |
| Vektör DB     | ChromaDB                   | 0.6.3     |
| PDF Okuma     | PyMuPDF (fitz)             | 1.25.4    |
| DOCX Okuma    | python-docx                | 1.1.2     |
| Frontend      | Gradio                     | 5.25.2    |
| HTTP İstemci  | httpx                      | 0.28.1    |
| Container     | Docker + Docker Compose    | —         |

## ⚙️ Yapılandırma Parametreleri

| Parametre       | Varsayılan    | Açıklama                                 |
|-----------------|---------------|------------------------------------------|
| `GOOGLE_API_KEY`| — (zorunlu)   | Google AI Studio API anahtarı            |
| `CHUNK_SIZE`    | `2000`        | Metin parçalama boyutu (karakter)        |
| `CHUNK_OVERLAP` | `400`         | Parçalar arası örtüşme (%20)             |
| `MAX_FILE_SIZE_MB`| `10`        | Maksimum dosya boyutu                    |
| `BACKEND_HOST`  | `0.0.0.0`     | Backend sunucu adresi                    |
| `BACKEND_PORT`  | `8000`        | Backend sunucu portu                     |
| `CHROMA_PERSIST_DIR`| `./chroma_db` | ChromaDB kalıcı depolama dizini      |

---

## 📝 Lisans

MIT
