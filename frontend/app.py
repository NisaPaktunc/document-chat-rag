"""
app.py — Gradio Frontend (📚 Dokümanlarınla Sohbet Et)
========================================================
Kullanıcıya doküman yükleme, sohbet (geçmişli), doküman özetleme ve
doküman yönetimi arayüzü sunan Gradio uygulaması.

Tüm işlemler backend FastAPI sunucusuna HTTP istekleri ile yapılır.

Sekmeler:
  1. 📤 Dosya Yükleme   — Çoklu dosya yükleme, sonuç gösterimi
  2. 💬 Sohbet          — Konuşma geçmişli RAG soru-cevap
  3. 📝 Özet            — Doküman bazlı veya toplu özetleme
  4. 📂 Doküman Listesi — Yüklü dokümanlar tablosu, toplu silme
"""

from __future__ import annotations

import os
from typing import List, Tuple

import gradio as gr
import httpx

# ── Yapılandırma ────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 180.0  # Uzun işlemler için (yükleme, özetleme)


# =====================================================================
# BACKEND İLETİŞİM FONKSİYONLARI
# =====================================================================

def _api_get(path: str, timeout: float = 15.0) -> dict | None:
    """Backend'e GET isteği gönderir. Hata varsa None döndürür."""
    try:
        resp = httpx.get(f"{BACKEND_URL}{path}", timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except httpx.ConnectError:
        pass
    except Exception:
        pass
    return None


# ── 1) Dosya Yükleme ───────────────────────────────────────────────

def upload_files(file_list) -> str:
    """
    Birden fazla dosyayı backend'e yükler.

    Args:
        file_list: Gradio'dan gelen dosya nesneleri listesi.

    Returns:
        Kullanıcıya gösterilecek sonuç mesajı.
    """
    if not file_list:
        return "⚠️ Lütfen en az bir dosya seçin."

    file_handles = []
    try:
        # Gradio dosyalarını multipart form-data olarak hazırla
        multipart_files = []
        for f in file_list:
            # Gradio dosya yolundan dosya adını çıkar
            filepath = f.name if hasattr(f, "name") else str(f)
            filename = os.path.basename(filepath)
            fh = open(filepath, "rb")
            file_handles.append(fh)
            multipart_files.append(("files", (filename, fh)))

        response = httpx.post(
            f"{BACKEND_URL}/upload",
            files=multipart_files,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 200:
            data = response.json()
            return (
                f"✅ Yükleme başarılı!\n\n"
                f"📁 İşlenen dosya sayısı: {data['files_processed']}\n"
                f"🧩 Oluşturulan chunk sayısı: {data['chunks_added']}\n\n"
                f"💡 Artık sohbet sekmesinden sorularınızı sorabilirsiniz."
            )
        else:
            detail = response.json().get("detail", response.text)
            return f"❌ Yükleme hatası:\n{detail}"

    except httpx.ConnectError:
        return (
            "❌ Backend sunucusuna bağlanılamadı.\n\n"
            "Lütfen backend'in çalıştığından emin olun:\n"
            f"  → {BACKEND_URL}/health"
        )
    except Exception as e:
        return f"❌ Beklenmeyen hata: {str(e)}"
    finally:
        # Dosya handle'larını temizle
        for fh in file_handles:
            try:
                fh.close()
            except Exception:
                pass


# ── 2) Sohbet (Konuşma Geçmişli) ──────────────────────────────────

def chat_ask(
    message: str,
    chatbot_history: List[Tuple[str, str]],
    server_history: List[dict],
) -> Tuple[List[Tuple[str, str]], str, List[dict]]:
    """
    Sohbet geçmişi ile birlikte soruyu backend'e gönderir.

    Returns:
        (güncel chatbot geçmişi, boş input, güncel sunucu geçmişi)
    """
    if not message or not message.strip():
        return chatbot_history or [], "", server_history or []

    try:
        response = httpx.post(
            f"{BACKEND_URL}/chat",
            json={
                "question": message.strip(),
                "history": server_history or [],
                "k": 5,
            },
            timeout=60.0,
        )

        if response.status_code == 200:
            data = response.json()
            answer = data["answer"]

            # ── Kaynak gösterimi ──
            sources = data.get("sources", [])
            if sources:
                answer += "\n\n"
                for src in sources:
                    answer += f"📄 Kaynak: {src}\n"

            updated_history = data.get("history", server_history or [])

        elif response.status_code == 400:
            detail = response.json().get("detail", "")
            answer = f"⚠️ {detail}"
            updated_history = server_history or []
        else:
            answer = f"❌ Sunucu hatası (HTTP {response.status_code})"
            updated_history = server_history or []

    except httpx.ConnectError:
        answer = (
            "❌ Backend sunucusuna bağlanılamadı.\n"
            "Lütfen sunucunun çalıştığını kontrol edin."
        )
        updated_history = server_history or []
    except httpx.ReadTimeout:
        answer = "⏱️ İstek zaman aşımına uğradı. Lütfen tekrar deneyin."
        updated_history = server_history or []
    except Exception as e:
        answer = f"❌ Hata: {str(e)}"
        updated_history = server_history or []

    chatbot_history = chatbot_history or []
    chatbot_history.append((message, answer))
    return chatbot_history, "", updated_history


def clear_chat() -> Tuple[list, str, list]:
    """Sohbet geçmişini sıfırlar."""
    return [], "", []


# ── 3) Doküman Özetleme ────────────────────────────────────────────

def summarize(doc_choice: str) -> str:
    """
    Seçilen dokümanı veya tüm dokümanları özetler.

    Args:
        doc_choice: Dropdown'dan gelen seçim.
                    "(Tümünü özetle)" ise tüm dokümanlar özetlenir.
    """
    try:
        payload = {}
        if doc_choice and doc_choice != "(Tümünü özetle)":
            payload["document_name"] = doc_choice

        response = httpx.post(
            f"{BACKEND_URL}/summarize",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 200:
            return response.json().get("summary", "Özet oluşturulamadı.")
        else:
            detail = response.json().get("detail", response.text)
            return f"❌ Özetleme hatası:\n{detail}"

    except httpx.ConnectError:
        return "❌ Backend sunucusuna bağlanılamadı."
    except httpx.ReadTimeout:
        return "⏱️ Özetleme zaman aşımına uğradı. Doküman çok büyük olabilir."
    except Exception as e:
        return f"❌ Hata: {str(e)}"


def refresh_doc_dropdown() -> gr.Dropdown:
    """Özet sekmesindeki doküman dropdown'ını günceller."""
    data = _api_get("/documents")
    docs = data.get("documents", []) if data else []
    choices = ["(Tümünü özetle)"] + docs
    return gr.Dropdown(choices=choices, value="(Tümünü özetle)")


# ── 4) Doküman Listesi ─────────────────────────────────────────────

def fetch_document_table() -> list:
    """
    Yüklü dokümanları tablo formatında döndürür.

    Returns:
        [[No, Doküman Adı], ...] formatında liste.
    """
    data = _api_get("/documents")
    if not data:
        return [["—", "Backend'e bağlanılamadı", "—"]]

    docs = data.get("documents", [])
    total_chunks = data.get("total_chunks", 0)

    if not docs:
        return [["—", "Henüz doküman yüklenmemiş", "—"]]

    # Her doküman için satır oluştur
    rows = []
    for i, doc_name in enumerate(docs, start=1):
        rows.append([str(i), doc_name])

    # Toplam satırı
    rows.append(["", f"📊 Toplam: {len(docs)} doküman, {total_chunks} chunk"])

    return rows


def delete_all_documents() -> Tuple[str, list]:
    """
    Tüm dokümanları siler ve tabloyu günceller.

    Returns:
        (sonuç mesajı, güncel tablo verisi)
    """
    try:
        response = httpx.delete(
            f"{BACKEND_URL}/documents",
            timeout=15.0,
        )
        if response.status_code == 200:
            data = response.json()
            msg = (
                f"🗑️ {data.get('message', 'Silindi.')}\n"
                f"📄 Silinen doküman: {data.get('deleted_documents', '?')}\n"
                f"🧩 Silinen chunk: {data.get('deleted_chunks', '?')}"
            )
            return msg, fetch_document_table()
        return f"❌ Hata: {response.text}", fetch_document_table()
    except httpx.ConnectError:
        return "❌ Backend sunucusuna bağlanılamadı.", fetch_document_table()
    except Exception as e:
        return f"❌ Hata: {str(e)}", fetch_document_table()


# ── 5) Sistem Sağlık Kontrolü ──────────────────────────────────────

def health_check() -> str:
    """Backend sağlık durumunu sorgular."""
    data = _api_get("/health")
    if not data:
        return "🔴 Backend sunucusuna bağlanılamadı."

    icon = "🟢" if data.get("status") == "healthy" else "🔴"
    uptime_min = data.get("uptime_seconds", 0) / 60

    return (
        f"{icon} Durum: {data['status']}\n"
        f"📄 Yüklü doküman: {data.get('total_documents', 0)}\n"
        f"🧩 Toplam chunk: {data.get('total_chunks', 0)}\n"
        f"⏱️ Çalışma süresi: {uptime_min:.1f} dakika"
    )


# =====================================================================
# GRADIO ARAYÜZÜ
# =====================================================================

CUSTOM_CSS = """
.main-title {
    text-align: center;
    margin-bottom: 0.5em;
}
.main-title h1 {
    font-size: 2em;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}
.subtitle {
    text-align: center;
    color: #6b7280;
    margin-top: -0.5em;
    margin-bottom: 1.5em;
    font-size: 1.05em;
}
.danger-zone {
    border: 1px solid #ef4444;
    border-radius: 8px;
    padding: 16px;
    margin-top: 16px;
}
footer { display: none !important; }
"""

with gr.Blocks(
    title="📚 Dokümanlarınla Sohbet Et",
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.purple,
    ),
    css=CUSTOM_CSS,
) as demo:

    # ── State: Sunucu tarafı sohbet geçmişi ──
    server_history = gr.State([])

    # ── Başlık ──
    gr.HTML(
        """
        <div class="main-title">
            <h1>📚 Dokümanlarınla Sohbet Et</h1>
        </div>
        <p class="subtitle">
            Dokümanlarınızı yükleyin, sorularınızı sorun — yapay zeka yanıtlasın.
            <br/>Desteklenen formatlar: TXT, PDF, DOC, DOCX
        </p>
        """
    )

    with gr.Tabs() as tabs:

        # =============================================================
        # SEKME 1: DOSYA YÜKLEME
        # =============================================================
        with gr.Tab("📤 Dosya Yükleme", id="upload_tab"):

            gr.Markdown(
                "### 📁 Doküman Yükleme\n"
                "Birden fazla dosya seçip aynı anda yükleyebilirsiniz. "
                "Her dosya maksimum **10 MB** olabilir."
            )

            file_input = gr.File(
                label="Dosyalarınızı buraya sürükleyin veya seçin",
                file_types=[".txt", ".pdf", ".doc", ".docx"],
                file_count="multiple",
                height=180,
            )

            upload_btn = gr.Button(
                "📤 Dokümanları Yükle",
                variant="primary",
                size="lg",
            )

            upload_result = gr.Textbox(
                label="Yükleme Sonucu",
                interactive=False,
                lines=5,
                placeholder="Dosyaları seçip 'Dokümanları Yükle' butonuna tıklayın...",
            )

            upload_btn.click(
                fn=upload_files,
                inputs=[file_input],
                outputs=[upload_result],
                show_progress="full",
            )

        # =============================================================
        # SEKME 2: SOHBET
        # =============================================================
        with gr.Tab("💬 Sohbet", id="chat_tab"):

            chatbot = gr.Chatbot(
                label="Konuşma",
                height=480,
                type="tuples",
                placeholder=(
                    "Henüz bir konuşma yok.\n"
                    "Önce doküman yükleyin, sonra sorularınızı sorun! 💡"
                ),
                show_copy_button=True,
            )

            with gr.Row():
                chat_input = gr.Textbox(
                    label="Sorunuz",
                    placeholder="Dokümanlarınız hakkında bir soru yazın...",
                    scale=5,
                    lines=1,
                    max_lines=3,
                )
                ask_btn = gr.Button(
                    "Soru Sor 🚀",
                    variant="primary",
                    scale=1,
                    min_width=120,
                )

            with gr.Row():
                clear_btn = gr.Button(
                    "🗑️ Konuşmayı Temizle",
                    variant="secondary",
                    size="sm",
                )

            # Buton ile soru sor
            ask_btn.click(
                fn=chat_ask,
                inputs=[chat_input, chatbot, server_history],
                outputs=[chatbot, chat_input, server_history],
                show_progress="minimal",
            )
            # Enter ile soru sor
            chat_input.submit(
                fn=chat_ask,
                inputs=[chat_input, chatbot, server_history],
                outputs=[chatbot, chat_input, server_history],
                show_progress="minimal",
            )
            # Konuşmayı temizle
            clear_btn.click(
                fn=clear_chat,
                outputs=[chatbot, chat_input, server_history],
            )

        # =============================================================
        # SEKME 3: ÖZET
        # =============================================================
        with gr.Tab("📝 Özet", id="summary_tab"):

            gr.Markdown(
                "### 📝 Doküman Özetleme\n"
                "Belirli bir dokümanı veya tüm dokümanları "
                "madde madde Türkçe özetleyin."
            )

            with gr.Row():
                doc_dropdown = gr.Dropdown(
                    label="Doküman seçin",
                    choices=["(Tümünü özetle)"],
                    value="(Tümünü özetle)",
                    interactive=True,
                    scale=4,
                )
                refresh_btn = gr.Button(
                    "🔄 Güncelle",
                    scale=1,
                    min_width=100,
                )

            summarize_btn = gr.Button(
                "📝 Özetle",
                variant="primary",
                size="lg",
            )

            summary_output = gr.Markdown(
                label="Özet Sonucu",
                value="*Doküman seçip 'Özetle' butonuna tıklayın...*",
            )

            # Dropdown'ı güncelle
            refresh_btn.click(
                fn=refresh_doc_dropdown,
                outputs=[doc_dropdown],
            )

            # Özetleme
            summarize_btn.click(
                fn=summarize,
                inputs=[doc_dropdown],
                outputs=[summary_output],
                show_progress="full",
            )

        # =============================================================
        # SEKME 4: DOKÜMAN LİSTESİ
        # =============================================================
        with gr.Tab("📂 Doküman Listesi", id="docs_tab"):

            gr.Markdown("### 📂 Yüklü Dokümanlar")

            doc_table = gr.Dataframe(
                headers=["#", "Doküman Adı"],
                datatype=["str", "str"],
                value=fetch_document_table,
                interactive=False,
                wrap=True,
                height=350,
            )

            refresh_table_btn = gr.Button(
                "🔄 Listeyi Yenile",
                variant="secondary",
            )

            refresh_table_btn.click(
                fn=fetch_document_table,
                outputs=[doc_table],
            )

            # ── Tehlikeli bölge ──
            gr.Markdown("---")

            with gr.Accordion("⚠️ Tehlikeli Bölge", open=False):
                gr.Markdown(
                    "Bu işlem **tüm yüklü dokümanları ve verileri kalıcı olarak siler**. "
                    "Geri alınamaz!"
                )
                delete_btn = gr.Button(
                    "🗑️ Tüm Dokümanları Temizle",
                    variant="stop",
                )
                delete_result = gr.Textbox(
                    label="Sonuç",
                    interactive=False,
                )

                delete_btn.click(
                    fn=delete_all_documents,
                    outputs=[delete_result, doc_table],
                )

    # ── Alt bilgi: Sağlık kontrolü ──
    with gr.Accordion("🏥 Sistem Durumu", open=False):
        with gr.Row():
            health_output = gr.Textbox(
                label="Durum",
                interactive=False,
                scale=4,
            )
            health_btn = gr.Button(
                "Kontrol Et",
                variant="secondary",
                scale=1,
            )
            health_btn.click(
                fn=health_check,
                outputs=[health_output],
            )


# =====================================================================
# UYGULAMA BAŞLATMA
# =====================================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
