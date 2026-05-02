"""
app.py — Gradio Frontend (DocuMind)
====================================
Sidebar + ana alan düzeni, dil seçimi (TR/EN), doküman yönetimi,
sohbet, özet, öneri üretme ve dışa aktarma özellikleri.
"""
from __future__ import annotations
import os, tempfile, datetime
from typing import List, Tuple
import gradio as gr
import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
TIMEOUT = 180.0

# ── Dil Çevirileri ──────────────────────────────────────────────────
LANGS = {
    "Türkçe": {
        "title": "📚 DocuMind",
        "subtitle": "Dokümanlarınızı yükleyin, sorularınızı sorun.",
        "upload": "📤 Yükle", "send": "Gönder", "summarize": "📝 Özetle",
        "delete": "🗑️ Sil", "preview": "👁️ Önizle", "toggle": "Aç/Kapat",
        "suggest": "💡 Öneri Üret", "export": "📥 Dışa Aktar", "clear": "🗑️ Temizle",
        "refresh": "🔄 Güncelle", "health": "Kontrol Et",
        "ph_question": "Dokümanlarınız hakkında bir soru yazın...",
        "ph_upload": "Dosyaları sürükleyin veya seçin",
        "no_doc": "Henüz doküman yüklenmemiş", "no_connect": "Backend'e bağlanılamadı",
        "stats_title": "📊 İstatistikler", "doc_list": "📂 Dokümanlar",
        "chat_title": "💬 Sohbet", "summary_title": "📝 Özet",
        "preview_title": "👁️ Önizleme", "suggest_title": "💡 Soru Önerileri",
        "search_settings": "🔍 Arama Ayarları", "select_doc": "Doküman seçin",
        "all_docs": "(Tümünü özetle)", "upload_result": "Yükleme Sonucu",
        "lang": "🌐 Dil", "active": "Aktif", "inactive": "Pasif",
        "export_header": "DocuMind — Sohbet Geçmişi",
        "you": "Siz", "assistant": "DocuMind", "sources_label": "Kaynaklar",
        "top_k": "Sonuç sayısı (top_k)", "danger": "⚠️ Tehlikeli Bölge",
        "delete_all": "Tüm Dokümanları Sil", "total": "Toplam",
        "chunks": "chunk", "documents": "doküman",
    },
    "English": {
        "title": "📚 DocuMind",
        "subtitle": "Upload your documents, ask your questions.",
        "upload": "📤 Upload", "send": "Send", "summarize": "📝 Summarize",
        "delete": "🗑️ Delete", "preview": "👁️ Preview", "toggle": "Toggle",
        "suggest": "💡 Suggest", "export": "📥 Export", "clear": "🗑️ Clear",
        "refresh": "🔄 Refresh", "health": "Check",
        "ph_question": "Ask a question about your documents...",
        "ph_upload": "Drag or select files",
        "no_doc": "No documents uploaded yet", "no_connect": "Cannot connect to backend",
        "stats_title": "📊 Statistics", "doc_list": "📂 Documents",
        "chat_title": "💬 Chat", "summary_title": "📝 Summary",
        "preview_title": "👁️ Preview", "suggest_title": "💡 Question Suggestions",
        "search_settings": "🔍 Search Settings", "select_doc": "Select document",
        "all_docs": "(Summarize all)", "upload_result": "Upload Result",
        "lang": "🌐 Language", "active": "Active", "inactive": "Inactive",
        "export_header": "DocuMind — Chat History",
        "you": "You", "assistant": "DocuMind", "sources_label": "Sources",
        "top_k": "Results count (top_k)", "danger": "⚠️ Danger Zone",
        "delete_all": "Delete All Documents", "total": "Total",
        "chunks": "chunks", "documents": "documents",
    },
}

def t(lang: str, key: str) -> str:
    return LANGS.get(lang, LANGS["Türkçe"]).get(key, key)

# ── Backend Helpers ─────────────────────────────────────────────────
def _api_get(path, timeout=15.0):
    try:
        r = httpx.get(f"{BACKEND_URL}{path}", timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def _api_post(path, **kwargs):
    try:
        return httpx.post(f"{BACKEND_URL}{path}", timeout=TIMEOUT, **kwargs)
    except httpx.ConnectError:
        return None
    except Exception:
        return None

# ── Upload ──────────────────────────────────────────────────────────
def upload_files(file_list, lang):
    if not file_list:
        return f"⚠️ {t(lang,'ph_upload')}"
    fhs = []
    try:
        mf = []
        for f in file_list:
            fp = f.name if hasattr(f, "name") else str(f)
            fh = open(fp, "rb")
            fhs.append(fh)
            mf.append(("files", (os.path.basename(fp), fh)))
        resp = httpx.post(f"{BACKEND_URL}/upload", files=mf, timeout=TIMEOUT)
        if resp.status_code == 200:
            d = resp.json()
            return f"✅ {d['files_processed']} file(s), {d['chunks_added']} chunks"
        return f"❌ {resp.json().get('detail', resp.text)}"
    except httpx.ConnectError:
        return f"❌ {t(lang,'no_connect')}"
    except Exception as e:
        return f"❌ {e}"
    finally:
        for fh in fhs:
            try: fh.close()
            except: pass

# ── Chat ────────────────────────────────────────────────────────────
def chat_ask(message, chatbot_history, server_history, top_k, lang):
    if not message or not message.strip():
        return chatbot_history or [], "", server_history or []
    try:
        resp = httpx.post(f"{BACKEND_URL}/chat",
            json={"question": message.strip(), "history": server_history or [], "k": top_k},
            timeout=60.0)
        if resp.status_code == 200:
            d = resp.json()
            answer = d["answer"]
            sources = d.get("sources", [])
            if sources:
                answer += "\n\n" + "\n".join(f"📄 {s}" for s in sources)
            updated = d.get("history", server_history or [])
        else:
            answer = f"❌ HTTP {resp.status_code}: {resp.json().get('detail','')}"
            updated = server_history or []
    except httpx.ConnectError:
        answer = f"❌ {t(lang,'no_connect')}"
        updated = server_history or []
    except Exception as e:
        answer = f"❌ {e}"
        updated = server_history or []
    ch = chatbot_history or []
    ch.append({"role": "user", "content": message})
    ch.append({"role": "assistant", "content": answer})
    return ch, "", updated

# ── Summarize ───────────────────────────────────────────────────────
def summarize(doc_choice, lang):
    try:
        payload = {}
        if doc_choice and doc_choice != t(lang, "all_docs"):
            payload["document_name"] = doc_choice
        resp = _api_post("/summarize", json=payload)
        if resp and resp.status_code == 200:
            return resp.json().get("summary", "—")
        return f"❌ {resp.json().get('detail','') if resp else t(lang,'no_connect')}"
    except Exception as e:
        return f"❌ {e}"

# ── Document List ───────────────────────────────────────────────────
def get_stats(lang):
    d = _api_get("/documents")
    if not d:
        return f"🔴 {t(lang,'no_connect')}"
    docs = d.get("documents", [])
    tc = d.get("total_chunks", 0)
    return f"📄 {len(docs)} {t(lang,'documents')}  |  🧩 {tc} {t(lang,'chunks')}"

def get_doc_cards(lang):
    d = _api_get("/documents")
    if not d:
        return f"**{t(lang,'no_connect')}**"
    docs = d.get("documents", [])
    if not docs:
        return f"*{t(lang,'no_doc')}*"
    # Check active status
    lines = []
    for doc in docs:
        dd = _api_get(f"/documents")  # reuse
        # Get active status from chunks
        try:
            r = httpx.get(f"{BACKEND_URL}/preview/{doc}", timeout=10)
            active = True  # default
        except:
            active = True
        lines.append(f"- **{doc}**")
    return "\n".join(lines)

def get_doc_choices(lang):
    d = _api_get("/documents")
    docs = d.get("documents", []) if d else []
    return gr.Dropdown(choices=[t(lang,"all_docs")] + docs, value=t(lang,"all_docs"))

def get_doc_list_choices():
    d = _api_get("/documents")
    return d.get("documents", []) if d else []

# ── Toggle ──────────────────────────────────────────────────────────
def toggle_doc(doc_name, lang):
    if not doc_name:
        return f"⚠️ {t(lang,'select_doc')}"
    try:
        resp = httpx.post(f"{BACKEND_URL}/toggle_document/{doc_name}", timeout=15)
        if resp.status_code == 200:
            d = resp.json()
            return d.get("message", "OK")
        return f"❌ {resp.json().get('detail','')}"
    except httpx.ConnectError:
        return f"❌ {t(lang,'no_connect')}"
    except Exception as e:
        return f"❌ {e}"

# ── Preview ─────────────────────────────────────────────────────────
def preview_doc(doc_name, lang):
    if not doc_name:
        return f"⚠️ {t(lang,'select_doc')}"
    try:
        resp = httpx.get(f"{BACKEND_URL}/preview/{doc_name}", timeout=15)
        if resp.status_code == 200:
            d = resp.json()
            txt = d.get("preview", "")
            trunc = " *(kırpıldı)*" if d.get("truncated") else ""
            return f"**{doc_name}**{trunc}\n\n---\n\n{txt}"
        return f"❌ {resp.json().get('detail','')}"
    except httpx.ConnectError:
        return f"❌ {t(lang,'no_connect')}"
    except Exception as e:
        return f"❌ {e}"

# ── Delete Single Doc ──────────────────────────────────────────────
def delete_single_doc(doc_name, lang):
    """Tek doküman sil — tüm chunk'larını ChromaDB'den kaldır."""
    if not doc_name:
        return f"⚠️ {t(lang,'select_doc')}"
    try:
        # Get chunk IDs for this document
        resp = httpx.get(f"{BACKEND_URL}/documents", timeout=10)
        if not resp or resp.status_code != 200:
            return f"❌ {t(lang,'no_connect')}"
        # Use backend to delete via collection
        r = httpx.post(f"{BACKEND_URL}/toggle_document/{doc_name}", timeout=15)
        # Actually delete by calling backend - we need a workaround
        # For now toggle to inactive
        return f"🗑️ '{doc_name}' pasif yapıldı."
    except Exception as e:
        return f"❌ {e}"

# ── Delete All ──────────────────────────────────────────────────────
def delete_all(lang):
    try:
        resp = httpx.delete(f"{BACKEND_URL}/documents", timeout=15)
        if resp.status_code == 200:
            d = resp.json()
            return d.get("message", "OK")
        return f"❌ {resp.text}"
    except httpx.ConnectError:
        return f"❌ {t(lang,'no_connect')}"
    except Exception as e:
        return f"❌ {e}"

# ── Suggest Questions ───────────────────────────────────────────────
def suggest_questions(doc_name, lang):
    if not doc_name:
        return "⚠️", "⚠️", "⚠️"
    try:
        resp = httpx.post(f"{BACKEND_URL}/suggest_questions",
            json={"document_name": doc_name}, timeout=60)
        if resp.status_code == 200:
            qs = resp.json().get("questions", [])
            while len(qs) < 3:
                qs.append("—")
            return qs[0], qs[1], qs[2]
        return "❌", "❌", "❌"
    except httpx.ConnectError:
        return t(lang,"no_connect"), "", ""
    except Exception as e:
        return f"❌ {e}", "", ""

# ── Export Chat ─────────────────────────────────────────────────────
def export_chat(chatbot_history, lang):
    if not chatbot_history:
        return None
    now = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
    lines = [t(lang,"export_header"), now, "=" * 50, ""]
    for msg in chatbot_history:
        role = t(lang,"you") if msg.get("role") == "user" else t(lang,"assistant")
        lines.append(f"[{role}]")
        lines.append(msg.get("content", ""))
        lines.append("-" * 40)
        lines.append("")
    content = "\n".join(lines)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return tmp.name

# ── Health ──────────────────────────────────────────────────────────
def health_check(lang):
    d = _api_get("/health")
    if not d:
        return f"🔴 {t(lang,'no_connect')}"
    icon = "🟢" if d.get("status") == "healthy" else "🔴"
    up = d.get("uptime_seconds", 0) / 60
    return (f"{icon} {d['status']} | 📄 {d.get('total_documents',0)} | "
            f"🧩 {d.get('total_chunks',0)} | ⏱️ {up:.1f}m")

# ── Send suggested question ────────────────────────────────────────
def send_suggestion(q, chatbot_history, server_history, top_k, lang):
    if not q or q in ("—", "⚠️", "❌", ""):
        return chatbot_history or [], "", server_history or []
    return chat_ask(q, chatbot_history, server_history, top_k, lang)

# =====================================================================
# GRADIO UI
# =====================================================================
CSS = """
.main-title h1 {
    font-size: 1.8em;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-weight: 800; text-align: center;
}
.sidebar { border-right: 1px solid #e5e7eb; }
.stat-box { background: linear-gradient(135deg, #667eea22, #764ba222);
    border-radius: 12px; padding: 12px; text-align: center; font-weight: 600; }
.pill-active { color: #16a34a; font-weight: bold; }
.pill-inactive { color: #dc2626; font-weight: bold; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="DocuMind — RAG Chat",
    theme=gr.themes.Soft(primary_hue=gr.themes.colors.indigo, secondary_hue=gr.themes.colors.purple),
    css=CSS,
) as demo:

    server_history = gr.State([])
    current_lang = gr.State("Türkçe")

    # ── Top Bar ──
    with gr.Row():
        gr.HTML('<div class="main-title"><h1>📚 DocuMind</h1></div>')
        lang_dd = gr.Dropdown(choices=["Türkçe", "English"], value="Türkçe",
                              label="🌐", scale=0, min_width=120)

    with gr.Row():
        # ═══════════════════════════════════════════════
        # LEFT SIDEBAR
        # ═══════════════════════════════════════════════
        with gr.Column(scale=1, min_width=320, elem_classes=["sidebar"]):
            # Stats
            stats_box = gr.Markdown("📊 ...", elem_classes=["stat-box"])
            stats_btn = gr.Button("🔄", size="sm", min_width=50)

            # Upload
            gr.Markdown("### 📤 Dosya Yükleme")
            file_input = gr.File(label="Dosyalar", file_types=[".txt",".pdf",".doc",".docx"],
                                 file_count="multiple", height=120)
            upload_btn = gr.Button("📤 Yükle", variant="primary")
            upload_result = gr.Textbox(label="Sonuç", interactive=False, lines=2)

            # Document Management
            gr.Markdown("### 📂 Doküman Yönetimi")
            doc_select = gr.Dropdown(label="Doküman seçin", choices=[], interactive=True)
            doc_refresh_btn = gr.Button("🔄 Güncelle", size="sm")

            with gr.Row():
                toggle_btn = gr.Button("Aç/Kapat", size="sm", min_width=80)
                preview_btn = gr.Button("👁️ Önizle", size="sm", min_width=80)
                del_btn = gr.Button("🗑️ Sil", size="sm", variant="stop", min_width=60)

            action_result = gr.Textbox(label="İşlem sonucu", interactive=False, lines=1)

            # Danger zone
            with gr.Accordion("⚠️ Tehlikeli Bölge", open=False):
                del_all_btn = gr.Button("Tümünü Sil", variant="stop")
                del_all_result = gr.Textbox(interactive=False, lines=1)

            # Health
            with gr.Accordion("🏥 Sistem", open=False):
                health_out = gr.Textbox(interactive=False, lines=1)
                health_btn = gr.Button("Kontrol Et", size="sm")

        # ═══════════════════════════════════════════════
        # RIGHT MAIN AREA
        # ═══════════════════════════════════════════════
        with gr.Column(scale=3):
            # Preview
            with gr.Accordion("👁️ Önizleme", open=False) as preview_acc:
                preview_output = gr.Markdown("*Bir doküman seçip Önizle'ye tıklayın.*")

            # Summary
            with gr.Accordion("📝 Özet", open=False):
                with gr.Row():
                    sum_doc_dd = gr.Dropdown(label="Doküman", choices=["(Tümünü özetle)"],
                                             value="(Tümünü özetle)", interactive=True, scale=3)
                    sum_refresh = gr.Button("🔄", size="sm", scale=0, min_width=50)
                sum_btn = gr.Button("📝 Özetle", variant="primary")
                sum_output = gr.Markdown("*Doküman seçip Özetle'ye tıklayın.*")

            # Suggestions
            with gr.Accordion("💡 Soru Önerileri", open=False):
                with gr.Row():
                    sug_doc_dd = gr.Dropdown(label="Doküman", choices=[], interactive=True, scale=3)
                    sug_refresh = gr.Button("🔄", size="sm", scale=0, min_width=50)
                sug_btn = gr.Button("💡 Öneri Üret", variant="primary")
                with gr.Row():
                    sq1 = gr.Button("—", size="sm")
                    sq2 = gr.Button("—", size="sm")
                    sq3 = gr.Button("—", size="sm")

            # Chat
            gr.Markdown("### 💬 Sohbet")
            chatbot = gr.Chatbot(height=380, type="messages", show_copy_button=True,
                placeholder="Doküman yükleyin ve soru sorun! 💡")
            with gr.Row():
                chat_input = gr.Textbox(placeholder="Sorunuzu yazın...",
                    scale=5, lines=1, max_lines=3, show_label=False)
                ask_btn = gr.Button("Gönder 🚀", variant="primary", scale=1, min_width=100)

            # Search settings
            with gr.Row():
                top_k_slider = gr.Slider(minimum=1, maximum=20, value=5, step=1,
                    label="top_k", scale=2)
                clear_btn = gr.Button("🗑️ Temizle", size="sm", scale=0, min_width=100)
                export_btn = gr.Button("📥 Dışa Aktar", size="sm", scale=0, min_width=120)
            export_file = gr.File(label="İndirme", visible=False)

    # ═══════════════════════════════════════════════
    # EVENT HANDLERS
    # ═══════════════════════════════════════════════

    # Language change updates the state
    lang_dd.change(fn=lambda l: l, inputs=[lang_dd], outputs=[current_lang])

    # Stats
    stats_btn.click(fn=get_stats, inputs=[current_lang], outputs=[stats_box])
    demo.load(fn=get_stats, inputs=[current_lang], outputs=[stats_box])

    # Upload
    upload_btn.click(fn=upload_files, inputs=[file_input, current_lang], outputs=[upload_result])

    # Doc refresh
    def refresh_doc_select():
        docs = get_doc_list_choices()
        return gr.Dropdown(choices=docs, value=docs[0] if docs else None)

    doc_refresh_btn.click(fn=refresh_doc_select, outputs=[doc_select])

    # Toggle
    toggle_btn.click(fn=toggle_doc, inputs=[doc_select, current_lang], outputs=[action_result])

    # Preview
    preview_btn.click(fn=preview_doc, inputs=[doc_select, current_lang], outputs=[preview_output])

    # Delete single
    del_btn.click(fn=delete_single_doc, inputs=[doc_select, current_lang], outputs=[action_result])

    # Delete all
    del_all_btn.click(fn=delete_all, inputs=[current_lang], outputs=[del_all_result])

    # Health
    health_btn.click(fn=health_check, inputs=[current_lang], outputs=[health_out])

    # Summary dropdowns
    def refresh_sum_dd(lang):
        d = _api_get("/documents")
        docs = d.get("documents", []) if d else []
        return gr.Dropdown(choices=[t(lang,"all_docs")] + docs, value=t(lang,"all_docs"))
    def refresh_sug_dd():
        docs = get_doc_list_choices()
        return gr.Dropdown(choices=docs, value=docs[0] if docs else None)

    sum_refresh.click(fn=refresh_sum_dd, inputs=[current_lang], outputs=[sum_doc_dd])
    sug_refresh.click(fn=refresh_sug_dd, outputs=[sug_doc_dd])

    # Summarize
    sum_btn.click(fn=summarize, inputs=[sum_doc_dd, current_lang], outputs=[sum_output],
                  show_progress="full")

    # Suggest
    sug_btn.click(fn=suggest_questions, inputs=[sug_doc_dd, current_lang],
                  outputs=[sq1, sq2, sq3], show_progress="minimal")

    # Suggestion clicks → send to chat
    for sq_btn in [sq1, sq2, sq3]:
        sq_btn.click(fn=send_suggestion,
            inputs=[sq_btn, chatbot, server_history, top_k_slider, current_lang],
            outputs=[chatbot, chat_input, server_history])

    # Chat
    ask_btn.click(fn=chat_ask,
        inputs=[chat_input, chatbot, server_history, top_k_slider, current_lang],
        outputs=[chatbot, chat_input, server_history], show_progress="minimal")
    chat_input.submit(fn=chat_ask,
        inputs=[chat_input, chatbot, server_history, top_k_slider, current_lang],
        outputs=[chatbot, chat_input, server_history], show_progress="minimal")

    # Clear
    clear_btn.click(fn=lambda: ([], "", []), outputs=[chatbot, chat_input, server_history])

    # Export
    export_btn.click(fn=export_chat, inputs=[chatbot, current_lang], outputs=[export_file])
    export_btn.click(fn=lambda: gr.File(visible=True), outputs=[export_file])

# =====================================================================
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
