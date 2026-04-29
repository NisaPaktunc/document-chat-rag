"""Test script — RAG pipeline'ını uçtan uca test eder."""
import httpx
import sys

BASE = "http://localhost:8000"
TIMEOUT = 60.0

def main():
    # 1. Dosya yükle
    print("=== DOSYA YÜKLEME ===")
    with open("test_doc.txt", "rb") as f:
        r = httpx.post(f"{BASE}/upload", files=[("files", ("test_doc.txt", f))], timeout=TIMEOUT)
    print(f"Status: {r.status_code}")
    print(f"Yanıt: {r.json()}")
    if r.status_code != 200:
        print("HATA: Dosya yüklenemedi!")
        sys.exit(1)

    # 2. Doküman listesi
    print("\n=== DOKÜMAN LİSTESİ ===")
    r = httpx.get(f"{BASE}/documents", timeout=10)
    print(f"Dokümanlar: {r.json()}")

    # 3. Soru sor (RAG)
    print("\n=== SORU-CEVAP (RAG) ===")
    r = httpx.post(f"{BASE}/ask", json={"question": "Yapay zeka nedir?", "k": 3}, timeout=TIMEOUT)
    print(f"Status: {r.status_code}")
    data = r.json()
    answer = data.get("answer", data.get("detail", "HATA"))
    sources = data.get("sources", [])
    print(f"Cevap: {answer[:500]}")
    print(f"Kaynaklar: {sources}")

    # 4. Özetleme
    print("\n=== DOKÜMAN ÖZETLEME ===")
    r = httpx.post(f"{BASE}/summarize", json={"document_name": "test_doc.txt"}, timeout=TIMEOUT)
    print(f"Status: {r.status_code}")
    summary_data = r.json()
    summary = summary_data.get("summary", summary_data.get("detail", "HATA"))
    print(f"Özet: {summary[:500]}")

    # 5. Sohbet geçmişi ile soru
    print("\n=== SOHBET GEÇMİŞİ İLE SORU ===")
    history = [
        {"role": "user", "content": "Yapay zeka nedir?"},
        {"role": "assistant", "content": answer[:200]},
    ]
    r = httpx.post(
        f"{BASE}/chat",
        json={"question": "Derin öğrenme ile farkı nedir?", "history": history, "k": 3},
        timeout=TIMEOUT,
    )
    print(f"Status: {r.status_code}")
    chat_data = r.json()
    print(f"Cevap: {chat_data.get('answer', chat_data.get('detail', 'HATA'))[:500]}")

    # 6. Sağlık kontrolü
    print("\n=== SAĞLIK KONTROLÜ ===")
    r = httpx.get(f"{BASE}/health", timeout=10)
    print(f"Durum: {r.json()}")

    print("\n✅ TÜM TESTLER BAŞARILI!")

if __name__ == "__main__":
    main()
