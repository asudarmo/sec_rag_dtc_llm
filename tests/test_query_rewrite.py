from rag import query_rewrite as qr


def test_rewrite_falls_back_to_original_query_when_no_api_key(monkeypatch):
    monkeypatch.setattr(qr, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = qr.rewrite("What are Google's AI risks?")

    assert result == [{"query": "What are Google's AI risks?", "filter_dict": None}]
