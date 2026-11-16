from fastapi.testclient import TestClient

from app import backend
from rag import service


def _client(monkeypatch):
    async def noop():
        return None

    monkeypatch.setattr(service, "startup", noop)
    monkeypatch.setattr(service, "shutdown", noop)
    return TestClient(backend.app)


def test_chat_miss_then_hit_and_stats(offline, monkeypatch):
    with _client(monkeypatch) as client:
        q = {"question": "How long is an access token valid?"}
        assert client.post("/chat", json=q).json()["cache_hit"] is False
        assert client.post("/chat", json=q).json()["cache_hit"] is True
        stats = client.get("/stats").json()
        assert stats["hits"] == 1 and stats["entries"] == 1
        assert client.delete("/cache").json()["cleared"] is True


def test_errors_map_to_http_status(offline, monkeypatch):
    async def down(*a, **k):
        raise ConnectionError("gateway down")

    with _client(monkeypatch) as client:
        assert client.post("/chat", json={"question": ""}).status_code == 422
        monkeypatch.setattr(service, "answer", down)
        r = client.post("/chat", json={"question": "hi"})
        assert r.status_code == 502 and "gateway down" in r.json()["detail"]
