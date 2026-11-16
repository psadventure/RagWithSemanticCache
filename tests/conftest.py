"""
Offline fixtures: a deterministic hashing embedder and an in-memory semantic cache, so the
whole service runs without Docker, Valkey or the LiteLLM gateway.
"""
import hashlib
import math
import re

import pytest

from rag import config, service
from rag.cache import CheckResult
from rag.ingest import ingest

STOPWORDS = set("a an the is are was were be of to in on for and or how what why when which who do does did "
                "i my me can should it its this that with at by from as".split())


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9_]+", text.lower())
    return [w.rstrip("s") for w in words if w not in STOPWORDS]


class HashEmbedder:
    """Bag-of-words hashing embedder: same words → same vector; no network."""

    model = "hash-256"

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.calls = 0

    def vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in _tokens(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed(self, texts, progress=None):
        self.calls += len(texts)
        return [self.vec(t) for t in texts]

    async def aembed(self, text):
        self.calls += 1
        return self.vec(text)


def _cosine_distance(a, b) -> float:
    return 1.0 - sum(x * y for x, y in zip(a, b))


class FakeCache:
    """In-memory stand-in for rag.cache.BetterDBCache (same methods, same semantics)."""

    name = "test-cache"

    def __init__(self, embedder: HashEmbedder, threshold: float = 0.1):
        self.embedder = embedder
        self.threshold = threshold
        self.entries: list[dict] = []
        self.hits = self.misses = 0
        self.cost_saved_micros = 0

    async def initialize(self):
        pass

    async def check(self, question):
        if not self.entries:
            self.misses += 1
            return CheckResult(hit=False, distance=None)
        q = await self.embedder.aembed(question)
        best = min(self.entries, key=lambda e: _cosine_distance(q, e["vec"]))
        d = _cosine_distance(q, best["vec"])
        if d <= self.threshold:
            self.hits += 1
            self.cost_saved_micros += best["cost"]
            return CheckResult(True, d, best["answer"], best["question"], best["metadata"])
        self.misses += 1
        return CheckResult(hit=False, distance=d)

    async def store(self, question, answer, *, model, input_tokens, output_tokens, metadata):
        self.entries.append({
            "question": question, "answer": answer, "metadata": metadata, "model": model,
            "vec": await self.embedder.aembed(question), "cost": input_tokens + output_tokens,
        })

    async def stats(self):
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "total": total,
                "hit_rate": self.hits / total if total else 0.0,
                "cost_saved_usd": self.cost_saved_micros / 1_000_000, "entries": len(self.entries)}

    async def clear(self):
        self.entries.clear()
        self.hits = self.misses = self.cost_saved_micros = 0

    async def ping(self):
        return True

    async def close(self):
        pass


class FakeLLM:
    def __init__(self):
        self.calls: list[list[dict]] = []

    async def __call__(self, messages):
        self.calls.append(messages)
        return f"ANSWER to: {messages[-1]['content']}", 100, 20


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def offline(tmp_path, monkeypatch, embedder):
    """Real data/ ingested with the fake embedder into a temp store; service wired to fakes."""
    store_dir = tmp_path / "vector_store"
    ingest(embedder, data_dir=config.DATA_PATH, out_dir=store_dir)
    monkeypatch.setattr(config, "VECTOR_STORE_PATH", store_dir)
    service.reset()

    cache, llm = FakeCache(embedder), FakeLLM()

    async def get_cache():
        return cache

    monkeypatch.setattr(service, "get_embedder", lambda: embedder)
    monkeypatch.setattr(service, "get_cache", get_cache)
    monkeypatch.setattr(service, "chat", llm)
    yield type("Offline", (), {"cache": cache, "llm": llm, "embedder": embedder, "store_dir": store_dir})
    service.reset()
