"""
FastAPI backend: a thin HTTP layer over rag/service.py.

Run from the project root:
    python -m uvicorn app.backend:app --reload
"""
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag import config, service
from rag.store import NotIngestedError


@asynccontextmanager
async def lifespan(app: FastAPI):
    await service.startup()
    print(f"✅ Ready: gateway {config.LITELLM_BASE_URL}, chat {config.LLM_MODEL!r}, "
          f"embeddings {config.EMBEDDING_MODEL!r}, cache '{config.SEMANTIC_CACHE_NAME}' "
          f"@ {config.VALKEY_HOST}:{config.VALKEY_PORT} (threshold {config.SEMANTIC_CACHE_THRESHOLD})")
    yield
    await service.shutdown()


app = FastAPI(title="RagWithSemanticCache", lifespan=lifespan)


class Question(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


async def _call(coro):
    """Map service errors to HTTP errors."""
    try:
        return await coro
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except NotIngestedError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"{type(exc).__name__}: {exc}")


@app.get("/")
def root():
    return {"status": "Backend is running", "endpoints": ["/chat", "/search", "/cache/check", "/stats", "/cache", "/health"]}


@app.post("/chat")
async def chat(req: Question):
    result = await _call(service.answer(req.question, req.top_k))
    tag = "CACHE HIT " if result["cache_hit"] else "CACHE MISS"
    print(f"[{tag}] distance={result['distance']} threshold={result['threshold']} {req.question!r}")
    return result


@app.post("/search")
async def search(req: Question):
    return await _call(service.search(req.question, req.top_k))


@app.post("/cache/check")
async def cache_check(req: Question):
    return await _call(service.check_cache(req.question))


@app.get("/stats")
async def stats():
    return await _call(service.cache_stats())


@app.delete("/cache")
async def clear_cache():
    return await _call(service.clear_cache())


@app.get("/health")
async def health():
    try:
        valkey_ok = await (await service.get_cache()).ping()
    except Exception:
        valkey_ok = False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{config.LITELLM_BASE_URL.removesuffix('/v1')}/health/liveliness")
        gateway_ok = r.status_code == 200
    except httpx.HTTPError:
        gateway_ok = False
    return {"status": "ok" if valkey_ok and gateway_ok else "degraded", "valkey": valkey_ok, "litellm_gateway": gateway_ok}
