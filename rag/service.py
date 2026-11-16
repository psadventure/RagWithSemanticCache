"""
The single API used by both the FastAPI backend and the ADK agent.

    answer(question)        cache check → retrieve → Gemini → cache store
    search(question)        retrieval only, no LLM, no cache
    check_cache(question)   would this question hit the cache? (counts in the stats)
    cache_stats()           hits, misses, hit rate, cost saved, entries
    clear_cache()           drop every cached answer and reset the stats

Everything is async. Functions raise on failure; callers decide how to show the error
(HTTP status in FastAPI, {"error": ...} in agent tools).

Similarity convention: BetterDB returns a cosine DISTANCE (0 = identical). Results expose
both `distance` and `similarity = 1 - distance`; a hit means distance <= threshold.
"""
import asyncio

from rag import config, gateway
from rag.cache import BetterDBCache
from rag.store import NotIngestedError, VectorStore  # noqa: F401  (re-exported for callers)

SYSTEM_PROMPT = """You answer questions about the Kestrel platform and Project Heron using ONLY the
context below. The context is numbered excerpts, each labelled "doc | section".

Rules:
- If the context does not contain the answer, say you can't find it in the documents. Do not guess
  or use outside knowledge (e.g. prices, people or dates that aren't in the context).
- If excerpts disagree, prefer the most recent decision or dated entry and point out the conflict.
- Keep answers short and cite the excerpts you used like [1], [2].

Context:
{context}"""

# ============================================================================
# Dependencies (lazy, and swappable in tests)
# ============================================================================

_embedder = None
_store: VectorStore | None = None
_cache: tuple[asyncio.AbstractEventLoop, BetterDBCache] | None = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = gateway.GatewayEmbedder()
    return _embedder


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore.load(config.VECTOR_STORE_PATH)
    return _store


async def get_cache() -> BetterDBCache:
    """One cache per event loop: the async Valkey client is bound to the loop it runs on."""
    global _cache
    loop = asyncio.get_running_loop()
    if _cache is None or _cache[0] is not loop:
        cache = BetterDBCache(embed_fn=get_embedder().aembed)
        await cache.initialize()
        _cache = (loop, cache)
    return _cache[1]


async def chat(messages: list[dict]) -> tuple[str, int, int]:
    return await gateway.achat(messages)


async def startup() -> None:
    """Fail fast at server start: load the index and connect the cache."""
    get_store()
    await get_cache()


async def shutdown() -> None:
    global _cache
    if _cache is not None and _cache[0] is asyncio.get_running_loop():
        await _cache[1].close()
    _cache = None


def reset() -> None:
    """Forget loaded dependencies (e.g. after re-ingesting)."""
    global _embedder, _store, _cache
    _embedder = _store = _cache = None


# ============================================================================
# Helpers
# ============================================================================


def _question(q: str) -> str:
    q = (q or "").strip()
    if not q:
        raise ValueError("The question is empty.")
    return q


def _round(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


def _source(chunk: dict) -> str:
    return f"{chunk['doc']} | {chunk['section']}"


def _cache_view(distance: float | None, threshold: float) -> dict:
    if distance is not None:
        distance = max(0.0, distance)  # float noise can give -0.0 for identical questions
    return {
        "distance": _round(distance),
        "similarity": None if distance is None else _round(1 - distance),
        "threshold": threshold,
    }


async def _retrieve(question: str, top_k: int) -> list[dict]:
    store = get_store()
    vector = await get_embedder().aembed(question)
    return [
        {"rank": i, "source": _source(c), "doc": c["doc"], "section": c["section"],
         "score": _round(score), "text": c["text"]}
        for i, (c, score) in enumerate(store.search(vector, top_k), 1)
    ]


# ============================================================================
# Public API
# ============================================================================


async def search(question: str, top_k: int | None = None) -> dict:
    """Top-k chunks by cosine similarity. No LLM call and no cache."""
    question = _question(question)
    return {"question": question, "results": await _retrieve(question, top_k or config.TOP_K)}


async def answer(question: str, top_k: int | None = None) -> dict:
    question = _question(question)
    cache = await get_cache()

    # 1. Semantic cache
    hit = await cache.check(question)
    if hit.hit:
        meta = hit.metadata or {}
        return {
            "question": question,
            "answer": hit.response,
            "sources": meta.get("sources", []),
            "cache_hit": True,
            "cached_question": hit.cached_question,
            **_cache_view(hit.distance, cache.threshold),
        }

    # 2. Retrieve
    chunks = await _retrieve(question, top_k or config.TOP_K)
    context = "\n\n".join(f"[{c['rank']}] {c['source']}\n{c['text']}" for c in chunks)

    # 3. Generate
    text, input_tokens, output_tokens = await chat([
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": question},
    ])
    sources = list(dict.fromkeys(c["source"] for c in chunks))

    # 4. Store, with token counts so BetterDB can price the calls a future hit saves
    await cache.store(
        question, text,
        model=config.CACHE_COST_MODEL,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        metadata={"sources": sources},
    )
    return {
        "question": question,
        "answer": text,
        "sources": sources,
        "cache_hit": False,
        "cached_question": None,
        # On a miss, distance/similarity describe the closest cached question (None = cache was empty).
        **_cache_view(hit.distance, cache.threshold),
        "tokens": {"input": input_tokens, "output": output_tokens},
    }


async def check_cache(question: str) -> dict:
    """Would this question hit? Returns the closest cached question's distance either way.
    Note: BetterDB counts this lookup as a hit or miss in the stats."""
    question = _question(question)
    cache = await get_cache()
    r = await cache.check(question)
    return {
        "question": question,
        "hit": r.hit,
        "cached_question": r.cached_question,
        "cached_answer": r.response,
        "cache_empty": r.distance is None,
        **_cache_view(r.distance, cache.threshold),
    }


async def cache_stats() -> dict:
    cache = await get_cache()
    s = await cache.stats()
    return {
        **s,
        "hit_rate": f"{s['hit_rate']:.1%}",
        "cost_saved_usd": f"${s['cost_saved_usd']:.6f}",
        "cache_name": cache.name,
        "threshold": cache.threshold,
        "cost_model": config.CACHE_COST_MODEL,
    }


async def clear_cache() -> dict:
    cache = await get_cache()
    before = (await cache.stats()).get("entries")
    await cache.clear()
    return {"cleared": True, "entries_removed": before, "cache_name": cache.name}
