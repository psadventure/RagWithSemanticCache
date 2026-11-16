"""
ADK tool wrappers around rag/service.py.

Each tool returns the service result, or {"error": "..."} instead of raising, so the model
can read the problem and explain it (e.g. "run ingestion first", "start Docker").
Docstrings and type hints become the tool schema the model sees.
"""
import functools
from typing import Optional

from rag import service


def _safe(fn):
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
    return wrapper


@_safe
async def ask(question: str) -> dict:
    """
    Answer a question about the Kestrel documents with RAG behind the semantic cache.
    First checks the cache; on a hit returns the cached answer without calling the LLM.
    On a miss retrieves chunks, asks Gemini, and stores the answer in the cache.
    Returns: answer, sources ("doc | section"), cache_hit, cached_question (on a hit),
    distance (cosine distance to the closest cached question, None if the cache was empty),
    similarity (= 1 - distance) and threshold (a hit needs distance <= threshold).
    """
    return await service.answer(question)


@_safe
async def search_chunks(question: str, top_k: Optional[int] = None) -> dict:
    """
    Retrieve the top_k most similar document chunks for a question, without calling the
    LLM or the cache. Each result has rank, source ("doc | section"), cosine score and text.
    Use it to show what the retriever finds or to debug a wrong answer.
    """
    return await service.search(question, top_k)


@_safe
async def check_cache(question: str) -> dict:
    """
    Check whether a question would be a cache hit, without answering it. Returns hit,
    the closest cached_question and its cached_answer, distance, similarity and threshold.
    Note: the lookup itself counts as a hit or miss in cache_stats.
    """
    return await service.check_cache(question)


@_safe
async def cache_stats() -> dict:
    """
    Semantic cache statistics: hits, misses, total lookups, hit_rate, cost_saved_usd
    (LLM cost avoided by hits), entries (cached answers), cache_name, threshold, cost_model.
    """
    return await service.cache_stats()


@_safe
async def clear_cache() -> dict:
    """Delete every cached answer and reset the cache statistics. Documents are untouched."""
    return await service.clear_cache()
