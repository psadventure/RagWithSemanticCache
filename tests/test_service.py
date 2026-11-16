import asyncio

import pytest

from rag import config, service
from rag.store import NotIngestedError

Q = "How long is an access token valid?"


def run(coro):
    return asyncio.run(coro)


def test_miss_then_exact_repeat_hits(offline):
    first = run(service.answer(Q))
    assert first["cache_hit"] is False
    assert first["distance"] is None            # cache was empty
    assert first["sources"] and all(" | " in s for s in first["sources"])
    assert len(offline.llm.calls) == 1
    assert len(offline.cache.entries) == 1

    second = run(service.answer(Q))
    assert second["cache_hit"] is True
    assert second["answer"] == first["answer"]
    assert second["similarity"] == pytest.approx(1.0)
    assert second["cached_question"] == Q
    assert second["sources"] == first["sources"]  # sources come back from cache metadata
    assert len(offline.llm.calls) == 1            # no second LLM call


def test_different_topic_misses_and_reports_nearest(offline):
    run(service.answer(Q))
    other = run(service.answer("What is the rollback decision point during cutover weekend?"))
    assert other["cache_hit"] is False
    assert other["distance"] > other["threshold"]
    assert other["similarity"] == pytest.approx(1 - other["distance"], abs=1e-3)
    assert len(offline.llm.calls) == 2


def test_prompt_contains_numbered_doc_section_context(offline):
    run(service.answer(Q))
    system = offline.llm.calls[0][0]["content"]
    assert "[1] kestrel_api_reference.md | " in system


def test_check_cache_stats_and_clear(offline):
    assert run(service.check_cache(Q))["cache_empty"] is True
    run(service.answer(Q))
    check = run(service.check_cache(Q))
    assert check["hit"] is True and check["cached_answer"]

    stats = run(service.cache_stats())
    assert stats["entries"] == 1 and stats["hits"] == 1
    assert stats["hit_rate"].endswith("%") and stats["cost_saved_usd"].startswith("$")
    assert stats["threshold"] == offline.cache.threshold

    cleared = run(service.clear_cache())
    assert cleared["cleared"] is True and cleared["entries_removed"] == 1
    assert run(service.cache_stats())["entries"] == 0


def test_empty_question_raises(offline):
    with pytest.raises(ValueError):
        run(service.answer("   "))


def test_missing_index_raises_not_ingested(offline, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VECTOR_STORE_PATH", tmp_path / "nothing-here")
    service.reset()
    with pytest.raises(NotIngestedError):
        run(service.search(Q))
