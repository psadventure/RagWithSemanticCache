"""
Golden questions against the real corpus in data/ (fictional Kestrel docs).
They pin the facts and lessons the corpus teaches, so editing the docs can't silently
remove them. Retrieval runs offline with the hashing embedder.
"""
import asyncio

from rag import config, service

API, OVERVIEW, TIMELINE = "kestrel_api_reference.md", "kestrel_platform_overview.md", "project_heron_timeline.md"

# (question, doc that answers it, a fact only the answering chunk contains)
GOLDEN = [
    ("How long is an access token valid?", API, "3600 seconds"),
    ("How do I verify the webhook signature and prevent replay attacks?", API, "more than 300 seconds old"),
    ("What should I do when I hit the rate limit and get HTTP 429?", API, "Retry-After"),
    ("Can I cancel a transfer that is already processing?", API, "only while it is `pending`"),
    ("What does error idempotency_conflict mean?", API, "reused with a different body"),
    ("Why did the Fraud Shield pilot start late?", TIMELINE, "fraud history"),
    ("What is the rollback decision point during cutover weekend?", TIMELINE, "Saturday 20:00"),
    ("Which loan applications are auto-approved by credit score and debt-to-income?", OVERVIEW, "700 or higher"),
    ("When does the end-of-day close run and how is interest accrued?", OVERVIEW, "23:30 Eastern"),
    ("How many points does the dormant reactivation rule add?", OVERVIEW, "+350"),
    ("What was the reconciliation variance in each mock conversion?", TIMELINE, "$41,207.13"),
    ("Why was go-live moved to October?", TIMELINE, "PCI DSS audit found"),
]


def _corpus() -> str:
    return "".join(p.read_text(encoding="utf-8") for p in sorted(config.DATA_PATH.glob("*.md")))


def test_corpus_is_three_fictional_docs():
    docs = sorted(config.DATA_PATH.glob("*.md"))
    assert len(docs) == 3
    assert all("Fictional sample document" in p.read_text(encoding="utf-8") for p in docs)


def test_every_golden_fact_exists():
    text = _corpus()
    assert [f for _, _, f in GOLDEN if f not in text] == []


def test_planted_lessons_exist():
    text = _corpus()
    assert "September 2026" in text                      # stale go-live in the overview
    assert "17–18 October 2026" in text                  # D-11 moved it
    assert "price" not in text.lower() and "pricing" not in text.lower()  # licence cost is unanswerable


def test_golden_questions_retrieve_the_expected_doc(offline):
    async def run():
        found = 0
        for question, doc, _ in GOLDEN:
            results = (await service.search(question, top_k=3))["results"]
            found += any(r["doc"] == doc for r in results)
        return found

    found = asyncio.run(run())
    assert found >= len(GOLDEN) - 1, f"expected doc in top-3 for {found}/{len(GOLDEN)}"


def test_search_results_carry_doc_section_sources(offline):
    results = asyncio.run(service.search("How long is an access token valid?", top_k=3))["results"]
    assert len(results) == 3
    for r in results:
        assert r["source"] == f"{r['doc']} | {r['section']}"
