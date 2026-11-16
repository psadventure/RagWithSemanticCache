from rag import config
from rag.ingest import chunk_corpus, ingest, split_markdown
from rag.store import VectorStore

DOCS = {"kestrel_api_reference.md", "kestrel_platform_overview.md", "project_heron_timeline.md"}


def test_all_three_docs_are_chunked_with_doc_and_section():
    chunks = chunk_corpus()
    assert {c["doc"] for c in chunks} == DOCS
    for c in chunks:
        assert c["doc"] and c["section"] and c["text"].strip()


def test_chunks_respect_the_size_cap():
    assert max(len(c["text"]) for c in chunk_corpus()) <= config.CHUNK_SIZE


def test_section_is_the_header_path():
    md = "# Title\n\nIntro.\n\n## Auth\n\nText.\n\n### Tokens\n\nToken text.\n"
    chunks = split_markdown(md, "x.md", 1000, 0)
    assert [c["section"] for c in chunks] == ["Title", "Auth", "Auth > Tokens"]
    assert all(c["title"] == "Title" for c in chunks)


def test_ingest_writes_faiss_and_json_and_reloads(tmp_path, embedder):
    summary = ingest(embedder, out_dir=tmp_path)
    assert (tmp_path / "index.faiss").exists() and (tmp_path / "chunks.json").exists()
    assert not list(tmp_path.glob("*.pkl"))
    assert sorted(summary["documents"]) == sorted(DOCS)
    store = VectorStore.load(tmp_path)
    assert store.index.ntotal == summary["chunks"] == len(store.chunks)
    assert store.dimension == embedder.dim
    assert store.meta["embedding_model"] == embedder.model
