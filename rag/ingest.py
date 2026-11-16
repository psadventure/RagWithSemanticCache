"""
Ingestion: Markdown docs in data/ → header-aware chunks → gateway embeddings → FAISS.

Run from the project root:
    python -m rag.ingest
"""
import sys
import time
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from rag import config
from rag.store import VectorStore

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def load_documents(data_dir: Path) -> list[tuple[str, str]]:
    """(file name, text) for every .md file in data_dir, sorted by name."""
    docs = [(p.name, p.read_text(encoding="utf-8")) for p in sorted(data_dir.glob("*.md"))]
    if not docs:
        raise FileNotFoundError(f"No .md files in '{data_dir}'.")
    return docs


def split_markdown(text: str, doc: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Split on Markdown headers first, then cap each section at chunk_size characters.

    Each chunk: {"text", "doc", "section", "title"}. `section` is the deepest header path
    ("Authentication > Access tokens"), or the document title for text before the first ##.
    """
    sections = MarkdownHeaderTextSplitter(HEADERS, strip_headers=False).split_text(text)
    capper = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for part in capper.split_documents(sections):
        md = part.metadata
        title = md.get("h1", doc)
        section = " > ".join(md[h] for h in ("h2", "h3") if md.get(h)) or title
        body = part.page_content.strip()
        if body:
            chunks.append({"text": body, "doc": doc, "section": section, "title": title})
    return chunks


def embedding_text(chunk: dict) -> str:
    """What gets embedded: title and section path give each chunk its context."""
    return f"{chunk['title']} | {chunk['section']}\n\n{chunk['text']}"


def chunk_corpus(data_dir: Path | None = None) -> list[dict]:
    data_dir = data_dir or config.DATA_PATH
    chunks = []
    for name, text in load_documents(data_dir):
        chunks.extend(split_markdown(text, name, config.CHUNK_SIZE, config.CHUNK_OVERLAP))
    return chunks


def ingest(embedder, data_dir: Path | None = None, out_dir: Path | None = None, progress=None) -> dict:
    """Chunk, embed and save. Returns a summary."""
    out_dir = out_dir or config.VECTOR_STORE_PATH
    chunks = chunk_corpus(data_dir)
    vectors = embedder.embed([embedding_text(c) for c in chunks], progress=progress)
    meta = {"embedding_model": embedder.model, "dimension": len(vectors[0]), "chunks": len(chunks)}
    VectorStore.build(chunks, vectors, meta).save(out_dir)
    docs = sorted({c["doc"] for c in chunks})
    return {**meta, "documents": docs, "path": str(out_dir)}


def main() -> None:
    from rag.gateway import GatewayEmbedder

    print(f"Gateway {config.LITELLM_BASE_URL}, embedding alias {config.EMBEDDING_MODEL!r}")

    def progress(i, n):
        print(f"\r  embedded {i}/{n} chunks (one request each)", end="", flush=True)

    start = time.perf_counter()
    try:
        summary = ingest(GatewayEmbedder(), progress=progress)
    except Exception as exc:  # CLI: show the problem, not a traceback
        sys.exit(f"\n✗ Ingestion failed: {type(exc).__name__}: {exc}")
    print(f"\n✅ {summary['chunks']} chunks from {len(summary['documents'])} docs, "
          f"{summary['dimension']}-dim vectors → {summary['path']}/ ({time.perf_counter() - start:.0f}s)")
    for doc in summary["documents"]:
        print(f"   - {doc}")


if __name__ == "__main__":
    main()
