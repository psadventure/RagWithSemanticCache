"""
FAISS vector store without pickle: an inner-product index over L2-normalised vectors
(= cosine similarity), saved as index.faiss + chunks.json.
"""
import json
from pathlib import Path

import faiss
import numpy as np

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"


class NotIngestedError(RuntimeError):
    """The vector store hasn't been built yet."""


def _normalised(vectors) -> np.ndarray:
    arr = np.asarray(vectors, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = np.ascontiguousarray(arr)
    faiss.normalize_L2(arr)
    return arr


class VectorStore:
    def __init__(self, index: faiss.Index, chunks: list[dict], meta: dict):
        self.index = index
        self.chunks = chunks
        self.meta = meta

    @property
    def dimension(self) -> int:
        return self.index.d

    @classmethod
    def build(cls, chunks: list[dict], vectors, meta: dict | None = None) -> "VectorStore":
        arr = _normalised(vectors)
        if len(arr) != len(chunks):
            raise ValueError(f"{len(chunks)} chunks but {len(arr)} vectors")
        index = faiss.IndexFlatIP(arr.shape[1])
        index.add(arr)
        return cls(index, chunks, meta or {})

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / INDEX_FILE))
        payload = {"meta": self.meta, "chunks": self.chunks}
        (path / CHUNKS_FILE).write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        index_file, chunks_file = path / INDEX_FILE, path / CHUNKS_FILE
        if not index_file.exists() or not chunks_file.exists():
            raise NotIngestedError(f"No vector store at '{path}'. Run `python -m rag.ingest` first.")
        payload = json.loads(chunks_file.read_text(encoding="utf-8"))
        return cls(faiss.read_index(str(index_file)), payload["chunks"], payload.get("meta", {}))

    def search(self, query_vector, k: int) -> list[tuple[dict, float]]:
        q = _normalised(query_vector)
        if q.shape[1] != self.dimension:
            raise ValueError(
                f"Query embedding has {q.shape[1]} dimensions but the index has {self.dimension}. "
                "The embedding model changed: re-run `python -m rag.ingest`."
            )
        scores, ids = self.index.search(q, min(k, self.index.ntotal))
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
