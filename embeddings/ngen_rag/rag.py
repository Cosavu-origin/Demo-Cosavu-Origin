"""RAG over TNSA ngen-embeddings.

Glues the embedding client and the vector store into the usual retrieval flow:

    index:   text -> chunks -> embeddings -> VectorStore (saved to disk)
    query:   question -> embedding -> top-k chunks -> context string

Generation (calling an LLM with the retrieved context) is intentionally left to
the caller — embeddings only do retrieval. :meth:`RAG.build_prompt` assembles a
ready-to-send prompt; pass it to whatever model you like.

    from ngen_rag import RAG

    rag = RAG()
    rag.add_texts(["... long doc ..."], metadatas=[{"source": "handbook"}])
    rag.save("kb.npz")

    # later / elsewhere
    rag = RAG.load("kb.npz")
    hits = rag.retrieve("how do refunds work?", k=4)
    prompt = rag.build_prompt("how do refunds work?")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .client import EmbeddingClient
from .store import SearchResult, VectorStore


def chunk_text(
    text: str, chunk_size: int = 800, overlap: int = 100
) -> List[str]:
    """Split text into overlapping word-based chunks.

    Sizes are measured in characters (approx). ``overlap`` keeps context across
    boundaries so a sentence split across two chunks is still retrievable.
    """
    text = text.strip()
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size - 1))

    words = text.split()
    chunks: List[str] = []
    current: List[str] = []
    length = 0
    for word in words:
        # +1 for the joining space
        if length + len(word) + 1 > chunk_size and current:
            chunks.append(" ".join(current))
            # carry over the tail for overlap
            tail, tail_len = [], 0
            for w in reversed(current):
                if tail_len + len(w) + 1 > overlap:
                    break
                tail.insert(0, w)
                tail_len += len(w) + 1
            current, length = tail, tail_len
        current.append(word)
        length += len(word) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


class RAG:
    """A document index you can build, save, reload, and query."""

    def __init__(
        self,
        client: Optional[EmbeddingClient] = None,
        store: Optional[VectorStore] = None,
        chunk_size: int = 800,
        overlap: int = 100,
        embed_batch_size: int = 64,
    ) -> None:
        self.client = client or EmbeddingClient()
        self.store = store or VectorStore()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embed_batch_size = embed_batch_size

    def __len__(self) -> int:
        return len(self.store)

    # ── indexing ─────────────────────────────────────────────────────────────
    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        chunk: bool = True,
    ) -> int:
        """Chunk (optionally), embed, and add documents to the index.

        Each chunk inherits its parent document's metadata, plus ``chunk`` and
        ``doc`` indices. Returns the number of chunks added.
        """
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas must match texts in length")

        all_chunks: List[str] = []
        all_meta: List[Dict[str, Any]] = []
        for doc_i, text in enumerate(texts):
            base = dict(metadatas[doc_i]) if metadatas else {}
            pieces = chunk_text(text, self.chunk_size, self.overlap) if chunk else [text.strip()]
            for chunk_i, piece in enumerate(pieces):
                if not piece:
                    continue
                meta = {**base, "doc": doc_i, "chunk": chunk_i}
                all_chunks.append(piece)
                all_meta.append(meta)

        if not all_chunks:
            return 0

        vectors: List[List[float]] = []
        for start in range(0, len(all_chunks), self.embed_batch_size):
            batch = all_chunks[start : start + self.embed_batch_size]
            vectors.extend(self.client.embed_batch(batch))

        self.store.add(all_chunks, vectors, all_meta)
        return len(all_chunks)

    # ── retrieval ────────────────────────────────────────────────────────────
    def retrieve(
        self, query: str, k: int = 4, min_score: float = 0.0
    ) -> List[SearchResult]:
        """Embed ``query`` and return the top-``k`` matching chunks."""
        if not query or not query.strip():
            raise ValueError("query is empty")
        qvec = self.client.embed(query)
        return self.store.search(qvec, k=k, min_score=min_score)

    def build_context(self, query: str, k: int = 4, min_score: float = 0.0) -> str:
        """Return retrieved chunks joined into a single context block."""
        hits = self.retrieve(query, k=k, min_score=min_score)
        parts = []
        for n, h in enumerate(hits, 1):
            src = h.metadata.get("source")
            tag = f"[{n}]" + (f" ({src})" if src else "")
            parts.append(f"{tag} {h.text}")
        return "\n\n".join(parts)

    def build_prompt(
        self,
        query: str,
        k: int = 4,
        min_score: float = 0.0,
        instructions: Optional[str] = None,
    ) -> str:
        """Assemble a ready-to-send RAG prompt: context + question.

        Pass the result to any LLM. Embeddings do retrieval; the model generates.
        """
        instructions = instructions or (
            "Answer the question using only the context below. "
            "If the answer isn't in the context, say you don't know."
        )
        context = self.build_context(query, k=k, min_score=min_score) or "(no relevant context found)"
        return f"{instructions}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        self.store.save(path)

    @classmethod
    def load(
        cls, path: str | Path, client: Optional[EmbeddingClient] = None, **kwargs
    ) -> "RAG":
        """Load an index from disk. A fresh embedding client is created for queries
        unless you pass one."""
        return cls(client=client, store=VectorStore.load(path), **kwargs)
