"""A small, file-backed vector store with cosine-similarity search.

Stores embedding vectors alongside their source text and arbitrary metadata,
persists to a single ``.npz`` file, and answers top-k nearest-neighbour queries.

No external vector DB needed — fine for thousands to low-hundreds-of-thousands
of chunks. For larger corpora swap this for a dedicated ANN index; the RAG layer
only relies on ``add`` / ``search`` / ``save`` / ``load``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class SearchResult:
    """One retrieved item."""

    text: str
    score: float
    metadata: Dict[str, Any]
    index: int

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.text


def _normalize(mat: np.ndarray) -> np.ndarray:
    """L2-normalize rows so dot product == cosine similarity."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class VectorStore:
    """In-memory vectors + texts + metadata, with disk persistence."""

    def __init__(self) -> None:
        self._vectors: Optional[np.ndarray] = None  # (N, D), L2-normalized
        self._texts: List[str] = []
        self._metadata: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._texts)

    @property
    def dim(self) -> Optional[int]:
        return None if self._vectors is None else int(self._vectors.shape[1])

    # ── building ─────────────────────────────────────────────────────────────
    def add(
        self,
        texts: Sequence[str],
        vectors: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """Append texts with their precomputed vectors (and optional metadata)."""
        if len(texts) != len(vectors):
            raise ValueError("texts and vectors must be the same length")
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas must match texts in length")
        if not texts:
            return

        new = _normalize(np.asarray(vectors, dtype=np.float32))
        if self._vectors is None:
            self._vectors = new
        else:
            if new.shape[1] != self._vectors.shape[1]:
                raise ValueError(
                    f"vector dim {new.shape[1]} != store dim {self._vectors.shape[1]}"
                )
            self._vectors = np.vstack([self._vectors, new])

        self._texts.extend(texts)
        self._metadata.extend(metadatas if metadatas is not None else [{} for _ in texts])

    # ── querying ─────────────────────────────────────────────────────────────
    def search(
        self, query_vector: Sequence[float], k: int = 4, min_score: float = 0.0
    ) -> List[SearchResult]:
        """Return the top-``k`` items by cosine similarity to ``query_vector``."""
        if self._vectors is None or len(self._texts) == 0:
            return []
        q = _normalize(np.asarray([query_vector], dtype=np.float32))[0]
        scores = self._vectors @ q  # cosine, since both sides are normalized
        k = max(1, min(k, len(self._texts)))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        results = []
        for i in top:
            s = float(scores[i])
            if s < min_score:
                continue
            results.append(
                SearchResult(text=self._texts[i], score=s, metadata=self._metadata[i], index=int(i))
            )
        return results

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        """Persist the store to ``path`` (a single ``.npz`` archive)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        vectors = self._vectors if self._vectors is not None else np.zeros((0, 0), np.float32)
        np.savez_compressed(
            path,
            vectors=vectors,
            texts=np.array(self._texts, dtype=object),
            metadata=np.array([json.dumps(m) for m in self._metadata], dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        """Load a store previously written with :meth:`save`."""
        data = np.load(Path(path), allow_pickle=True)
        store = cls()
        vectors = data["vectors"]
        store._vectors = vectors.astype(np.float32) if vectors.size else None
        store._texts = list(data["texts"])
        store._metadata = [json.loads(m) for m in data["metadata"]]
        return store
