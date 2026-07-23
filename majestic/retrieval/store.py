"""Vector store and memory interfaces (+ offline implementations) for grounding."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class VectorStore(ABC):
    @abstractmethod
    def add(self, key: str, embedding: list[float], payload: Any) -> None: ...

    @abstractmethod
    def search(self, embedding: list[float], top_k: int = 5) -> list[Any]: ...


class Memory(ABC):
    """Conversational / user memory."""

    @abstractmethod
    def write(self, item: Any) -> None: ...

    @abstractmethod
    def recall(self, query: Any) -> list[Any]: ...


def _normalize(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0 else arr


class InMemoryVectorStore(VectorStore):
    """Cosine-similarity store backed by a NumPy matrix. No external services.

    Embeddings are L2-normalized on insert so an inner product equals cosine
    similarity. Fine for thousands of vectors; swap in FAISS for larger corpora.
    """

    def __init__(self) -> None:
        self._keys: list[str] = []
        self._payloads: list[Any] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._keys)

    def add(self, key: str, embedding: list[float], payload: Any) -> None:
        vec = _normalize(embedding)[None, :]
        self._keys.append(key)
        self._payloads.append(payload)
        self._matrix = vec if self._matrix is None else np.vstack([self._matrix, vec])

    def search(self, embedding: list[float], top_k: int = 5) -> list[Any]:
        return [payload for _, payload in self.search_scored(embedding, top_k)]

    def search_scored(
        self, embedding: list[float], top_k: int = 5
    ) -> list[tuple[float, Any]]:
        """Return ``(cosine_score, payload)`` pairs, highest score first."""
        if self._matrix is None or not self._keys:
            return []
        query = _normalize(embedding)
        sims = self._matrix @ query
        order = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), self._payloads[i]) for i in order]


class FaissVectorStore(VectorStore):
    """FAISS-backed store (optional). Lazily imports ``faiss``.

    API-compatible with :class:`InMemoryVectorStore`; selected when
    ``retrieval.vector_store == "faiss"`` and ``faiss-cpu`` is installed.
    """

    def __init__(self, dim: int) -> None:
        import faiss  # lazy: only needed if explicitly selected

        self._faiss = faiss
        self.index = faiss.IndexFlatIP(dim)
        self._keys: list[str] = []
        self._payloads: list[Any] = []

    def __len__(self) -> int:
        return len(self._keys)

    def add(self, key: str, embedding: list[float], payload: Any) -> None:
        vec = _normalize(embedding)[None, :]
        self.index.add(vec)
        self._keys.append(key)
        self._payloads.append(payload)

    def search(self, embedding: list[float], top_k: int = 5) -> list[Any]:
        return [payload for _, payload in self.search_scored(embedding, top_k)]

    def search_scored(
        self, embedding: list[float], top_k: int = 5
    ) -> list[tuple[float, Any]]:
        if not self._keys:
            return []
        query = _normalize(embedding)[None, :]
        scores, idxs = self.index.search(query, min(top_k, len(self._keys)))
        return [
            (float(scores[0][j]), self._payloads[i])
            for j, i in enumerate(idxs[0])
            if i != -1
        ]
