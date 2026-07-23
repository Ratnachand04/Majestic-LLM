"""Retrieval-augmented generation pipeline."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from majestic.perception.encoders import Encoder
from majestic.retrieval.store import VectorStore
from majestic.types import Request


class RAGPipeline(ABC):
    @abstractmethod
    def retrieve(self, request: Request) -> list[str]:
        """Fetch grounding context (list of text snippets) for a request."""
        raise NotImplementedError


class NoOpRAG(RAGPipeline):
    """A pipeline that retrieves nothing. Used when grounding is disabled."""

    def retrieve(self, request: Request) -> list[str]:
        return []


class VectorRAG(RAGPipeline):
    """Encode the request, search a vector store, return the top text snippets.

    Documents are embedded with the same encoder used for queries so scores are
    comparable. ``min_score`` filters out weak matches (the hashing encoder
    yields small but positive scores for unrelated text).
    """

    def __init__(
        self,
        encoder: Encoder,
        store: VectorStore,
        top_k: int = 5,
        min_score: float = 0.1,
    ) -> None:
        self.encoder = encoder
        self.store = store
        self.top_k = top_k
        self.min_score = min_score

    def add_documents(
        self, docs: list[str], keys: list[str] | None = None
    ) -> None:
        for i, doc in enumerate(docs):
            key = keys[i] if keys else f"doc-{len(self._all_keys()) + i}"
            self.store.add(key, self.encoder.encode(doc), doc)

    def _all_keys(self) -> list[Any]:
        return getattr(self.store, "_keys", [])

    def retrieve(self, request: Request) -> list[str]:
        embedding = self.encoder.encode(request.content)
        if hasattr(self.store, "search_scored"):
            scored = self.store.search_scored(embedding, self.top_k)
            return [str(p) for score, p in scored if score >= self.min_score]
        return [str(p) for p in self.store.search(embedding, self.top_k)]
