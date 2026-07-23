"""Simple in-process conversational / working memory."""
from __future__ import annotations

from typing import Any

from majestic.retrieval.store import Memory


class InMemoryMemory(Memory):
    """A list-backed memory with naive substring recall.

    Enough to persist turns within a session and retrieve related ones. Swap for
    a vector-backed memory for semantic recall at scale.
    """

    def __init__(self, max_items: int = 1000) -> None:
        self._items: list[Any] = []
        self.max_items = max_items

    def write(self, item: Any) -> None:
        self._items.append(item)
        if len(self._items) > self.max_items:
            self._items = self._items[-self.max_items :]

    def recall(self, query: Any) -> list[Any]:
        q = str(query).lower()
        return [item for item in self._items if q in str(item).lower()]

    def all(self) -> list[Any]:
        return list(self._items)
