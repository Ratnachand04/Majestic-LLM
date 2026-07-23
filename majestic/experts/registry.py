"""Registry of available experts and tools."""
from __future__ import annotations

from majestic.experts.base import Expert


class ExpertRegistry:
    def __init__(self) -> None:
        self._experts: dict[str, Expert] = {}

    def register(self, expert: Expert) -> None:
        self._experts[expert.name] = expert

    def get(self, name: str) -> Expert:
        return self._experts[name]

    def has(self, name: str) -> bool:
        return name in self._experts

    def find_by_capability(self, capability: str) -> list[Expert]:
        """Return all registered experts that advertise ``capability``."""
        return [e for e in self._experts.values() if capability in e.capabilities]

    def all(self) -> list[Expert]:
        return list(self._experts.values())
