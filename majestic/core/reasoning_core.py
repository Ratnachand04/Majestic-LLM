"""The reasoning core: the orchestrating multimodal model interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from majestic.types import Plan, Request


class ReasoningCore(ABC):
    """Abstract multimodal reasoning core (dense or Mixture-of-Experts).

    Concrete implementations wrap an existing open foundation model. The core
    does NOT perform tasks itself; it plans, decomposes, and delegates.
    """

    @abstractmethod
    def plan(self, request: Request) -> Plan:
        """Decompose a request into an ordered plan of steps."""
        raise NotImplementedError

    @abstractmethod
    def synthesize(
        self, request: Request, results: list[Any], grounding: list[str] | None = None
    ) -> str:
        """Combine step results (and optional retrieval grounding) into an answer."""
        raise NotImplementedError
