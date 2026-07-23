"""Capability-routing compiler: choose the cheapest sufficient path.

Generalizes Mixture-of-Experts from neurons to whole models and tools: each
step is routed to the target that clears a confidence threshold at least cost,
escalating to the core only when the on-hand option is likely to fail.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from majestic.types import RouteDecision, Step


class Router(ABC):
    @abstractmethod
    def route(self, step: Step) -> RouteDecision:
        """Pick a target for a step and decide whether to escalate."""
        raise NotImplementedError
