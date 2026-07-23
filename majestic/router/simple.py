"""A pass-through router used in Phase 0.

It trusts the plan's chosen target and never escalates. Phase 2 replaces this
with a rule + confidence + cost router that can escalate to the core.
"""
from __future__ import annotations

from majestic.router.router import Router
from majestic.types import RouteDecision, Step


class SimpleRouter(Router):
    """Routes each step straight to the target named in the plan."""

    def route(self, step: Step) -> RouteDecision:
        return RouteDecision(
            target=step.target,
            confidence=1.0,
            estimated_cost=0.0,
            escalate=False,
        )
