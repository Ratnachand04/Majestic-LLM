"""A deterministic mock reasoning core for offline runs and tests.

It performs no ML: ``plan`` emits a single step aimed at a default target and
``synthesize`` stitches the step results (and any grounding) into a string. This
keeps the full ``encode -> plan -> route -> execute -> verify -> respond``
pipeline runnable with zero heavy dependencies.
"""
from __future__ import annotations

from typing import Any

from majestic.core.reasoning_core import ReasoningCore
from majestic.types import Plan, Request, Step


class MockReasoningCore(ReasoningCore):
    """Echo/plan-stub core. Delegates the request to a single expert target."""

    def __init__(self, default_target: str = "echo") -> None:
        self.default_target = default_target

    def plan(self, request: Request) -> Plan:
        """Decompose into one step that hands the content to ``default_target``."""
        text = request.content if request.content is not None else ""
        step = Step(
            description=f"handle request via {self.default_target}",
            target=self.default_target,
            args={"text": text},
        )
        return Plan(steps=[step])

    def synthesize(
        self, request: Request, results: list[Any], grounding: list[str] | None = None
    ) -> str:
        """Combine step results (and optional grounding) into a final answer."""
        parts = [str(r) for r in results if r is not None]
        answer = " ".join(parts) if parts else str(request.content)
        if grounding:
            ctx = " | ".join(grounding)
            answer = f"{answer}\n\n[grounded on: {ctx}]"
        return answer
