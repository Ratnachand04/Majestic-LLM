"""Hardening: the orchestrator degrades gracefully around bad/stub/missing steps."""
from __future__ import annotations

from typing import Any

from majestic.core.reasoning_core import ReasoningCore
from majestic.experts.pool import DiffusionExpert
from majestic.experts.registry import ExpertRegistry
from majestic.orchestrator import Orchestrator
from majestic.router.simple import SimpleRouter
from majestic.types import Plan, Request, Step


class _PlanTo(ReasoningCore):
    def __init__(self, target: str) -> None:
        self.target = target

    def plan(self, request: Request) -> Plan:
        return Plan(steps=[Step(description="x", target=self.target, args={})])

    def synthesize(self, request, results, grounding=None) -> str:
        return f"done ({len(results)} results)"


def test_missing_target_is_skipped_not_crashed():
    experts = ExpertRegistry()  # nothing registered
    orch = Orchestrator(core=_PlanTo("ghost"), router=SimpleRouter(), experts=experts)
    resp = orch.handle(Request(content="hi"))
    assert "missing:ghost" in resp.trace
    assert resp.content == "done (0 results)"


def test_stub_expert_is_skipped_not_crashed():
    experts = ExpertRegistry()
    experts.register(DiffusionExpert())  # run() raises NotImplementedError
    orch = Orchestrator(core=_PlanTo("diffusion"), router=SimpleRouter(), experts=experts)
    resp = orch.handle(Request(content="draw a cat"))
    assert "stub:diffusion" in resp.trace
    assert resp.content == "done (0 results)"


def test_encoder_failure_is_non_fatal():
    class _BadEncoder:
        modality = None

        def encode(self, data: Any):
            raise NotImplementedError

    experts = ExpertRegistry()
    orch = Orchestrator(
        core=_PlanTo("ghost"), router=SimpleRouter(), experts=experts,
        encoder=_BadEncoder(),
    )
    # Should not raise even though the encoder is not implemented.
    resp = orch.handle(Request(content="hi"))
    assert resp.content == "done (0 results)"
