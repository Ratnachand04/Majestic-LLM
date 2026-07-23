"""End-to-end: the orchestrator routes to a tool and uses its result."""
from __future__ import annotations

from typing import Any

from majestic.core.reasoning_core import ReasoningCore
from majestic.experts.registry import ExpertRegistry
from majestic.experts.tools import CodeExecTool
from majestic.orchestrator import Orchestrator
from majestic.router.rule_router import RuleRouter
from majestic.types import Plan, Request, Step
from majestic.verification.basic import BasicVerifier


class CodePlanningCore(ReasoningCore):
    """A tiny core that plans a single code-execution step."""

    def plan(self, request: Request) -> Plan:
        return Plan(
            steps=[Step(description="compute", target="code_exec", args={"code": str(request.content)})]
        )

    def synthesize(
        self, request: Request, results: list[Any], grounding: list[str] | None = None
    ) -> str:
        if results and isinstance(results[0], dict):
            return results[0].get("stdout", "")
        return ""


def test_orchestrator_uses_code_tool_result():
    experts = ExpertRegistry()
    experts.register(CodeExecTool(timeout=15))
    orch = Orchestrator(
        core=CodePlanningCore(),
        router=RuleRouter(experts),
        experts=experts,
        verifier=BasicVerifier(),
    )
    resp = orch.handle(Request(content="print(2 + 3)"))
    assert resp.content == "5"
    assert "run:code_exec" in resp.trace
    assert resp.verified is True
