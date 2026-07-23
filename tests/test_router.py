"""Tests for the rule + confidence + cost router."""
from __future__ import annotations

from typing import Any

from majestic.experts.base import Expert
from majestic.experts.echo import EchoExpert
from majestic.experts.registry import ExpertRegistry
from majestic.experts.specialist import SpecialistExpert
from majestic.experts.tools import CodeExecTool, WebTool
from majestic.router.rule_router import RuleRouter
from majestic.types import Step


class CheapSearch(Expert):
    name = "cheap_search"
    capabilities = ("search",)

    def run(self, **kwargs: Any) -> Any:
        return "cheap"

    def estimate_cost(self, **kwargs: Any) -> float:
        return 0.5


def _registry() -> ExpertRegistry:
    reg = ExpertRegistry()
    for e in (EchoExpert(), SpecialistExpert(), CodeExecTool(), WebTool()):
        reg.register(e)
    return reg


def test_explicit_registered_target_is_trusted():
    router = RuleRouter(_registry())
    d = router.route(Step(description="do it", target="code_exec"))
    assert d.target == "code_exec"
    assert d.confidence == 1.0
    assert d.escalate is False


def test_capability_inference_when_target_unknown():
    router = RuleRouter(_registry())
    d = router.route(Step(description="search the web for news", target="unknown"))
    assert d.target == "web"
    assert d.escalate is False


def test_cheapest_sufficient_is_chosen():
    reg = _registry()
    reg.register(CheapSearch())
    router = RuleRouter(reg)
    d = router.route(Step(description="look up the latest", target="unknown"))
    # web costs 3.0, cheap_search costs 0.5 -> cheapest wins.
    assert d.target == "cheap_search"
    assert d.estimated_cost == 0.5


def test_escalation_when_no_expert_matches():
    router = RuleRouter(_registry(), escalate_target="echo")
    d = router.route(Step(description="do something unmappable xyzzy", target="ghost"))
    assert d.escalate is True
    assert d.target == "echo"
    assert d.confidence == 0.0


def test_threshold_blocks_low_confidence_inference():
    # A very high threshold means inferred (0.8) matches are not sufficient.
    router = RuleRouter(_registry(), confidence_threshold=0.95)
    d = router.route(Step(description="classify this sentiment", target="unknown"))
    assert d.escalate is True
