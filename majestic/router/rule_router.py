"""Capability router: choose the cheapest sufficient path, else escalate.

Two-stage policy:

1. **Rule stage** — if the plan already names a registered target, trust it.
   Otherwise infer the needed capability from keywords in the step text and pick
   the *cheapest* registered expert that advertises it.
2. **Confidence + cost stage** — attach a confidence to the match. If the best
   available option does not clear ``confidence_threshold``, mark the decision
   ``escalate`` so the orchestrator sends it to the reasoning core instead of a
   likely-to-fail specialist.

This generalizes Mixture-of-Experts from neurons to whole models and tools.
"""
from __future__ import annotations

from majestic.experts.registry import ExpertRegistry
from majestic.logging_utils import get_logger
from majestic.router.router import Router
from majestic.types import RouteDecision, Step

logger = get_logger(__name__)

# Keyword -> capability rules for inferring intent from a step's text.
_RULES: dict[str, str] = {
    "search": "search",
    "look up": "search",
    "browse": "search",
    "web": "search",
    "latest": "search",
    "news": "search",
    "fetch": "search",
    "url": "search",
    "run code": "run_code",
    "execute": "run_code",
    "compute": "run_code",
    "calculate": "run_code",
    "evaluate": "run_code",
    "python": "run_code",
    "classify": "classify",
    "category": "classify",
    "sentiment": "classify",
    "label": "classify",
    "summarize": "summarize",
    "summary": "summarize",
    "extract": "extract",
}

# Confidence assigned when intent is inferred by rule (vs. an explicit target).
_RULE_CONFIDENCE = 0.8


class RuleRouter(Router):
    def __init__(
        self,
        registry: ExpertRegistry,
        confidence_threshold: float = 0.7,
        escalate_target: str = "echo",
    ) -> None:
        self.registry = registry
        self.confidence_threshold = confidence_threshold
        self.escalate_target = escalate_target

    def _infer_capability(self, step: Step) -> str | None:
        text = f"{step.description} {step.target}".lower()
        for keyword, capability in _RULES.items():
            if keyword in text:
                return capability
        return None

    def _cost(self, name: str, step: Step) -> float:
        if not self.registry.has(name):
            return 1.0
        try:
            return self.registry.get(name).estimate_cost(**step.args)
        except Exception:  # noqa: BLE001 - never let cost estimation break routing
            return 1.0

    def route(self, step: Step) -> RouteDecision:
        # 1. Explicit, registered target -> trust it (confidence 1.0).
        if self.registry.has(step.target):
            return RouteDecision(
                target=step.target,
                confidence=1.0,
                estimated_cost=self._cost(step.target, step),
                escalate=False,
            )

        # 2. Infer capability, pick the cheapest sufficient expert.
        capability = self._infer_capability(step)
        candidates = (
            self.registry.find_by_capability(capability) if capability else []
        )
        if candidates:
            best = min(candidates, key=lambda e: self._cost(e.name, step))
            if _RULE_CONFIDENCE >= self.confidence_threshold:
                return RouteDecision(
                    target=best.name,
                    confidence=_RULE_CONFIDENCE,
                    estimated_cost=self._cost(best.name, step),
                    escalate=False,
                )

        # 3. Nothing sufficient -> escalate to the core's fallback target.
        logger.debug("escalating step %r (no sufficient expert)", step.description)
        return RouteDecision(
            target=self.escalate_target,
            confidence=0.0,
            estimated_cost=self._cost(self.escalate_target, step),
            escalate=True,
        )
