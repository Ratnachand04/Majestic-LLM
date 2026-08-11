"""Parallel candidate builds, then score both and pick one (B-01, C-02 acts 5-6).

The planner starts at the LARGEST base that fits device RAM at 4-bit (A-01), but
RAM is not the only budget: the largest base that fits memory may still break the
latency budget. C-02 shows exactly this — a 4B scores 0.95 F1 at 4.1 s while a
1.7B scores 0.94 at 1.6 s, and the 1.7B wins because the spec demanded under
2 s. That trade cannot be settled by prediction alone, so where the plan is
uncertain Majestic **trains candidates in parallel and lets the Proving Ground
decide**.

Selection is lexicographic, and deliberately so:

1. Discard any candidate that fails its gate — a failing model is not a choice.
2. Discard any candidate that breaks the latency budget — the budget is a
   contract, not a preference.
3. Among survivors take the highest task score.
4. Break ties toward the SMALLER base: equal quality for less memory, latency
   and battery is strictly better on a device.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from majestic.logging_utils import get_logger
from modelrig.catalogue import DEFAULT_CATALOGUE, Catalogue
from modelrig.ir import BuildPlanIR, SpecIR
from modelrig.proving_ground import Scorecard

logger = get_logger(__name__)


@dataclass
class CandidateResult:
    """One trained-and-scored candidate."""

    plan: BuildPlanIR
    scorecard: Optional[Scorecard] = None
    latency_ms: float = 0.0
    params_b: float = 0.0
    error: str = ""
    compression: dict = field(default_factory=dict)
    quantisation: dict = field(default_factory=dict)

    @property
    def task_score(self) -> float:
        if self.scorecard is None:
            return 0.0
        axis = next((a for a in self.scorecard.axes if a.name == "task_metric"), None)
        return axis.score if axis else 0.0

    @property
    def passed(self) -> bool:
        return self.scorecard is not None and self.scorecard.passed and not self.error


@dataclass
class Selection:
    """The winner, the losers, and why."""

    winner: Optional[CandidateResult] = None
    candidates: list[CandidateResult] = field(default_factory=list)
    rationale: str = ""

    @property
    def chosen(self) -> bool:
        return self.winner is not None

    def comparison_table(self) -> list[dict[str, object]]:
        """Side-by-side rows for the customer scorecard (B-07)."""
        return [
            {
                "base": c.plan.base_ref,
                "params_b": c.params_b,
                "score": round(c.task_score, 4),
                "latency_ms": round(c.latency_ms, 1),
                "passed": c.passed,
                "winner": c is self.winner,
                "note": c.error,
            }
            for c in self.candidates
        ]


BuildFn = Callable[[BuildPlanIR], CandidateResult]


def build_candidates(
    plans: Sequence[BuildPlanIR],
    build_fn: BuildFn,
    max_workers: int = 2,
) -> list[CandidateResult]:
    """Train every candidate plan in parallel.

    ``build_fn`` does the real work for one plan and returns a scored
    :class:`CandidateResult`. Failures are captured per-candidate rather than
    aborting the batch: one candidate crashing must not lose the other.
    """
    if not plans:
        return []
    if len(plans) == 1:
        return [_guarded(build_fn, plans[0])]

    with ThreadPoolExecutor(max_workers=min(max_workers, len(plans))) as pool:
        return list(pool.map(lambda p: _guarded(build_fn, p), plans))


def _guarded(build_fn: BuildFn, plan: BuildPlanIR) -> CandidateResult:
    try:
        return build_fn(plan)
    except Exception as exc:  # noqa: BLE001 - one candidate must not sink the batch
        logger.warning("candidate %s failed to build: %s", plan.base_ref, exc)
        return CandidateResult(plan=plan, error=str(exc))


def select(
    results: Sequence[CandidateResult],
    spec: SpecIR,
    catalogue: Optional[Catalogue] = None,
) -> Selection:
    """Score both, pick one — latency budget is a hard filter, not a tiebreak."""
    catalogue = catalogue or DEFAULT_CATALOGUE
    candidates = list(results)
    for c in candidates:
        if not c.params_b:
            base = catalogue.base(c.plan.base_ref)
            c.params_b = base.params_b if base else 0.0

    if not candidates:
        return Selection(rationale="no candidates were built")

    passing = [c for c in candidates if c.passed]
    if not passing:
        return Selection(
            candidates=candidates,
            rationale="no candidate cleared its gate; nothing is offered to the customer",
        )

    within_budget = passing
    if spec.latency_budget_ms is not None:
        within_budget = [c for c in passing if c.latency_ms <= spec.latency_budget_ms]
        if not within_budget:
            return Selection(
                candidates=candidates,
                rationale=(
                    f"every passing candidate broke the {spec.latency_budget_ms}ms "
                    "latency budget; the budget is a contract, not a preference"
                ),
            )

    best_score = max(c.task_score for c in within_budget)
    # Ties go to the SMALLER base: same quality, less memory and battery.
    finalists = [c for c in within_budget if c.task_score >= best_score - 1e-9]
    winner = min(finalists, key=lambda c: c.params_b)

    rejected = [c for c in passing if c not in within_budget]
    parts = [
        f"{winner.plan.base_ref} scored {winner.task_score:.3f} "
        f"at {winner.latency_ms:.0f}ms"
    ]
    for c in rejected:
        parts.append(
            f"{c.plan.base_ref} scored {c.task_score:.3f} but took "
            f"{c.latency_ms:.0f}ms and broke the budget"
        )
    for c in within_budget:
        if c is not winner and c.task_score >= best_score - 1e-9:
            parts.append(f"{c.plan.base_ref} tied on quality but is larger")

    logger.info("candidates: selected %s (%s)", winner.plan.base_ref, "; ".join(parts))
    return Selection(winner=winner, candidates=candidates, rationale="; ".join(parts))
