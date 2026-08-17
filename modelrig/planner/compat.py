"""The stateful adapter the build pipeline uses.

This is a **facade**, not a second planner. Every decision is delegated to
:mod:`modelrig.planner.core`; what lives here is the mutable state a long-running
pipeline needs — accumulated precedents and build outcomes — plus the
``PlanningResult`` shape the pipeline already consumes.

The one piece of real logic is the rare path of §7: when no precedent matches and
an external proposer (LLM role 5, tie-breaker only) offers a plan, the validator
disposes of it and the mutation loop repairs it. An LLM never freestyles a
training plan — a hallucinated tokenizer mismatch costs sixty dollars of GPU to
discover.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from majestic.logging_utils import get_logger
from modelrig.gates import GateResult, gate2_plan_feasibility
from modelrig.ir import BuildPlanIR, SpecIR
from modelrig.planner import core
from modelrig.planner.catalog import Catalog, CatalogError, ModelSpec, default_catalog
from modelrig.planner.metalearn import (
    Observation,
    OutcomePredictor,
    PrecedentIndex,
    features_of,
)
from modelrig.planner.objective import Tier
from modelrig.planner.predicates import ordered_predicates, seed_floor
from modelrig.planner.refusal import Refusal

logger = get_logger(__name__)

_MAX_MUTATIONS = 6


@dataclass
class PrecedentRecord:
    """One past build: the shape of its spec and the plan that passed its gate."""

    spec_shape: tuple
    plan: BuildPlanIR
    passed: bool


@dataclass
class PlanningResult:
    """What the planner produced, and how it got there."""

    plan: Optional[BuildPlanIR]
    gate: GateResult
    path: str = "deterministic"           # deterministic | precedent | proposed | refused
    mutations: list[str] = field(default_factory=list)
    refusal: Optional[Refusal] = None
    outcome: Optional[core.PlanOutcome] = None

    @property
    def admitted(self) -> bool:
        return self.plan is not None and self.gate.passed


class Planner:
    """Compiles a Spec IR into an admitted Build Plan IR, or refuses."""

    def __init__(
        self,
        catalogue=None,
        profiler=None,
        proposer: Optional[Callable[[SpecIR, object], BuildPlanIR]] = None,
        outcome_predictor: Optional[OutcomePredictor] = None,
        catalog: Optional[Catalog] = None,
        tier: Tier = Tier.COMMERCIAL,
        exploration_rate: float = 0.0,
    ) -> None:
        # ``catalogue``/``profiler`` are the legacy positional arguments the
        # pipeline passes; the typed catalogue supersedes both.
        self.legacy_catalogue = catalogue
        self.profiler = profiler
        self.proposer = proposer          # LLM role 5 — tie-breaker only
        self.tier = tier
        self.catalog = catalog or default_catalog()
        self.precedents = PrecedentIndex(exploration_rate=exploration_rate)
        self.outcomes = outcome_predictor or OutcomePredictor()
        self.history: list[PrecedentRecord] = []

    # -- history ---------------------------------------------------------- #
    def record_outcome(
        self, spec: SpecIR, plan: BuildPlanIR, passed: bool, margin: float | None = None
    ) -> None:
        """Feed a finished build back into precedents and the meta-learner."""
        shape = core.spec_shape(spec)
        self.history.append(PrecedentRecord(shape, plan, passed))
        self.precedents.add(shape, plan, passed, margin or (1.0 if passed else 0.0))

        try:
            base = self.catalog.model(plan.base_ref)
            bits = self.catalog.bits_effective(base, plan.quantiser)
            params_b = base.params_b
        except CatalogError:
            bits, params_b = 5.2, 1.0
        self.outcomes.record(Observation(
            spec_hash=spec.hash, plan_hash=plan.hash,
            features=features_of(
                params_b=params_b,
                seed_ratio=spec.seed_data_count / max(seed_floor(spec.task_primitive), 1),
                bits_effective=bits, distil_mode=plan.distil_mode,
                context=2048, quality_gate=spec.quality_gate,
            ),
            outcome=margin if margin is not None else (1.0 if passed else 0.0),
            primitive=spec.task_primitive.value, passed=passed,
        ))

    # -- the §12 chain ---------------------------------------------------- #
    def _eligible_bases(self, spec: SpecIR) -> list[ModelSpec]:
        """Bases that fit, largest first. The chain of §12."""
        best = core.select_base(spec, self.catalog)
        chain = self.catalog.bases(spec.task_primitive)
        if best is None:
            return list(reversed(chain))[:1]
        # Antitone: everything at or below b* fits, so the eligible set is the
        # prefix of the reversed chain starting at b*.
        return [m for m in reversed(chain) if m.params <= best.params]

    def _device_cap(self, spec: SpecIR) -> float:
        """Largest base (in billions) this device admits at 4-bit."""
        best = core.select_base(spec, self.catalog)
        return best.params_b if best else 0.0

    def _plan_for_base(self, spec: SpecIR, base) -> BuildPlanIR:
        """Build a complete, validated plan around one chosen base.

        Accepts either a typed :class:`ModelSpec` or a legacy
        ``modelrig.catalogue.BaseModel``, because the repair loop in the pipeline
        passes the latter.
        """
        ref = getattr(base, "ref", str(base))
        model = self.catalog.model(ref)
        enum = core.enumerate_feasible(spec, self.catalog, max_bases=len(self.catalog.models))
        for candidate in enum.feasible:
            if candidate.base.ref == model.ref:
                return candidate.to_ir(spec, self._cost_of(candidate))
        # Nothing feasible on that base: emit the shape anyway so the caller's
        # validator produces a precise numeric reason rather than a blank.
        fallback = core.PlanCandidate(
            base=model, teacher=None, distil_mode="none", peft_method="lora",
            rank=16, quantiser="q4_k_m", bit_width="int4",
            target="gguf" if spec.offline_required else "onnx",
            data_recipe=core._data_recipe(spec, model),
        )
        return fallback.to_ir(spec, self._cost_of(fallback))

    def _cost_of(self, candidate: core.PlanCandidate) -> int:
        from modelrig.planner import costmodel

        recipe = candidate.data_recipe
        return costmodel.estimate(
            candidate.base, method=candidate.peft_method,
            n_synthetic=int(recipe.get("n_synthetic", 0)),
            n_train_tokens=int(recipe.get("n_train_tokens", 0)),
            n_eval_examples=int(recipe.get("n_eval_examples", 0)),
            uses_teacher=candidate.teacher is not None,
        ).total

    def _default_plan(self, spec: SpecIR) -> BuildPlanIR:
        """The largest base that fits, fully planned (§12 + A-01)."""
        base = core.select_base(spec, self.catalog)
        if base is None:
            base = self.catalog.bases(spec.task_primitive)[0]
        return self._plan_for_base(spec, base)

    # -- parallel candidates: opt-in only (§15) ---------------------------- #
    def candidate_plans(self, spec: SpecIR, limit: int = 2) -> list[BuildPlanIR]:
        """Alternate plans to build in parallel — **only if the spec asks**.

        §15: running k candidates is strictly worse in expected cost at every
        pass probability. For two identical candidates,

            E[cost]_2 / E[cost]_1 = 2 / (2 - p) > 1     for all p in (0, 1]

        At p = 0.5 that is 33% more expected spend than building one and
        retrying on failure. What parallelism buys is **wall-clock time and
        variance reduction** — a time-for-money trade the customer must choose,
        never one the planner makes on their behalf. So this returns a single
        plan unless ``io_schema.allow_parallel_candidates`` is set.
        """
        outcome = core.plan(spec, self.catalog, precedents=self.precedents,
                            predictor=self.outcomes, tier=self.tier)
        if outcome.plan is None:
            return []
        if not core._wants_parallel(spec):
            return [outcome.plan]
        return [outcome.plan, *outcome.candidates][:limit]

    @staticmethod
    def parallel_cost_ratio(p: float, k: int = 2) -> float:
        """``E[cost]_k / E[cost]_1`` for k identical candidates (§15).

        Exposed so the customer-facing surface can quote the real number instead
        of the folklore that parallel candidates are free.
        """
        if not 0.0 < p <= 1.0:
            raise ValueError("pass probability must lie in (0, 1]")
        return (k * p) / (1.0 - (1.0 - p) ** k)

    # -- the rare path: an LLM proposes, the validator disposes ------------ #
    def _mutate(self, plan: BuildPlanIR, gate: GateResult) -> tuple[BuildPlanIR, str] | None:
        reasons = " ".join(gate.reasons).lower()
        current = self.catalog.models.get(plan.base_ref)
        smaller = [
            m for m in reversed(self.catalog.bases())
            if current is None or m.params < current.params
        ]

        if "ram" in reasons or "free ram" in reasons or "over by" in reasons:
            if smaller:
                plan.base_ref = smaller[0].ref
                return plan, f"downsized base to {smaller[0].ref}"
            if plan.bit_width != "int4":
                plan.bit_width, plan.quantiser = "int4", "q4_k_m"
                return plan, "dropped to int4"
            return None
        if "tokenizer" in reasons or "logit distillation" in reasons:
            plan.distil_mode = "sequence_kd"
            return plan, "switched to sequence-level distillation (crosses tokenizers)"
        if "latency" in reasons:
            if smaller:
                plan.base_ref = smaller[0].ref
                return plan, f"downsized base for latency to {smaller[0].ref}"
            return None
        if "offline closure" in reasons and plan.target == "vllm":
            plan.target = "gguf"
            return plan, "retargeted to gguf for offline closure"
        if "budget" in reasons or "ceiling" in reasons:
            if plan.peft_method != "qlora":
                plan.peft_method = "qlora"
                return plan, "switched to QLoRA to fit the budget"
            return None
        return None      # licence and data-rights failures are not mutable

    # -- the public entry point ------------------------------------------- #
    def plan(self, spec: SpecIR) -> PlanningResult:
        """Produce an admitted Build Plan IR, or a gate result explaining why not."""
        if self.proposer is not None and self.precedents.nearest(core.spec_shape(spec)) is None:
            return self._proposed_path(spec)

        outcome = core.plan(spec, self.catalog, precedents=self.precedents,
                            predictor=self.outcomes, tier=self.tier)
        if outcome.plan is None:
            gate = GateResult(
                "gate2_plan_feasibility", False,
                reasons=list(outcome.refusal.reasons) if outcome.refusal else ["no feasible plan"],
                evidence=outcome.refusal.as_dict() if outcome.refusal else {},
            )
            return PlanningResult(None, gate, "refused", [], outcome.refusal, outcome)

        gate = gate2_plan_feasibility(spec, outcome.plan, self.profiler, self.legacy_catalogue)
        return PlanningResult(outcome.plan, gate, outcome.path, [], None, outcome)

    def _proposed_path(self, spec: SpecIR) -> PlanningResult:
        """LLM role 5. The proposal is validated and repaired, never trusted."""
        plan = self.proposer(spec, self.legacy_catalogue)
        logger.info("planner: LLM proposed a plan for %s", spec.hash[:8])
        mutations: list[str] = []
        gate = gate2_plan_feasibility(spec, plan, self.profiler, self.legacy_catalogue)

        for _ in range(_MAX_MUTATIONS):
            if gate.passed:
                break
            mutated = self._mutate(plan, gate)
            if mutated is None:
                break
            plan, note = mutated
            mutations.append(note)
            logger.info("planner: mutating proposal — %s", note)
            gate = gate2_plan_feasibility(spec, plan, self.profiler, self.legacy_catalogue)

        if not gate.passed:
            return PlanningResult(None, gate, "proposed", mutations)
        return PlanningResult(plan, gate, "proposed", mutations)

    # -- diagnostics ------------------------------------------------------- #
    def explain(self, spec: SpecIR) -> dict:
        """Everything behind a decision, for auditing."""
        outcome = core.plan(spec, self.catalog, precedents=self.precedents,
                            predictor=self.outcomes, tier=self.tier)
        return {
            "path": outcome.path,
            "admitted": outcome.admitted,
            "considered": outcome.considered,
            "predicate_evaluations": outcome.predicate_evaluations,
            "predicate_order": [p.name for p in ordered_predicates()],
            "predicted_quality": round(outcome.predicted_quality, 4),
            "threshold": round(outcome.threshold, 4),
            "refusal": outcome.refusal.as_dict() if outcome.refusal else None,
        }
