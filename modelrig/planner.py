"""PLANNER — deterministic first, LLM only at the edges (B-05).

    Spec IR -> warm start from precedent -> [reuse | propose] -> VALIDATOR -> Build Plan IR

The common path is deterministic: embed the spec, find a past build with the same
shape that passed its gate, and adapt its plan. No LLM is involved. Only when no
precedent matches may a proposer suggest a plan — and the validator always
disposes. **An LLM never freestyles a training plan**: a hallucinated tokenizer
mismatch costs sixty dollars of GPU to discover.

Search-based design is deliberately rejected in the build loop. Neural
architecture search costs thousands of GPU-hours per result, which is
irreconcilable with a $20-$120 build. Rules plus meta-learning, not search.

GAP-02 (open): the warm start becomes a compounding asset only once
:class:`~modelrig.feasibility.OutcomePredictor` can predict gate-pass from a spec
embedding. Until then precedent lookup is exact-shape matching.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from majestic.logging_utils import get_logger
from modelrig.catalogue import DEFAULT_CATALOGUE, BaseModel, Catalogue, ladder_tier
from modelrig.feasibility import HeuristicFeasibilityEngine, OutcomePredictor
from modelrig.gates import GateResult, gate2_plan_feasibility
from modelrig.ir import BuildPlanIR, SpecIR
from modelrig.primitives import TaskPrimitive, spec_for

logger = get_logger(__name__)

# Cost model for the budget predicate, in USD (QLoRA on rented A100s, A-03).
_COST_PER_PARAM_B = {"lora": 4.0, "dora": 5.0, "qlora": 3.0, "full_ft": 25.0}
_MAX_MUTATIONS = 6
#: A device that cannot hold this many billion resident parameters is
#: "memory-constrained" and MoE bases are excluded from it entirely (A-07).
_MOE_RESIDENCY_HEADROOM_B = 30.0
#: Declared share of novel domain vocabulary above which LoRA is not enough (A-03).
_NOVEL_VOCAB_FULL_FT_THRESHOLD = 0.30
#: Primitives where coverage matters more than mode-seeking, so forward-KL (A-02).
_COVERAGE_PRIMITIVES = frozenset(
    {TaskPrimitive.GENERATE, TaskPrimitive.SUMMARISE, TaskPrimitive.REWRITE}
)


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
    path: str = "deterministic"          # deterministic | precedent | proposed
    mutations: list[str] = field(default_factory=list)

    @property
    def admitted(self) -> bool:
        return self.plan is not None and self.gate.passed


def _spec_shape(spec: SpecIR) -> tuple:
    """The coarse shape used for precedent matching (not the exact hash)."""
    return (
        spec.task_primitive.value,
        spec.device_target,
        spec.offline_required,
        round(spec.quality_gate, 2),
    )


class Planner:
    """Compiles a Spec IR into an admitted Build Plan IR."""

    def __init__(
        self,
        catalogue: Catalogue | None = None,
        profiler=None,
        proposer: Optional[Callable[[SpecIR, Catalogue], BuildPlanIR]] = None,
        outcome_predictor: Optional[OutcomePredictor] = None,
    ) -> None:
        self.catalogue = catalogue or DEFAULT_CATALOGUE
        self.profiler = profiler
        self.proposer = proposer          # LLM role 5 — tie-breaker only
        self.history: list[PrecedentRecord] = []
        self.outcomes = outcome_predictor or OutcomePredictor()

    # -- warm start ------------------------------------------------------ #
    def record_outcome(self, spec: SpecIR, plan: BuildPlanIR, passed: bool) -> None:
        """Feed a finished build back into the planner's memory."""
        self.history.append(PrecedentRecord(_spec_shape(spec), plan, passed))
        self.outcomes.record(spec.hash, plan.hash, spec.task_primitive.value, passed)

    def _precedent(self, spec: SpecIR) -> BuildPlanIR | None:
        shape = _spec_shape(spec)
        for record in reversed(self.history):
            if record.spec_shape == shape and record.passed:
                return record.plan
        return None

    # -- deterministic construction --------------------------------------- #
    def _device_cap(self, spec: SpecIR) -> float:
        """Largest base (in billions) this device can hold at 4-bit."""
        if self.profiler is None:
            return float("inf")
        try:
            device = self.profiler.profile(spec.device_target)
        except KeyError:
            return float("inf")
        engine = HeuristicFeasibilityEngine(profiler=self.profiler)
        return engine.max_base_for(device)

    def _eligible_bases(self, spec: SpecIR) -> list[BaseModel]:
        """Bases that fit the device, ordered LARGEST FIRST (A-01)."""
        prim = spec_for(spec.task_primitive)
        max_b = self._device_cap(spec)

        # MoE is excluded from memory-constrained targets: sparse activation is
        # not sparse residency, so the whole expert set must be held (A-07).
        memory_constrained = max_b < _MOE_RESIDENCY_HEADROOM_B
        candidates = self.catalogue.bases_for(
            spec.task_primitive,
            largest_first=True,
            allow_moe=not memory_constrained,
            max_params_b=None if max_b == float("inf") else max_b,
            min_params_b=prim.min_params_b,
        )
        if candidates:
            return candidates
        # Nothing fits: return the smallest supporting base anyway so the
        # validator reports a precise, numeric reason instead of a blank refusal.
        fallback = self.catalogue.bases_for(spec.task_primitive, largest_first=False)
        return fallback[:1] or list(self.catalogue.bases)[:1]

    def _needs_full_finetune(self, spec: SpecIR) -> bool:
        """Escalate past LoRA only on declared novel domain vocabulary (A-03).

        Biderman et al. (2405.09673): LoRA learns less and forgets less. It
        underperforms full fine-tuning when a genuinely NEW domain must be
        learned, but preserves general ability far better — so the escalation is
        deliberate and threshold-driven, and a regression suite runs either way.
        """
        novel = float(spec.io_schema.get("novel_vocabulary_ratio", 0.0)) \
            if isinstance(spec.io_schema, dict) else 0.0
        return novel >= _NOVEL_VOCAB_FULL_FT_THRESHOLD

    def _kl_objective(self, spec: SpecIR) -> str:
        """Reverse-KL is mode-seeking and less hallucinatory (MiniLLM 2306.08543).

        Majestic selects reverse-KL for extraction and classification, and
        forward-KL for open-ended drafting where coverage matters (A-02).
        """
        return "forward_kl" if spec.task_primitive in _COVERAGE_PRIMITIVES else "reverse_kl"

    def _plan_for_base(self, spec: SpecIR, base: BaseModel) -> BuildPlanIR:
        """Build a complete plan around one chosen base."""
        prim = spec_for(spec.task_primitive)

        # Sequence-level KD is the default: it crosses any tokenizer boundary
        # and composes with synthetic data generation (A-02).
        distil_mode = "sequence_kd" if spec.seed_data_count < prim.seed_floor * 4 else "none"
        teacher = self.catalogue.teacher_for(base, distil_mode) if distil_mode != "none" else None

        if self._needs_full_finetune(spec):
            peft = "full_ft"
        else:
            peft = "qlora" if base.params_b >= 4.0 else "lora"

        bit_width = "int4"     # optimal accuracy-per-bit under a RAM budget (A-01)
        quantiser = "k_quant" if spec.device_target.startswith(("android", "raspberry")) else "awq"
        target = (
            "gguf" if spec.device_target.startswith(("android", "raspberry", "laptop_cpu"))
            else "onnx"
        )
        if spec.offline_required and target == "vllm":
            target = "gguf"

        budget = round(_COST_PER_PARAM_B.get(peft, 4.0) * max(base.params_b, 0.5), 2)

        return BuildPlanIR(
            spec_hash=spec.hash,
            base_ref=base.ref,
            teacher_ref=teacher.ref if teacher else None,
            distil_mode=distil_mode,
            peft_method=peft,
            rank=16 if base.params_b < 4 else 32,
            data_recipe={
                "seed_floor": prim.seed_floor,
                "amplification": "backtranslate+evolve",
                "kl_objective": self._kl_objective(spec),
                "requires_network": False,
            },
            quantiser=quantiser,
            bit_width=bit_width,
            grammar_ref=f"grammar:{spec.task_primitive.value}" if prim.structured_output else None,
            eval_suite_ref=f"eval:{spec.task_primitive.value}",
            target=target,
            budget_usd=budget,
            provenance={
                "selected_by": "deterministic",
                "base_cap_params_b": self._device_cap(spec),
                "ladder_tier": ladder_tier(base.params_b),
                "rule": "largest base that fits at 4-bit (A-01 k-bit scaling law)",
            },
        )

    def _default_plan(self, spec: SpecIR) -> BuildPlanIR:
        """Pick the LARGEST sufficient base under the device cap (A-01)."""
        return self._plan_for_base(spec, self._eligible_bases(spec)[0])

    def candidate_plans(self, spec: SpecIR, limit: int = 2) -> list[BuildPlanIR]:
        """Top-N plans to build IN PARALLEL when the choice is uncertain (B-01).

        C-02 acts 5-6: two candidates are trained in parallel and the Proving
        Ground picks the winner, because the largest base that fits RAM may still
        break the latency budget. Ordering is largest-first, so candidate 0 is the
        most capable option and later candidates are the cheaper fallbacks.
        """
        return [self._plan_for_base(spec, b) for b in self._eligible_bases(spec)[:limit]]

    # -- mutation on validator rejection ---------------------------------- #
    def _next_smaller(self, plan: BuildPlanIR) -> list[BaseModel]:
        """Dense bases smaller than the current one, largest first.

        Downsizing is the response to a validator rejection, not the starting
        point: the plan begins at the largest base that fits (A-01) and steps
        down only when a hard predicate says it must.
        """
        current = self.catalogue.base(plan.base_ref)
        ceiling = current.params_b if current else float("inf")
        return sorted(
            (b for b in self.catalogue.bases if b.params_b < ceiling and not b.is_moe),
            key=lambda b: -b.params_b,
        )

    def _mutate(self, plan: BuildPlanIR, gate: GateResult) -> tuple[BuildPlanIR, str] | None:
        """Apply one targeted repair for the first blocking reason."""
        reasons = " ".join(gate.reasons).lower()

        if "ram" in reasons or "file-size rule" in reasons:
            smaller = self._next_smaller(plan)
            if smaller:
                plan.base_ref = smaller[0].ref
                plan.budget_usd = round(
                    _COST_PER_PARAM_B.get(plan.peft_method, 4.0) * max(smaller[0].params_b, 0.5), 2
                )
                return plan, f"downsized base to {smaller[0].ref}"
            if plan.bit_width != "int4":
                plan.bit_width = "int4"
                return plan, "dropped to int4"
            return None

        if "identical tokenizer" in reasons or "logit distillation" in reasons:
            plan.distil_mode = "sequence_kd"
            return plan, "switched to sequence-level distillation (crosses tokenizers)"

        if "latency" in reasons:
            smaller = self._next_smaller(plan)
            if smaller:
                plan.base_ref = smaller[0].ref
                return plan, f"downsized base for latency to {smaller[0].ref}"
            return None

        if "offline closure" in reasons and plan.target == "vllm":
            plan.target = "gguf"
            return plan, "retargeted to gguf for offline closure"

        if "budget" in reasons or "tier ceiling" in reasons:
            if plan.peft_method == "full_ft":
                plan.peft_method = "qlora"
                return plan, "switched full fine-tune to QLoRA to fit the budget"
            base = self.catalogue.base(plan.base_ref)
            if base:
                plan.budget_usd = round(_COST_PER_PARAM_B["qlora"] * max(base.params_b, 0.5), 2)
                plan.peft_method = "qlora"
                return plan, "recosted with QLoRA"
            return None

        return None  # licence and data-rights failures are not mutable by the planner

    # -- the public entry point ------------------------------------------- #
    def plan(self, spec: SpecIR) -> PlanningResult:
        """Produce an admitted Build Plan IR, or a gate result explaining why not."""
        precedent = self._precedent(spec)
        if precedent is not None:
            plan = BuildPlanIR.from_dict({**precedent.to_dict(), "spec_hash": spec.hash})
            path = "precedent"
            logger.info("planner: reusing precedent plan for %s", spec.hash[:8])
        elif self.proposer is not None:
            plan = self.proposer(spec, self.catalogue)   # LLM role 5 — proposes
            path = "proposed"
            logger.info("planner: LLM proposed a plan for %s", spec.hash[:8])
        else:
            plan = self._default_plan(spec)
            path = "deterministic"

        mutations: list[str] = []
        gate = gate2_plan_feasibility(spec, plan, self.profiler, self.catalogue)

        # The validator disposes: mutate and re-validate until admitted or stuck.
        for _ in range(_MAX_MUTATIONS):
            if gate.passed:
                break
            mutated = self._mutate(plan, gate)
            if mutated is None:
                break
            plan, note = mutated
            mutations.append(note)
            logger.info("planner: mutating plan — %s", note)
            gate = gate2_plan_feasibility(spec, plan, self.profiler, self.catalogue)

        if not gate.passed:
            logger.warning("planner: no admissible plan for %s: %s", spec.hash[:8], gate.reasons)
            return PlanningResult(None, gate, path, mutations)

        return PlanningResult(plan, gate, path, mutations)
