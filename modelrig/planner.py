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
from modelrig.catalogue import DEFAULT_CATALOGUE, Catalogue
from modelrig.feasibility import HeuristicFeasibilityEngine, OutcomePredictor
from modelrig.gates import GateResult, gate2_plan_feasibility
from modelrig.ir import BuildPlanIR, SpecIR
from modelrig.primitives import spec_for

logger = get_logger(__name__)

# Cost model for the budget predicate, in USD (QLoRA on rented A100s, A-03).
_COST_PER_PARAM_B = {"lora": 4.0, "dora": 5.0, "qlora": 3.0, "full_ft": 25.0}
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
    def _default_plan(self, spec: SpecIR) -> BuildPlanIR:
        """Pick the cheapest sufficient parts under the device cap."""
        prim = spec_for(spec.task_primitive)

        # Cap the base by what the device can actually hold at 4-bit (A-01).
        max_b = float("inf")
        if self.profiler is not None:
            try:
                device = self.profiler.profile(spec.device_target)
                engine = HeuristicFeasibilityEngine(profiler=self.profiler)
                max_b = engine.max_base_for(device)
            except KeyError:
                pass

        candidates = [
            b for b in self.catalogue.bases_for(spec.task_primitive)
            if b.params_b <= max_b and b.params_b >= prim.min_params_b
        ]
        if not candidates:
            # Nothing fits: fall back to the smallest supporting base so the
            # validator can report a precise, numeric reason.
            candidates = self.catalogue.bases_for(spec.task_primitive) or list(
                self.catalogue.bases
            )
        base = candidates[0]

        # Sequence-level KD is the default: it crosses any tokenizer boundary
        # and composes with synthetic data generation (A-02).
        distil_mode = "sequence_kd" if spec.seed_data_count < prim.seed_floor * 4 else "none"
        teacher = self.catalogue.teacher_for(base, distil_mode) if distil_mode != "none" else None

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
                "requires_network": False,
            },
            quantiser=quantiser,
            bit_width=bit_width,
            grammar_ref=f"grammar:{spec.task_primitive.value}" if prim.structured_output else None,
            eval_suite_ref=f"eval:{spec.task_primitive.value}",
            target=target,
            budget_usd=budget,
            provenance={"selected_by": "deterministic", "base_cap_params_b": max_b},
        )

    # -- mutation on validator rejection ---------------------------------- #
    def _mutate(self, plan: BuildPlanIR, gate: GateResult) -> tuple[BuildPlanIR, str] | None:
        """Apply one targeted repair for the first blocking reason."""
        reasons = " ".join(gate.reasons).lower()

        if "ram" in reasons or "file-size rule" in reasons:
            smaller = sorted(
                (b for b in self.catalogue.bases
                 if b.params_b < (self.catalogue.base(plan.base_ref).params_b
                                  if self.catalogue.base(plan.base_ref) else 99)),
                key=lambda b: -b.params_b,
            )
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
            smaller = sorted(
                (b for b in self.catalogue.bases
                 if b.params_b < (self.catalogue.base(plan.base_ref).params_b
                                  if self.catalogue.base(plan.base_ref) else 99)),
                key=lambda b: -b.params_b,
            )
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
