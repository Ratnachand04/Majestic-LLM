"""The three verification gates (B-03) — checked BEFORE any GPU is spent.

    Gate 1  spec admissibility     Spec IR      -> may we attempt this at all?
    Gate 2  plan feasibility       Build Plan   -> will this physically work?
    Gate 3  artefact certification Artefact     -> did it actually work?

This is the discipline the ML-platform competition does not have. Every other
fine-tuning service accepts a config file and starts a GPU job; Majestic
type-checks the request against physics, law and its own catalogue first and
refuses impossible builds before they cost anything.

**A refusal in eight seconds is a better product than a failed build in ninety
minutes.** An infeasible plan caught at Gate 2 costs nothing; caught after
training it costs sixty dollars.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from majestic.logging_utils import get_logger
from modelrig.buildspec import BuildSpec, DeviceProfile
from modelrig.catalogue import DEFAULT_CATALOGUE, Catalogue
from modelrig.feasibility import HeuristicFeasibilityEngine
from modelrig.ir import AbstentionPolicy, ArtefactIR, BuildPlanIR, SpecIR
from modelrig.licence import resolve_licence_chain
from modelrig.primitives import spec_for

logger = get_logger(__name__)


@dataclass
class GateResult:
    """Outcome of one gate: pass/fail plus every reason and any warnings."""

    gate: str
    passed: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    evidence: dict[str, object] = field(default_factory=dict)

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise ValueError(
                f"{self.gate} failed:\n  - " + "\n  - ".join(self.reasons)
            )


# --------------------------------------------------------------------------- #
def gate1_spec_admissibility(
    spec: SpecIR, profiler=None, catalogue: Catalogue | None = None
) -> GateResult:
    """Gate 1 — may we attempt this build at all?

    · Is the primitive inside the supported set of eight?
    · Does the seed-data volume clear the floor for this primitive?
    · Are the data rights sufficient to train and to deploy?
    · Is the quality bar achievable at this device tier at all?
    """
    catalogue = catalogue or DEFAULT_CATALOGUE
    reasons: list[str] = []
    warnings: list[str] = []
    evidence: dict[str, object] = {}

    # 1. primitive inside the closed set
    try:
        prim = spec_for(spec.task_primitive)
    except (ValueError, KeyError) as exc:
        return GateResult("gate1_spec_admissibility", False, [str(exc)])
    evidence["primitive"] = prim.primitive.value

    # 2. real-seed floor (B-06: below it, amplification collapses)
    evidence["seed_floor"] = prim.seed_floor
    evidence["seed_data_count"] = spec.seed_data_count
    if spec.seed_data_count < prim.seed_floor:
        reasons.append(
            f"seed data {spec.seed_data_count} is below the floor of "
            f"{prim.seed_floor} real examples for primitive "
            f"{prim.primitive.value!r}; amplifying from fewer risks irreversible "
            "tail loss (Curse of Recursion 2305.17493)"
        )

    # 3. data rights sufficient to train AND deploy
    chain = resolve_licence_chain(
        base_licence="apache-2.0",  # provisional; Gate 2 resolves the real base
        teacher_licence=None,
        data_rights=spec.data_rights,
        jurisdiction=spec.jurisdiction,
    )
    if not chain.permitted:
        reasons.extend(chain.reasons)

    # 4. quality bar achievable at this device tier
    device: DeviceProfile | None = None
    if profiler is not None:
        try:
            device = profiler.profile(spec.device_target)
        except KeyError as exc:
            reasons.append(str(exc))
    if device is not None:
        engine = HeuristicFeasibilityEngine(profiler=profiler)
        max_b = engine.max_base_for(device)
        evidence["max_base_params_b"] = max_b
        if max_b < prim.min_params_b:
            reasons.append(
                f"quality bar unachievable: {prim.primitive.value!r} needs at least "
                f"{prim.min_params_b}B but {spec.device_target} caps the base at "
                f"{max_b}B at 4-bit"
            )
        if not device.measured:
            warnings.append(
                f"device profile {device.name!r} is a heuristic, not a lab "
                "measurement (GAP-10)"
            )

    # 5. step depth the tier can sustain (A-09 capability ceiling)
    if spec.max_step_depth > prim.max_step_depth:
        reasons.append(
            f"step depth {spec.max_step_depth} exceeds {prim.max_step_depth}, the "
            f"maximum {prim.primitive.value!r} sustains at small scale (A-09)"
        )

    # 6. internal contradiction: offline plus escalation
    if spec.offline_required and spec.abstention_policy is AbstentionPolicy.ESCALATE:
        reasons.append(
            "offline_required=True contradicts abstention_policy='escalate': a "
            "customer who bought offline gets offline (B-10)"
        )

    if spec.quality_gate >= 0.99:
        warnings.append(
            f"quality_gate {spec.quality_gate} is near-perfect; gates above 0.98 "
            "are rarely achievable on real held-out data"
        )

    return GateResult("gate1_spec_admissibility", not reasons, reasons, warnings, evidence)


# --------------------------------------------------------------------------- #
def gate2_plan_feasibility(
    spec: SpecIR, plan: BuildPlanIR, profiler=None, catalogue: Catalogue | None = None
) -> GateResult:
    """Gate 2 — will this plan physically, legally and financially work?

    · Does base size at this bit-width fit device RAM, WITH the KV cache?
    · Does predicted latency clear the budget?
    · Tokenizer identity verified if logit distillation is requested.
    · Offline closure: every node locally resolvable, zero API calls.
    · Licence chain composes for base, teacher, data and jurisdiction.
    · Predicted build cost is inside the customer's tier ceiling.
    """
    catalogue = catalogue or DEFAULT_CATALOGUE
    reasons: list[str] = []
    warnings: list[str] = []
    evidence: dict[str, object] = {}

    base = catalogue.base(plan.base_ref)
    if base is None:
        reasons.append(f"base {plan.base_ref!r} is not in the catalogue")

    # --- memory / latency, including the KV cache (A-01) ------------------ #
    if profiler is not None and base is not None:
        try:
            device = profiler.profile(spec.device_target)
        except KeyError as exc:
            reasons.append(str(exc))
        else:
            engine = HeuristicFeasibilityEngine(profiler=profiler)
            probe = BuildSpec(
                task=spec.task_primitive.value,
                base_model=plan.base_ref,
                quantization=plan.bit_width,
                runtime=plan.target,
                device=device,
                extras={
                    "params_b": base.params_b,
                    "context_length": int(
                        spec.io_schema.get("context_length", 2048)
                        if isinstance(spec.io_schema, dict) else 2048
                    ),
                    **({"latency_budget_ms": spec.latency_budget_ms}
                       if spec.latency_budget_ms else {}),
                },
            )
            verdict = engine.evaluate(probe)
            evidence["feasibility"] = verdict.guarantee
            evidence["kv_cache_mb"] = verdict.estimate.kv_cache_mb
            if not verdict.feasible:
                reasons.extend(verdict.reasons)
            if not verdict.estimate.measured:
                warnings.append("performance figures are predicted, not measured (GAP-10)")

    # --- tokenizer identity for logit distillation (A-02, GAP-07) --------- #
    if plan.distil_mode == "logit_kd":
        teacher = catalogue.teacher(plan.teacher_ref) if plan.teacher_ref else None
        if teacher is None:
            reasons.append("logit distillation requested but no catalogue teacher is set")
        elif base is not None and teacher.tokenizer_family != base.tokenizer_family:
            reasons.append(
                f"logit distillation requires an identical tokenizer: base family "
                f"{base.tokenizer_family!r} != teacher family "
                f"{teacher.tokenizer_family!r}. Use sequence_kd, which crosses any "
                "tokenizer boundary (A-02)"
            )

    # --- offline closure --------------------------------------------------- #
    if spec.offline_required:
        if plan.distil_mode != "none" and plan.teacher_ref:
            # Distillation happens at BUILD time, not at inference; only note it.
            evidence["teacher_used_at_build_time_only"] = True
        online_target = plan.target in {"vllm"}
        if online_target:
            reasons.append(
                f"offline closure broken: target {plan.target!r} is a server runtime"
            )
        if plan.data_recipe.get("requires_network"):
            reasons.append("offline closure broken: data recipe requires network access")

    # --- licence chain ----------------------------------------------------- #
    teacher_licence = None
    if plan.teacher_ref:
        t = catalogue.teacher(plan.teacher_ref)
        teacher_licence = t.licence if t else "unknown"
    chain = resolve_licence_chain(
        base_licence=base.licence if base else "unknown",
        teacher_licence=teacher_licence,
        data_rights=spec.data_rights,
        jurisdiction=spec.jurisdiction,
    )
    evidence["licence_chain"] = chain.as_record()
    if not chain.permitted:
        reasons.extend(chain.reasons)

    # --- budget ------------------------------------------------------------ #
    if plan.budget_usd > spec.budget_ceiling_usd:
        reasons.append(
            f"predicted build cost ${plan.budget_usd:.0f} exceeds the tier ceiling "
            f"${spec.budget_ceiling_usd:.0f}"
        )

    # --- primitive still inside the set ------------------------------------ #
    try:
        spec_for(spec.task_primitive)
    except (ValueError, KeyError) as exc:
        reasons.append(str(exc))

    return GateResult("gate2_plan_feasibility", not reasons, reasons, warnings, evidence)


# --------------------------------------------------------------------------- #
def gate3_artefact_certification(
    artefact: ArtefactIR, spec: SpecIR, eval_report: dict
) -> GateResult:
    """Gate 3 — did it actually work, on the customer's own data?

    · Eval suite passed on REAL held-out customer data.
    · Re-evaluated after quantisation; answer-flip rate within bound.
    · Regression, safety and privacy suites all clean.
    · Only then does the artefact enter the registry.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    evidence: dict[str, object] = {"eval": eval_report}

    if not eval_report:
        return GateResult("gate3_artefact_certification", False, ["no eval report"])

    # 1. the task gate on real held-out data
    if not eval_report.get("held_out_is_real", True):
        reasons.append("eval did not run on real held-out customer data")
    if not eval_report.get("passed", False):
        reasons.append(
            f"eval gate not met: {eval_report.get('metric')}="
            f"{eval_report.get('score')} < {eval_report.get('threshold')}"
        )

    # 2. post-quantisation re-evaluation + answer-flip rate (A-05)
    if artefact.quantised_blob_hash:
        if not eval_report.get("post_quantisation", False):
            reasons.append(
                "quantised artefact was not re-evaluated; compression changes "
                "behaviour even when aggregate accuracy holds (A-05)"
            )
        flip = eval_report.get("answer_flip_rate")
        bound = eval_report.get("answer_flip_bound", 0.10)
        if flip is None:
            reasons.append("answer-flip rate against FP16 was not measured (A-05)")
        elif flip > bound:
            reasons.append(
                f"answer-flip rate {flip:.3f} exceeds bound {bound:.3f}: aggregate "
                "parity is an illusion when individual answers change"
            )

    # 3. the non-negotiable suites
    for suite in ("regression", "safety", "privacy"):
        status = eval_report.get(f"{suite}_passed")
        if status is None:
            reasons.append(f"{suite} suite was not run")
        elif not status:
            reasons.append(f"{suite} suite failed")

    # 4. certification must be attached (B-08)
    if not artefact.model_card:
        reasons.append("model card missing; artefacts enter the registry certified")
    if not artefact.licence_chain:
        reasons.append("licence chain missing from the artefact")
    elif not artefact.licence_chain.get("permitted", False):
        reasons.append("licence chain does not permit release")

    # 5. calibration is a warning, not a blocker, below 2B (GAP-03)
    ece = eval_report.get("ece")
    if ece is not None and ece > 0.15:
        warnings.append(
            f"expected calibration error {ece:.3f} is high; routing, abstention and "
            "human review all depend on trustworthy confidence (GAP-03)"
        )

    return GateResult("gate3_artefact_certification", not reasons, reasons, warnings, evidence)
