"""The Planner core: enumeration, base selection, and ``plan()`` (§7, §12).

    plan(s, catalog, history) -> BuildPlanIR | Refusal

**Deterministic on the common path.** Given the same spec, catalogue and history
it returns the same plan, with no language model involved. That is not a
stylistic preference — it is what makes the system auditable, cacheable and
reproducible.

**Why enumeration is correct, not merely affordable.** The plan space is about
1.4e6 points, which is enumerable with pruning in milliseconds. NAS operates over
1e18 or more and is forced into RL or evolutionary search. But §12 makes the
stronger argument: for the dominant dimension there is *nothing to enumerate*.
:func:`select_base` is a maximum over a chain — six comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Optional, Sequence

from majestic.logging_utils import get_logger
from modelrig.ir import BuildPlanIR, SpecIR
from modelrig import quantformat
from modelrig.planner import objective, predicates
from modelrig.catalogue import ladder_tier
from modelrig.planner.catalog import Catalog, ModelSpec, default_catalog
from modelrig.planner.costmodel import USD, usd
from modelrig.planner.metalearn import OutcomePredictor, PrecedentIndex, features_of
from modelrig.planner.objective import Tier
from modelrig.planner.predicates import (
    ALL_PREDICATES,
    PredicateResult,
    derive_distil_mode,
    ordered_predicates,
    seed_floor,
)
from modelrig.planner.refusal import Refusal, WitnessSet, build_refusal, quality_refusal
from modelrig.primitives import spec_for

logger = get_logger(__name__)

#: The grid the enumerator walks for the non-base dimensions.
PEFT_METHODS = ("lora", "qlora", "dora", "full_ft")
RANKS = (8, 16, 32, 64)
QUANTISERS = ("q4_k_m", "awq_int4", "gptq_int4", "spqr_int4", "int8")
TARGETS = ("gguf", "onnx", "executorch", "coreml", "vllm")

#: Novel-vocabulary share above which LoRA is not enough (A-03, Biderman 2405.09673).
NOVEL_VOCAB_FULL_FT = 0.30


@dataclass
class PlanCandidate:
    """One point of the plan space, with a stable identity for witnesses."""

    base: ModelSpec
    teacher: Optional[ModelSpec]
    distil_mode: str
    peft_method: str
    rank: int
    quantiser: str
    bit_width: str
    target: str
    data_recipe: dict[str, Any] = field(default_factory=dict)

    # The predicates accept anything with a BuildPlanIR's shape, so a candidate
    # can be checked before it is ever lowered into IR.
    @property
    def base_ref(self) -> str:
        return self.base.ref

    @property
    def teacher_ref(self) -> str | None:
        return self.teacher.ref if self.teacher else None

    @property
    def ident(self) -> str:
        return (
            f"{self.base.ref}|{self.teacher.ref if self.teacher else '-'}|{self.distil_mode}"
            f"|{self.peft_method}|r{self.rank}|{self.quantiser}|{self.target}"
        )

    def to_ir(self, spec: SpecIR, budget_micro: int) -> BuildPlanIR:
        prim = spec_for(spec.task_primitive)
        return BuildPlanIR(
            spec_hash=spec.hash,
            base_ref=self.base.ref,
            teacher_ref=self.teacher.ref if self.teacher else None,
            distil_mode=self.distil_mode,
            peft_method=self.peft_method,
            rank=self.rank,
            data_recipe=dict(self.data_recipe),
            quantiser=self.quantiser,
            bit_width=self.bit_width,
            grammar_ref=(
                f"grammar:{spec.task_primitive.value}" if prim.structured_output else None
            ),
            eval_suite_ref=f"eval:{spec.task_primitive.value}",
            target=self.target,
            budget_usd=usd(budget_micro),
            provenance={
                "selected_by": "deterministic_enumeration",
                "rule": "largest base that fits at 4-bit (A-01 k-bit scaling law)",
                "ladder_tier": ladder_tier(self.base.params_b),
                "distil_derived_from": "tokenizer identity (P_tok is definitional)",
                "params_b": round(self.base.params_b, 3),
            },
        )


# =========================================================================== #
# §12 — base selection is a maximum over a chain, not a search
# =========================================================================== #
def select_base(
    spec: SpecIR, catalog: Catalog, *, quantiser: str = "q4_k_m", allow_moe: bool = False
) -> ModelSpec | None:
    """The largest base that fits, found in ``|B|`` comparisons.

    **The structural result.** Order bases by parameter count. ``P_ram`` is
    *antitone* in that order: if a 4B fits then every smaller base fits, because
    both the weight term and the KV term are monotone in size. So the feasible
    set is a **down-set in a total order** — a chain — and therefore has a unique
    maximal element.

    Combine that with the k-bit scaling law (under a fixed memory budget the
    largest base at 4-bit beats any smaller base at higher precision) and base
    selection reduces to

        b* = max { b : P_ram(s, b, w=4) }

    A maximum over a chain. Six comparisons, no enumeration. This is *why*
    rejecting search is correct rather than merely affordable: for the dominant
    dimension there is nothing to search.

    Returns ``None`` when even the smallest base fails — which is itself
    information, and the caller turns it into a witness.
    """
    chain = catalog.bases(spec.task_primitive, allow_moe=allow_moe)
    if not chain:
        return None

    prim = spec_for(spec.task_primitive)
    # Walk the chain downward and take the first that fits: because P_ram is
    # antitone, the first hit from the top IS the maximum.
    for model in reversed(chain):
        if model.params_b < prim.min_params_b:
            continue
        probe = PlanCandidate(
            base=model, teacher=None, distil_mode="none", peft_method="lora",
            rank=16, quantiser=quantiser, bit_width="int4", target="gguf",
        )
        if predicates.P_ram(spec, probe, catalog):
            return model
    return None


def verify_antitone(spec: SpecIR, catalog: Catalog, quantiser: str = "q4_k_m") -> bool:
    """Assert the monotonicity §12 relies on, over the live catalogue.

    Cheap enough to run in tests and worth running: if a catalogue entry ever
    breaks antitonicity — an exotic architecture whose KV geometry dominates its
    parameter count — then :func:`select_base` silently stops being a maximum and
    starts being a guess.
    """
    chain = catalog.bases(spec.task_primitive)
    fits = []
    for model in chain:
        probe = PlanCandidate(
            base=model, teacher=None, distil_mode="none", peft_method="lora",
            rank=16, quantiser=quantiser, bit_width="int4", target="gguf",
        )
        fits.append(bool(predicates.P_ram(spec, probe, catalog)))
    # A down-set in a total order: every True precedes every False.
    seen_false = False
    for ok in fits:
        if not ok:
            seen_false = True
        elif seen_false:
            return False
    return True


# =========================================================================== #
# Enumeration
# =========================================================================== #
def _grid(spec: SpecIR, base: ModelSpec, catalog: Catalog) -> Iterable[tuple]:
    """The non-base dimensions, ordered so the best candidate appears first."""
    io = spec.io_schema if isinstance(spec.io_schema, dict) else {}
    novel = float(io.get("novel_vocabulary_ratio", 0.0))

    if novel >= NOVEL_VOCAB_FULL_FT:
        # A-03: LoRA learns less and forgets less. It underperforms full
        # fine-tuning when a genuinely NEW domain must be learned, so escalate —
        # but only on a declared threshold, and the regression suite runs either way.
        methods = ("full_ft", "qlora")
    else:
        methods = ("qlora", "lora") if base.params_b >= 4.0 else ("lora", "qlora")

    if spec.device_target.startswith(("android", "raspberry")):
        targets, quants = ("gguf", "executorch"), ("q4_k_m", "awq_int4", "int8")
    elif spec.device_target.startswith("laptop"):
        targets, quants = ("gguf", "onnx"), ("q4_k_m", "awq_int4", "int8")
    else:
        targets, quants = ("onnx", "vllm"), ("awq_int4", "gptq_int4", "int8")
    if spec.offline_required:
        targets = tuple(t for t in targets if t != "vllm") or ("gguf",)

    # Part 3 §10: the accelerator decides which container can be LOADED, and the
    # lines above infer it from the device's NAME — the guess the probe exists to
    # replace. Two units in the same class can differ in whether an NPU delegate
    # is usable at all, so a measured accelerator overrides the prefix.
    probed = _probed_accelerator(spec)
    if probed is not None:
        loadable = quantformat.target_formats_for(
            probed, offline=spec.offline_required, allowed=TARGETS
        )
        # Keep the class ordering where the two agree; fall back to what the
        # accelerator can actually load when they do not.
        targets = tuple(t for t in targets if t in loadable) or tuple(loadable)

    ranks = (16, 32) if base.params_b < 4.0 else (32, 64)
    for method in methods:
        for quant in quants:
            for rank in ranks:
                for target in targets:
                    yield method, rank, quant, target


def _probed_accelerator(spec: SpecIR) -> str | None:
    """The accelerator a probe actually measured, or ``None`` when unprobed.

    An assumed profile does not count: the whole point of §10 is that this
    coordinate stops being inferred from the device tier once someone measures it.
    """
    profile = spec.device_profile
    if not isinstance(profile, dict) or not spec.profile_source.measured:
        return None
    accelerator = profile.get("accelerator")
    return str(accelerator) if accelerator else None


def _data_recipe(spec: SpecIR, base: ModelSpec) -> dict[str, Any]:
    prim = spec_for(spec.task_primitive)
    n_syn = max(spec.seed_data_count * 40, 8_000)
    return {
        "seed_floor": seed_floor(spec.task_primitive),
        "amplification": "backtranslate+evolve",
        "n_synthetic": n_syn,
        "n_train_tokens": n_syn * 320,
        "n_eval_examples": max(int(spec.seed_data_count * 0.25), 20),
        "kl_objective": (
            "forward_kl" if spec.task_primitive.value in {"generate", "summarise", "rewrite"}
            else "reverse_kl"
        ),
        "requires_network": False,
        "local_teacher": spec.offline_required,
        "_primitive_metric": prim.default_metric,
    }


def _bit_width_of(quantiser: str) -> str:
    return "int8" if quantiser == "int8" else ("fp16" if quantiser == "fp16" else "int4")


@dataclass
class Enumeration:
    """What enumeration found, feasible and otherwise."""

    feasible: list[PlanCandidate] = field(default_factory=list)
    witnesses: WitnessSet = field(default_factory=WitnessSet)
    considered: int = 0
    predicate_evaluations: int = 0


def enumerate_feasible(
    spec: SpecIR,
    catalog: Catalog,
    *,
    max_bases: int = 3,
    allow_moe: bool = False,
    stop_after: int | None = None,
) -> Enumeration:
    """Walk the plan space, pruning with the ordered predicates.

    ``max_bases`` bounds how far down the chain to descend from ``b*``. The
    largest feasible base is the one the k-bit law wants; the next one or two
    down exist so that a *latency* or *cost* refusal has a fallback to name.
    """
    out = Enumeration()
    order = ordered_predicates()

    chain = catalog.bases(spec.task_primitive, allow_moe=allow_moe)
    if not chain:
        return out
    prim = spec_for(spec.task_primitive)
    eligible = [m for m in reversed(chain) if m.params_b >= prim.min_params_b] or [chain[-1]]
    bases = eligible[:max_bases]

    for base in bases:
        teachers: list[ModelSpec | None] = [None]
        same_family = catalog.teachers(tokenizer=base.tokenizer)
        cross_family = [t for t in catalog.teachers() if t.tokenizer != base.tokenizer]
        # Same-family first: it is the only way logit KD is even defined.
        teachers = [*same_family[:1], *cross_family[:1], None]

        for teacher in teachers:
            distil = derive_distil_mode(base, teacher)
            for method, rank, quant, target in _grid(spec, base, catalog):
                candidate = PlanCandidate(
                    base=base, teacher=teacher, distil_mode=distil,
                    peft_method=method, rank=rank, quantiser=quant,
                    bit_width=_bit_width_of(quant), target=target,
                    data_recipe=_data_recipe(spec, base),
                )
                out.considered += 1
                failed = False
                for p in order:
                    out.predicate_evaluations += 1
                    result: PredicateResult = p(spec, candidate, catalog)
                    if not result.ok:
                        out.witnesses.add(candidate.ident, result)
                        failed = True
                        break            # early exit — the §4 saving
                if not failed:
                    out.feasible.append(candidate)
                    if stop_after and len(out.feasible) >= stop_after:
                        return out
    return out


# =========================================================================== #
# Ranking and the decision
# =========================================================================== #
def _score(
    spec: SpecIR, candidate: PlanCandidate, catalog: Catalog,
    predictor: OutcomePredictor | None,
) -> tuple[float, float, int]:
    """Return ``(utility, predicted_quality, cost_micro)`` for a feasible plan."""
    from modelrig.planner import costmodel

    prim = spec_for(spec.task_primitive)
    bits = catalog.bits_effective(candidate.base, candidate.quantiser)
    recipe = candidate.data_recipe

    cost = costmodel.estimate(
        candidate.base, method=candidate.peft_method,
        n_synthetic=int(recipe.get("n_synthetic", 0)),
        n_train_tokens=int(recipe.get("n_train_tokens", 0)),
        n_eval_examples=int(recipe.get("n_eval_examples", 0)),
        uses_teacher=candidate.teacher is not None,
    ).total

    prior = objective.quality_prior(
        params_b=candidate.base.params_b, min_params_b=prim.min_params_b,
        seed_count=spec.seed_data_count, seed_floor=seed_floor(spec.task_primitive),
        distil_mode=candidate.distil_mode, bit_width_effective=bits,
    )
    quality = prior
    if predictor is not None:
        quality = predictor.predict(prior, features_of(
            params_b=candidate.base.params_b,
            seed_ratio=spec.seed_data_count / max(seed_floor(spec.task_primitive), 1),
            bits_effective=bits, distil_mode=candidate.distil_mode,
            context=2048, quality_gate=spec.quality_gate,
        ))

    device = catalog.device(spec.device_target)
    lat = predicates.latency(
        candidate.base, catalog, quantiser=candidate.quantiser, device=device,
        tokens_in=512, tokens_out=64,
    )
    weights = predicates.weight_bytes(candidate.base, catalog, candidate.quantiser)
    terms = objective.utility(
        quality=quality,
        cost_micro=cost, cost_ceiling_micro=int(spec.budget_ceiling_usd * USD),
        latency_ms=lat.total_ms if lat.total_ms != float("inf") else 0.0,
        latency_budget_ms=float(spec.latency_budget_ms) if spec.latency_budget_ms else None,
        weight_bytes=weights, weight_ceiling_bytes=device.ram_budget,
    )
    return terms.total, quality, cost


@dataclass
class PlanOutcome:
    """Either an admitted plan or a refusal, plus how it was reached."""

    plan: Optional[BuildPlanIR] = None
    refusal: Optional[Refusal] = None
    path: str = "deterministic"          # precedent | deterministic | refused
    utility: float = 0.0
    predicted_quality: float = 0.0
    threshold: float = 0.0
    cost_micro: int = 0
    considered: int = 0
    predicate_evaluations: int = 0
    candidates: list[BuildPlanIR] = field(default_factory=list)

    @property
    def admitted(self) -> bool:
        return self.plan is not None


def plan(
    spec: SpecIR,
    catalog: Catalog | None = None,
    *,
    precedents: PrecedentIndex | None = None,
    predictor: OutcomePredictor | None = None,
    tier: Tier = Tier.COMMERCIAL,
    allow_moe: bool = False,
) -> PlanOutcome:
    """The §7 algorithm. Returns an admitted plan or a refusal — never a guess.

    Parallel candidates are attached **only** when the spec asks for them. §15
    shows that running k candidates is strictly worse in expected cost at every
    pass probability — ``E[cost]_2 / E[cost]_1 = 2/(2-p) > 1`` — so the Planner
    must never enable them on its own initiative. What they buy is wall-clock
    time and variance reduction, which is the customer's trade to make.
    """
    catalog = catalog or default_catalog()

    # -- lines 1-3: deterministic warm start, no LLM ---------------------- #
    if precedents is not None:
        hit = precedents.nearest(spec_shape(spec))
        if hit is not None:
            adapted = BuildPlanIR.from_dict({**hit.plan.to_dict(), "spec_hash": spec.hash})
            candidate = _candidate_from_ir(adapted, catalog)
            if candidate is not None and all(
                p(spec, candidate, catalog).ok for p in ordered_predicates()
            ):
                logger.info("planner: reusing precedent for %s", spec.hash[:8])
                return PlanOutcome(plan=adapted, path="precedent")

    # -- lines 4-14: enumerate with early exit ---------------------------- #
    enum = enumerate_feasible(spec, catalog, allow_moe=allow_moe)

    # -- lines 15-16: refuse with a witness ------------------------------- #
    if not enum.feasible:
        notes = []
        device = catalog.devices.get(spec.device_target)
        if device is not None and not device.measured:
            notes.append(
                f"latency_source = {device.latency_source} for {device.name}. "
                "These estimates require device-lab validation before any commitment."
            )
        refusal = build_refusal(spec.hash, enum.witnesses, enum.considered, notes)
        logger.warning("planner: refusing %s — witness %s", spec.hash[:8], refusal.witness)
        return PlanOutcome(
            refusal=refusal, path="refused", considered=enum.considered,
            predicate_evaluations=enum.predicate_evaluations,
        )

    # -- line 17: rank ----------------------------------------------------- #
    scored = [(*_score(spec, c, catalog, predictor), c) for c in enum.feasible]
    scored.sort(key=lambda t: (-t[0], t[3].base.params, t[3].ident))
    utility, quality, cost, best = scored[0]

    # -- line 18: the soft refusal ---------------------------------------- #
    theta = objective.refusal_threshold(cost, tier)
    if quality < theta:
        logger.warning(
            "planner: refusing %s on predicted quality %.2f < theta* %.2f",
            spec.hash[:8], quality, theta,
        )
        return PlanOutcome(
            refusal=quality_refusal(
                spec.hash, quality, theta, enum.considered,
                remedies=[
                    "supply more real seed data to raise predicted quality",
                    "relax the quality gate if the use case tolerates it",
                    "move to a device tier that admits a larger base",
                ],
            ),
            path="refused", predicted_quality=quality, threshold=theta,
            considered=enum.considered, predicate_evaluations=enum.predicate_evaluations,
        )

    # -- line 19: candidates ONLY on explicit request (§15) ---------------- #
    admitted = best.to_ir(spec, cost)
    extra: list[BuildPlanIR] = []
    if _wants_parallel(spec):
        extra = [
            c.to_ir(spec, cst) for (_u, _q, cst, c) in _diverse_alternatives(scored[1:])
        ]

    logger.info(
        "planner: admitted %s for %s (U=%.3f, Q=%.2f >= theta*=%.2f, $%.2f) "
        "after %d candidates / %d predicate evaluations",
        best.base.ref, spec.hash[:8], utility, quality, theta, usd(cost),
        enum.considered, enum.predicate_evaluations,
    )
    return PlanOutcome(
        plan=admitted, path="deterministic", utility=utility,
        predicted_quality=quality, threshold=theta, cost_micro=cost,
        considered=enum.considered, predicate_evaluations=enum.predicate_evaluations,
        candidates=extra,
    )


def _wants_parallel(spec: SpecIR) -> bool:
    """§15: parallel candidates are opt-in, never planner-initiated."""
    io = spec.io_schema if isinstance(spec.io_schema, dict) else {}
    return bool(io.get("allow_parallel_candidates", False))


def _diverse_alternatives(rest: Sequence[tuple], limit: int = 1) -> list[tuple]:
    """Pick alternates that fail for *different* reasons (§15 consequence 2).

    Correlated failures make parallelism worse than the independence analysis
    suggests: if both candidates die of insufficient seed data then
    ``P[both fail] > prod(1-p_i)`` and the cost ratio degrades further. So vary
    **base family and data recipe**, not rank and bit width.
    """
    out: list[tuple] = []
    seen_families: set[str] = set()
    for entry in rest:
        candidate = entry[3]
        family = candidate.base.tokenizer
        if family in seen_families:
            continue
        seen_families.add(family)
        out.append(entry)
        if len(out) >= limit:
            break
    return out


def _candidate_from_ir(ir: BuildPlanIR, catalog: Catalog) -> PlanCandidate | None:
    try:
        base = catalog.model(ir.base_ref)
        teacher = catalog.model(ir.teacher_ref) if ir.teacher_ref else None
    except Exception:  # noqa: BLE001 - an off-catalogue precedent is simply unusable
        return None
    return PlanCandidate(
        base=base, teacher=teacher, distil_mode=ir.distil_mode,
        peft_method=ir.peft_method, rank=ir.rank, quantiser=ir.quantiser,
        bit_width=ir.bit_width, target=ir.target, data_recipe=dict(ir.data_recipe),
    )


def spec_shape(spec: SpecIR) -> tuple:
    """The coarse shape used for precedent matching — not the exact hash."""
    return (
        spec.task_primitive.value,
        spec.device_target,
        spec.offline_required,
        round(spec.quality_gate, 2),
    )


__all__ = [
    "Enumeration", "PlanCandidate", "PlanOutcome",
    "ALL_PREDICATES", "enumerate_feasible", "plan", "select_base",
    "spec_shape", "verify_antitone", "replace",
]
