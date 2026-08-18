"""The seven hard predicates (§3).

Each is **exact, cheap, and returns a reason**. They are the research
contribution; the optimiser that runs afterwards is ordinary.

    P_tok   tokenizer compatibility     definitional
    P_seed  data sufficiency            statistical
    P_lic   licence composition         legal
    P_ram   memory feasibility          physical
    P_off   offline closure             structural
    P_lat   latency feasibility         physical + epistemic
    P_cost  budget                      financial

No prior planner mixes constraint domains this way. auto-sklearn has no hard
feasibility at all and always returns a model; NAS optimises architecture alone;
FlexGen optimises a single objective with no legal or data constraints; HAT
models the device but not the law or the corpus.

**The hard/soft partition (§16).** ``P_ram``, ``P_tok`` and ``P_lic`` are sound
*by construction* — a model that does not fit does not fit, an undefined KL is
undefined, an illegal chain is illegal. Their precision is exactly 1 and no
experiment can inform it. ``P_seed``, ``P_lat`` and the quality gate rest on
claims that could be wrong, and they are the scientifically interesting ones.
:data:`HARD` and :data:`SOFT` carry that partition so evaluation can report them
separately instead of inflating one number with the other.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

from modelrig.ir import SpecIR
from modelrig.planner.licence_lattice import resolve as resolve_licence
from modelrig.planner.catalog import (
    EMBEDDER_BYTES,
    MB,
    RUNTIME_BYTES,
    Catalog,
    DeviceSpec,
    ModelSpec,
)
from modelrig.planner.costmodel import USD, usd
from modelrig.primitives import TaskPrimitive


class Soundness(str, Enum):
    """Whether a predicate's verdict is certain or inferential (§16)."""

    HARD = "hard"   # sound by construction; precision is 1 and unmeasurable
    SOFT = "soft"   # rests on a claim that could be wrong; must be calibrated


@dataclass(frozen=True)
class PredicateResult:
    """A verdict, and — when it fails — something the customer can act on."""

    ok: bool
    predicate: str
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    remedy: str = ""

    def __bool__(self) -> bool:
        return self.ok


PASS = PredicateResult(True, "")


class Predicate(Protocol):
    name: str
    cost: float
    pass_rate: float
    soundness: Soundness

    def __call__(self, spec: SpecIR, plan: Any, catalog: Catalog) -> PredicateResult: ...


@dataclass
class _P:
    """Concrete predicate: a callable plus the statistics that order it."""

    name: str
    cost: float           # relative evaluation cost
    pass_rate: float      # rho: typical fraction of candidates that survive
    soundness: Soundness
    fn: Callable[[SpecIR, Any, Catalog], PredicateResult]

    def __call__(self, spec: SpecIR, plan: Any, catalog: Catalog) -> PredicateResult:
        return self.fn(spec, plan, catalog)

    @property
    def selectivity(self) -> float:
        """The ordering key ``c / (1 - rho)`` from §4. Lower runs earlier."""
        if self.pass_rate >= 1.0:
            return math.inf
        return self.cost / (1.0 - self.pass_rate)


# =========================================================================== #
# P_tok — tokenizer compatibility (§3.3)
# =========================================================================== #
def derive_distil_mode(base: ModelSpec, teacher: ModelSpec | None) -> str:
    """Derive the distillation mode from tokenizer identity — never accept it.

    Logit distillation minimises ``KL(p_t || p_b)``, and KL requires both
    distributions to be defined **on the same sample space**. Different
    tokenizers induce different vocabularies, hence different sample spaces,
    hence an undefined objective. This is not a heuristic or a preference; it is
    a definitional constraint, so the planner *derives* the mode rather than
    letting a caller assert one.
    """
    if teacher is None:
        return "none"
    return "logit_kd" if base.tokenizer == teacher.tokenizer else "sequence_kd"


def _p_tok(spec: SpecIR, plan: Any, catalog: Catalog) -> PredicateResult:
    if plan.distil_mode != "logit_kd":
        return PredicateResult(True, "P_tok")
    if plan.teacher_ref is None:
        return PredicateResult(
            False, "P_tok",
            reason="logit distillation requested with no teacher",
            remedy="select a teacher, or use sequence-level distillation",
        )
    base = catalog.model(plan.base_ref)
    teacher = catalog.model(plan.teacher_ref)
    if base.tokenizer == teacher.tokenizer:
        return PredicateResult(True, "P_tok")
    return PredicateResult(
        False, "P_tok",
        reason=(
            f"logit distillation is undefined across tokenizers: base {base.ref} uses "
            f"{base.tokenizer!r}, teacher {teacher.ref} uses {teacher.tokenizer!r}. "
            "KL divergence requires one sample space."
        ),
        detail={"base_tokenizer": base.tokenizer, "teacher_tokenizer": teacher.tokenizer},
        remedy=(
            "use sequence-level distillation, which crosses any tokenizer boundary, "
            f"or pick a teacher in the {base.tokenizer!r} family"
        ),
    )


P_tok = _P("P_tok", cost=1.0, pass_rate=0.5, soundness=Soundness.HARD, fn=_p_tok)


# =========================================================================== #
# P_seed — data sufficiency (§3.4)
# =========================================================================== #
#: Miss probability tolerated per behavioural mode.
# HYPOTHESIS — R4: eta is a policy choice, not a measurement.
ETA = 0.05

#: Minimum mode mass that must be covered, per primitive.
# HYPOTHESIS — R4: these are the free parameters of the coverage argument. The
# collapse study measures the truth; they must NOT be tuned until the derived
# floor agrees with a previously asserted guess.
MU_MIN: dict[TaskPrimitive, float] = {
    TaskPrimitive.CLASSIFY: 0.050,
    TaskPrimitive.EXTRACT: 0.030,
    TaskPrimitive.REWRITE: 0.025,
    TaskPrimitive.SUMMARISE: 0.025,
    TaskPrimitive.ANSWER: 0.030,
    TaskPrimitive.ROUTE: 0.020,
    TaskPrimitive.GENERATE: 0.015,
    TaskPrimitive.TOOLCALL: 0.012,
}


def seed_floor(primitive: TaskPrimitive, eta: float = ETA, mu_min: float | None = None) -> int:
    """Derive the minimum real-seed count: ``n_min >= -ln(eta) / mu_min``.

    **The coverage argument.** Let a behavioural mode — an edge case, a rare
    field type, an unusual layout — have true probability mass ``mu``. The
    probability it appears nowhere in ``n`` i.i.d. seeds is ``(1-mu)^n ~ e^{-mu n}``.
    Requiring that miss probability to be at most ``eta`` for every mode of mass
    at least ``mu_min`` gives the bound above.

    Synthetic amplification cannot recover a missing mode. A teacher generating
    from seeds that omit a mode produces data that also omits it — amplification
    changes *presentation*, never *content*. This is the Curse of Recursion
    stated as an information bound: the synthetic corpus inherits the support of
    the seed set, and no volume of generation extends that support.

    **The assumption is explicit and falsifiable.** It presumes i.i.d. seeds.
    Customer documents usually are not — they arrive in batches with temporal and
    source correlation — so effective sample size is below the raw count and
    **these floors are optimistic**.
    """
    mu = mu_min if mu_min is not None else MU_MIN[primitive]
    if not 0.0 < mu < 1.0:
        raise ValueError("mu_min must lie in (0, 1)")
    if not 0.0 < eta < 1.0:
        raise ValueError("eta must lie in (0, 1)")
    return math.ceil(-math.log(eta) / mu)


def _p_seed(spec: SpecIR, plan: Any, catalog: Catalog) -> PredicateResult:
    floor = seed_floor(spec.task_primitive)
    have = spec.seed_data_count
    if have >= floor:
        return PredicateResult(True, "P_seed", detail={"floor": floor, "have": have})
    mu = MU_MIN[spec.task_primitive]
    miss = math.exp(-mu * have)
    return PredicateResult(
        False, "P_seed",
        reason=(
            f"{have} real seeds is below the derived floor of {floor} for "
            f"{spec.task_primitive.value!r}: a behavioural mode of mass {mu:.1%} would be "
            f"missed with probability {miss:.0%}, and amplification cannot recover it"
        ),
        detail={"floor": floor, "have": have, "mu_min": mu, "miss_probability": round(miss, 4)},
        remedy=f"supply at least {floor - have} more real examples covering the rare cases",
    )


P_seed = _P("P_seed", cost=1.0, pass_rate=0.8, soundness=Soundness.SOFT, fn=_p_seed)


# =========================================================================== #
# P_lic — licence composition (§3.5, §13)
# =========================================================================== #
def _p_lic(spec: SpecIR, plan: Any, catalog: Catalog) -> PredicateResult:
    base = catalog.model(plan.base_ref)
    teacher = catalog.model(plan.teacher_ref) if plan.teacher_ref else None
    audit = bool(spec.policy_rules and any("audit" in str(r).lower() for r in spec.policy_rules))

    outcome = resolve_licence(
        base_licence=base.licence,
        teacher_licence=teacher.licence if teacher else None,
        data_rights=spec.data_rights,
        audit_tier=audit,
    )
    if outcome.permitted:
        return PredicateResult(True, "P_lic", detail=outcome.as_record())

    fatal = outcome.position.fatal
    remedy = (
        "no downstream choice can relax an absorbing restriction — change the base, "
        "the teacher, or obtain training rights for the data"
        if fatal else
        "select a more permissive base or teacher, or deploy under terms that "
        "can carry these obligations"
    )
    return PredicateResult(
        False, "P_lic", reason=outcome.reason,
        detail=outcome.as_record(), remedy=remedy,
    )


P_lic = _P("P_lic", cost=2.0, pass_rate=0.9, soundness=Soundness.HARD, fn=_p_lic)


# =========================================================================== #
# P_ram — memory feasibility (§3.1, corrected by §14)
# =========================================================================== #
@dataclass(frozen=True)
class MemoryBreakdown:
    """``M = M_w + M_kv + M_emb + M_rt``, exactly, in bytes."""

    weights: int
    kv_cache: int
    embedder: int
    runtime: int
    app_reserve: int

    @property
    def model_total(self) -> int:
        return self.weights + self.kv_cache + self.embedder + self.runtime

    @property
    def total(self) -> int:
        return self.model_total + self.app_reserve

    def as_mb(self) -> dict[str, float]:
        return {
            "weights": round(self.weights / MB, 1),
            "kv_cache": round(self.kv_cache / MB, 1),
            "embedder": round(self.embedder / MB, 1),
            "runtime": round(self.runtime / MB, 1),
            "app_reserve": round(self.app_reserve / MB, 1),
            "total": round(self.total / MB, 1),
        }


def weight_bytes(model: ModelSpec, catalog: Catalog, quantiser: str) -> int:
    """``M_w = N_b * w_eff / 8 * (1 + epsilon)``.

    ``w_eff`` is the **effective** bit width, not the nominal one (§14). For MoE
    the full parameter count is used, never the active count: sparse activation
    is not sparse residency (A-07).
    """
    q = catalog.quantiser(quantiser)
    bits = catalog.bits_effective(model, quantiser)
    return int(model.params * bits / 8 * (1.0 + q.group_overhead))


def kv_bytes(model: ModelSpec, context: int, batch: int = 1, elem_bytes: int = 2) -> int:
    """``M_kv = 2 * L * H_kv * d_h * C * B * p_kv``."""
    if context <= 0 or batch <= 0:
        raise ValueError("context and batch must be positive")
    return model.kv_bytes_per_token(elem_bytes) * context * batch


def memory(
    model: ModelSpec, catalog: Catalog, *, quantiser: str, context: int,
    device: DeviceSpec, batch: int = 1,
) -> MemoryBreakdown:
    """The exact decomposition. No 2x heuristic anywhere in this path."""
    return MemoryBreakdown(
        weights=weight_bytes(model, catalog, quantiser),
        kv_cache=kv_bytes(model, context, batch),
        embedder=EMBEDDER_BYTES,
        runtime=RUNTIME_BYTES,
        app_reserve=device.app_reserve,
    )


def _p_ram(spec: SpecIR, plan: Any, catalog: Catalog) -> PredicateResult:
    model = catalog.model(plan.base_ref)
    device = catalog.device(spec.device_target)
    context = _context_of(spec, model)
    m = memory(model, catalog, quantiser=plan.quantiser, context=context, device=device)

    # A probe supersedes the prior, but the two measure different things and must
    # not be mixed:
    #   prior  — free_ram is the whole device, and app_reserve (M_app) is a
    #            MODELLED term that has to be added to the model's own footprint.
    #   probe  — ram_free_mb is FREE ram, already net of the OS and other apps,
    #            so adding M_app again would double-count it. The headroom factor
    #            covers the remaining §13 gap: the probe ran idle, production
    #            does not.
    profile = _probe_profile(spec)
    if profile is not None and profile.measured:
        budget = profile.usable_ram_mb * MB
        needed = m.model_total
        ram_source = f"{profile.source.value} (x{profile.headroom_factor:g} headroom)"
    else:
        budget = device.free_ram
        needed = m.total
        ram_source = "prior"

    if needed <= budget:
        return PredicateResult(
            True, "P_ram",
            detail={**m.as_mb(), "free_ram_mb": round(budget / MB, 1),
                    "needed_mb": round(needed / MB, 1),
                    "headroom_mb": round((budget - needed) / MB, 1),
                    "ram_source": ram_source,
                    "context_budget": context_budget(
                        model, catalog, quantiser=plan.quantiser, device=device,
                        free_ram_override=budget,
                        include_app_reserve=ram_source == "prior",
                    )},
        )

    over = needed - budget
    return PredicateResult(
        False, "P_ram",
        reason=(
            f"{model.ref} at {plan.quantiser} needs {needed / MB:.0f} MB "
            f"(weights {m.weights / MB:.0f} + KV {m.kv_cache / MB:.0f} at {context} ctx "
            f"+ runtime {(m.embedder + m.runtime) / MB:.0f}"
            + (f" + app {m.app_reserve / MB:.0f}" if ram_source == "prior" else "")
            + f") but {device.name} has {budget / MB:.0f} MB free "
            f"[{ram_source}] — over by {over / MB:.0f} MB"
        ),
        detail={**m.as_mb(), "over_mb": round(over / MB, 1), "context": context,
                "needed_mb": round(needed / MB, 1), "ram_source": ram_source,
                "free_ram_mb": round(budget / MB, 1)},
        remedy=(
            f"reduce context below {context}, or drop to the next smaller base"
            if m.kv_cache > m.weights // 3 else "drop to the next smaller base"
        ),
    )


def _context_of(spec: SpecIR, model: ModelSpec) -> int:
    """Planned context length, capped by what the model actually supports."""
    want = 2048
    if isinstance(spec.io_schema, dict):
        want = int(spec.io_schema.get("context_length", want))
    return max(1, min(want, model.max_context))


P_ram = _P("P_ram", cost=3.0, pass_rate=0.4, soundness=Soundness.HARD, fn=_p_ram)


# =========================================================================== #
# P_off — offline closure (§3.6)
# =========================================================================== #
def _p_off(spec: SpecIR, plan: Any, catalog: Catalog) -> PredicateResult:
    if not spec.offline_required:
        return PredicateResult(True, "P_off")

    offenders: list[str] = []
    if plan.target in {"vllm"}:
        offenders.append(f"target {plan.target!r} is a server runtime")
    if plan.data_recipe.get("requires_network"):
        offenders.append("data recipe requires network access at build time")
    if plan.data_recipe.get("remote_teacher_at_inference"):
        offenders.append("plan keeps a remote teacher on the inference path")
    if spec.abstention_policy.value == "escalate":
        offenders.append("abstention policy escalates to the cloud")

    # Data residency forces even TRAINING-time teacher generation to be local,
    # which can eliminate the strongest teachers entirely (§3.6).
    residency = isinstance(spec.io_schema, dict) and not spec.io_schema.get(
        "may_leave_jurisdiction", True
    )
    if residency and plan.teacher_ref and not plan.data_recipe.get("local_teacher"):
        offenders.append(
            f"data may not leave the jurisdiction, so teacher {plan.teacher_ref} "
            "must run locally at generation time"
        )

    if not offenders:
        return PredicateResult(True, "P_off")
    return PredicateResult(
        False, "P_off",
        reason="offline closure broken: " + "; ".join(offenders),
        detail={"offenders": offenders},
        remedy="split into an offline core and an online tail, or relax offline_required",
    )


P_off = _P("P_off", cost=2.0, pass_rate=0.95, soundness=Soundness.HARD, fn=_p_off)


# =========================================================================== #
# P_lat — latency feasibility (§3.2)
# =========================================================================== #
@dataclass(frozen=True)
class LatencyBreakdown:
    """Prefill and decode are different regimes and must not be conflated."""

    prefill_ms: float
    decode_ms: float
    tokens_in: int
    tokens_out: int
    source: str

    @property
    def total_ms(self) -> float:
        return self.prefill_ms + self.decode_ms

    @property
    def prefill_share(self) -> float:
        return self.prefill_ms / self.total_ms if self.total_ms else 0.0


def latency(
    model: ModelSpec, catalog: Catalog, *, quantiser: str, device: DeviceSpec,
    tokens_in: int, tokens_out: int,
) -> LatencyBreakdown:
    """Estimate generation latency.

    **Prefill is compute-bound.** Processing ``C_in`` tokens costs about
    ``2 * N_b * C_in`` FLOPs — two per parameter per token, multiply and
    accumulate.

    **Decode is memory-bandwidth-bound.** Each generated token reads every weight
    exactly once, so ``t_tok ~ M_w / BW_eff``.

    *Why decode is bandwidth-bound.* Arithmetic intensity is FLOPs per byte
    moved. Decoding one token performs ``~2 N_b`` FLOPs while moving ``~M_w``
    bytes, an intensity of about ``16 / w`` — roughly **4 FLOPs per byte at
    4-bit**. Modern processors balance at 50-200. Decode sits an order of
    magnitude below the ridge point, so no amount of extra compute helps; only
    smaller weights or faster memory do. This is why **quantisation buys latency
    roughly linearly** — it shrinks the numerator of the bandwidth-bound term.
    """
    m_w = weight_bytes(model, catalog, quantiser)
    if device.reference_params:
        # Rescale the device's quoted rates to THIS model by the roofline. The
        # reference weight size must be taken at the quantisation the rate was
        # MEASURED under, not at the candidate's — otherwise the bandwidth ratio
        # cancels and quantisation appears to buy no latency, which is backwards.
        ref_bits = catalog.quantiser(device.reference_quantiser).bits_effective
        ref_bytes = int(device.reference_params * ref_bits / 8)
        rates = device.rates_for(model.params, m_w, ref_bytes)
        if rates is not None:
            prefill_rate, decode_rate = rates
            return LatencyBreakdown(
                prefill_ms=1000.0 * tokens_in / prefill_rate,
                decode_ms=1000.0 * tokens_out / decode_rate,
                tokens_in=tokens_in, tokens_out=tokens_out, source="scaled_rates",
            )
    if device.flops_eff and device.bandwidth_eff:
        return LatencyBreakdown(
            prefill_ms=1000.0 * (2.0 * model.params * tokens_in) / device.flops_eff,
            decode_ms=1000.0 * tokens_out * m_w / device.bandwidth_eff,
            tokens_in=tokens_in, tokens_out=tokens_out, source="analytic",
        )
    return LatencyBreakdown(
        prefill_ms=math.inf, decode_ms=math.inf,
        tokens_in=tokens_in, tokens_out=tokens_out, source="unknown",
    )


def _probe_profile(spec: SpecIR):
    """The measured profile attached to the spec, if a probe has been run."""
    if not spec.device_profile:
        return None
    from modelrig.probe import DeviceProfile

    try:
        return DeviceProfile.from_dict(spec.device_profile)
    except Exception:  # noqa: BLE001 - a malformed profile is simply absent
        return None


def _p_lat(spec: SpecIR, plan: Any, catalog: Catalog) -> PredicateResult:
    if spec.latency_budget_ms is None:
        return PredicateResult(True, "P_lat", detail={"note": "no latency budget declared"})

    model = catalog.model(plan.base_ref)
    tokens_in, tokens_out = _token_shape(spec)

    # --- Part 3 §9: a probe moves this predicate from Tier 1 to Tier 3 ----- #
    # With measured hardware facts P_lat may PROMISE, not merely refuse. This is
    # what resolves the devices.yaml deadlock without buying twenty phones: the
    # customer's own device becomes the measurement instrument.
    profile = _probe_profile(spec)
    if profile is not None and profile.measured:
        from modelrig.probe import assess_latency

        m_w = weight_bytes(model, catalog, plan.quantiser)
        prefill_s = _prefill_seconds(model, catalog, spec, tokens_in)
        verdict = assess_latency(
            profile, m_w, tokens_out, prefill_s,
            params=model.params, tokens_in=tokens_in,
        )
        detail = {
            **verdict.as_dict(), "budget_ms": float(spec.latency_budget_ms),
            "profile_source": profile.source.value,
            "thermal_derate": profile.thermal_derate_180s,
            "measured": verdict.may_promise,
        }
        total_ms = verdict.seconds_total * 1000.0

        # A decode-only measurement cannot promise a prefill-dominated workload.
        if not verdict.may_promise:
            return PredicateResult(
                False, "P_lat",
                reason=f"cannot promise {total_ms / 1000:.1f} s: {verdict.reason}",
                detail=detail,
                remedy=(
                    "time a prefill of known length on the device — prefill is "
                    "compute-bound and needs its own measurement (§13.4)"
                ),
            )
        if total_ms <= float(spec.latency_budget_ms):
            return PredicateResult(True, "P_lat", detail=detail)

        dominant = "prefill" if verdict.prefill_dominates else "decode"
        return PredicateResult(
            False, "P_lat",
            reason=(
                f"{model.ref} measured on {profile.device_id}: "
                f"{verdict.seconds_total:.1f} s "
                f"({verdict.prefill_s:.1f} s prefill over {tokens_in} tokens + "
                f"{verdict.decode_s:.1f} s decode over {tokens_out}) against a "
                f"{float(spec.latency_budget_ms) / 1000:.1f} s budget — "
                f"{dominant} dominates, after a "
                f"{profile.thermal_derate_180s:.2f} thermal derate"
            ),
            detail=detail,
            remedy=(
                # §13.4's derived lever: when prefill dominates, shortening the
                # input reduces it linearly and costs no quality on the task.
                f"shorten the input below {tokens_in} tokens — prefill is "
                f"{verdict.prefill_share:.0%} of this workload and scales with it, "
                "so cropping or pre-filtering beats dropping a base tier"
                if verdict.prefill_dominates else
                "relax the budget or move down the size ladder — this figure is "
                "measured, so it will not improve on its own"
            ),
        )

    device = catalog.device(spec.device_target)
    accepts_estimates = bool(
        isinstance(spec.io_schema, dict) and spec.io_schema.get("accept_unmeasured_latency")
    )

    # The refusal that matters most (§3.2). Effective FLOPs and bandwidth are not
    # spec-sheet numbers: real efficiency runs at 30-60% of peak and varies with
    # thermal state, kernel quality and quantisation format. A planner that
    # interpolates them is fabricating a promise.
    if not device.measured and not accepts_estimates:
        return PredicateResult(
            False, "P_lat",
            reason=(
                f"latency_source for {device.name} is {device.latency_source!r}: a latency "
                "commitment cannot be made from unmeasured hardware. Effective throughput "
                "runs at 30-60% of peak and varies with thermal state and kernel quality."
            ),
            detail={"latency_source": device.latency_source, "measured": False},
            remedy=(
                "run the device probe on the target — two short benchmarks calibrate "
                "latency for the whole catalogue — or set "
                "io_schema.accept_unmeasured_latency=true to accept an estimate explicitly"
            ),
        )

    est = latency(model, catalog, quantiser=plan.quantiser, device=device,
                  tokens_in=tokens_in, tokens_out=tokens_out)

    if est.total_ms == math.inf:
        return PredicateResult(
            False, "P_lat",
            reason=f"{device.name} carries no throughput data, measured or analytic",
            remedy="add prefill_tok_s and decode_tok_s, or flops_eff and bandwidth_eff",
        )

    budget = float(spec.latency_budget_ms)
    detail = {
        "prefill_ms": round(est.prefill_ms, 1), "decode_ms": round(est.decode_ms, 1),
        "total_ms": round(est.total_ms, 1), "budget_ms": budget,
        "prefill_share": round(est.prefill_share, 3), "source": est.source,
        "measured": device.measured,
    }
    if est.total_ms <= budget:
        return PredicateResult(True, "P_lat", detail=detail)

    dominant = "prefill" if est.prefill_share > 0.5 else "decode"
    remedy = (
        f"reduce input length below {tokens_in} tokens — prefill is "
        f"{est.prefill_share:.0%} of the total and scales with it"
        if dominant == "prefill" else
        "use a smaller base or a narrower bit width; decode is bandwidth-bound "
        "so quantisation buys latency roughly linearly"
    )
    return PredicateResult(
        False, "P_lat",
        reason=(
            f"{model.ref} on {device.name}: estimated {est.total_ms / 1000:.1f} s "
            f"({est.prefill_ms / 1000:.1f} s prefill over {tokens_in} tokens + "
            f"{est.decode_ms / 1000:.1f} s decode over {tokens_out}) "
            f"against a {budget / 1000:.1f} s budget — {dominant} dominates"
        ),
        detail=detail, remedy=remedy,
    )


def _token_shape(spec: SpecIR) -> tuple[int, int]:
    """The workload shape. First-class fields win over the io_schema fallback.

    Part 4 §17: ``expected_input_tokens`` is a first-order latency driver, so it
    is a slot in its own right rather than a key buried in ``io_schema``.
    """
    io = spec.io_schema if isinstance(spec.io_schema, dict) else {}
    tokens_in = spec.expected_input_tokens or int(io.get("tokens_in", 512))
    tokens_out = spec.expected_output_tokens or int(io.get("tokens_out", 64))
    return tokens_in, tokens_out


def _prefill_seconds(model: ModelSpec, catalog: Catalog, spec: SpecIR, tokens_in: int) -> float:
    """Prefill time, taken from the device prior.

    A probe calibrates DECODE, which is bandwidth-bound and therefore the term the
    two-point model captures. Prefill is compute-bound and needs its own
    measurement, so until the probe sweeps it too this falls back to the prior's
    prefill rate — and the mixed provenance is recorded rather than hidden.
    """
    try:
        device = catalog.device(spec.device_target)
    except Exception:  # noqa: BLE001
        return 0.0
    est = latency(model, catalog, quantiser="q4_k_m", device=device,
                  tokens_in=tokens_in, tokens_out=0)
    return 0.0 if est.prefill_ms == math.inf else est.prefill_ms / 1000.0


def context_budget(
    model: ModelSpec, catalog: Catalog, *, quantiser: str, device: DeviceSpec,
    free_ram_override: int | None = None, batch: int = 1,
    include_app_reserve: bool = True,
) -> int:
    """Derive the maximum context the device can hold (§10).

    The coordinate most often missed. KV cache is linear in context length, so on
    a tight device the plan does not merely pick a smaller model — it **caps the
    context**, which caps how many retrieved chunks the cartridge may use, which
    feeds back into the retrieval design.

    **Device constraints propagate upward into task design**, not only downward
    into model size. This is derived, never elicited.
    """
    free = free_ram_override if free_ram_override is not None else device.free_ram
    # A measured free-RAM figure is already net of the application, so M_app must
    # not be subtracted a second time.
    reserve = device.app_reserve if include_app_reserve else 0
    spare = free - reserve - weight_bytes(model, catalog, quantiser) \
        - EMBEDDER_BYTES - RUNTIME_BYTES
    if spare <= 0:
        return 0
    per_token = model.kv_bytes_per_token() * batch
    return min(int(spare // per_token), model.max_context)


P_lat = _P("P_lat", cost=4.0, pass_rate=0.6, soundness=Soundness.SOFT, fn=_p_lat)


# =========================================================================== #
# P_cost — budget (§3.7)
# =========================================================================== #
def _p_cost(spec: SpecIR, plan: Any, catalog: Catalog) -> PredicateResult:
    from modelrig.planner import costmodel

    model = catalog.model(plan.base_ref)
    recipe = plan.data_recipe or {}
    n_syn = int(recipe.get("n_synthetic", 20_000))
    n_tok = int(recipe.get("n_train_tokens", n_syn * 320))
    n_eval = int(recipe.get("n_eval_examples", 50))

    breakdown = costmodel.estimate(
        model, method=plan.peft_method, n_synthetic=n_syn,
        n_train_tokens=n_tok, n_eval_examples=n_eval,
        uses_teacher=plan.teacher_ref is not None,
    )
    ceiling = int(round(spec.budget_ceiling_usd * USD))   # integer micro-USD
    detail = {
        "generation_usd": usd(breakdown.generation),
        "training_usd": usd(breakdown.training),
        "evaluation_usd": usd(breakdown.evaluation),
        "total_usd": usd(breakdown.total),
        "ceiling_usd": usd(ceiling),
        "dominant": breakdown.dominant,
    }
    if breakdown.total <= ceiling:
        return PredicateResult(True, "P_cost", detail=detail)
    return PredicateResult(
        False, "P_cost",
        reason=(
            f"estimated build cost ${usd(breakdown.total):.2f} exceeds the tier ceiling "
            f"${usd(ceiling):.2f}; {breakdown.dominant} dominates"
        ),
        detail=detail,
        remedy=(
            "reduce synthetic volume" if breakdown.dominant == "teacher generation"
            else "use QLoRA or a smaller base" if breakdown.dominant == "training"
            else "shrink the evaluation set"
        ),
    )


P_cost = _P("P_cost", cost=5.0, pass_rate=0.85, soundness=Soundness.SOFT, fn=_p_cost)


# =========================================================================== #
# P_storage — Part 4 §13.2
# =========================================================================== #
def _p_storage(spec: SpecIR, plan: Any, catalog: Catalog) -> PredicateResult:
    """Storage binds independently of memory, in **both** directions.

    A 4 GB-RAM phone with 8 GB free storage can hold an artefact it cannot load;
    a device with ample RAM and a full disk cannot install one. Two predicates,
    neither implying the other.
    """
    from modelrig import resources

    profile = _probe_profile(spec)
    free = (profile.storage_free_mb * MB) if profile and profile.storage_free_mb else 0
    if not free:
        return PredicateResult(
            True, "P_storage",
            detail={"note": "no measured free storage; probe it to enable this check"},
        )

    model = catalog.model(plan.base_ref)
    io = spec.io_schema if isinstance(spec.io_schema, dict) else {}
    s = resources.storage(
        artefact=weight_bytes(model, catalog, plan.quantiser),
        n_chunks=int(io.get("index_chunks", 0)),
        index_dim=int(io.get("index_dim", 768)),
        index_quantised=bool(io.get("index_quantised", False)),
    )
    detail = {
        "artefact_mb": round(s.artefact / MB, 1), "index_mb": round(s.index / MB, 1),
        "total_mb": round(s.total / MB, 1),
        "usable_mb": round(free * resources.STORAGE_SAFETY / MB, 1),
        "dominant": s.dominant,
    }
    if resources.storage_fits(s, free):
        return PredicateResult(True, "P_storage", detail=detail)
    return PredicateResult(
        False, "P_storage",
        reason=(
            f"artefact plus index needs {s.total / MB:.0f} MB but only "
            f"{free * resources.STORAGE_SAFETY / MB:.0f} MB is usable "
            f"({s.dominant} dominates; phones behave badly near full storage)"
        ),
        detail=detail,
        remedy=(
            "quantise the retrieval index to int8 — a 4x reduction for negligible "
            "retrieval loss, and usually the cheapest win available"
            if s.dominant == "index" else "drop to a smaller base"
        ),
    )


P_storage = _P("P_storage", cost=2.0, pass_rate=0.9, soundness=Soundness.HARD,
               fn=_p_storage)


# =========================================================================== #
# P_energy — Part 4 §13.6
# =========================================================================== #
def _p_energy(spec: SpecIR, plan: Any, catalog: Catalog) -> PredicateResult:
    """A day's volume must fit inside one charge, with margin.

    Only checked when the spec declares a daily volume — a mains-powered target
    has no battery to run down.
    """
    from modelrig import resources
    from modelrig.probe import assess_latency

    volume = spec.expected_daily_volume
    profile = _probe_profile(spec)
    if not volume or profile is None or not profile.measured:
        return PredicateResult(
            True, "P_energy",
            detail={"note": "no declared daily volume or no measured device"},
        )

    model = catalog.model(plan.base_ref)
    tokens_in, tokens_out = _token_shape(spec)
    m_w = weight_bytes(model, catalog, plan.quantiser)
    verdict = assess_latency(profile, m_w, tokens_out, 0.0,
                             params=model.params, tokens_in=tokens_in)
    e = resources.energy(
        verdict.seconds_total,
        power_draw_w=profile.power_draw_w or resources.DEFAULT_POWER_DRAW_W,
    )
    detail = {**e.as_dict(), "daily_volume": volume,
              "battery_share": round(e.battery_share(volume), 3),
              "power_measured": profile.power_draw_w is not None}
    if resources.energy_fits(e, volume):
        return PredicateResult(True, "P_energy", detail=detail)
    return PredicateResult(
        False, "P_energy",
        reason=(
            f"{e.requests_per_charge:.0f} requests per charge against a "
            f"{volume}/day volume — {e.battery_share(volume):.0%} of a full battery, "
            "before any other app runs"
        ),
        detail=detail,
        remedy="shorten the input, use a smaller base, or plan for mid-day charging",
    )


P_energy = _P("P_energy", cost=4.0, pass_rate=0.95, soundness=Soundness.SOFT,
              fn=_p_energy)


# =========================================================================== #
# Ordering (§4)
# =========================================================================== #
ALL_PREDICATES: tuple[_P, ...] = (
    P_tok, P_seed, P_lic, P_ram, P_off, P_lat, P_cost, P_storage, P_energy,
)

#: §16's partition. Report these separately; never blend them into one number.
HARD: tuple[str, ...] = tuple(p.name for p in ALL_PREDICATES if p.soundness is Soundness.HARD)
SOFT: tuple[str, ...] = tuple(p.name for p in ALL_PREDICATES if p.soundness is Soundness.SOFT)


def ordered_predicates(predicates: tuple[_P, ...] = ALL_PREDICATES) -> list[_P]:
    """Sort ascending on ``c_i / (1 - rho_i)`` — cheapest-and-most-discriminating first.

    Predicates form a conjunction, so order cannot affect correctness; it affects
    only cost. Expected cost under ordering ``sigma`` is
    ``sum_i c_sigma(i) * prod_{j<i} rho_sigma(j)``, minimised by this key. It is
    the classical sequential-testing result — the same rule a query optimiser
    uses to order selective predicates.
    """
    return sorted(predicates, key=lambda p: (p.selectivity, p.name))


def expected_cost(order: list[_P]) -> float:
    """``E[cost] = sum_i c_i * prod_{j<i} rho_j`` for a given ordering."""
    total, reach = 0.0, 1.0
    for p in order:
        total += p.cost * reach
        reach *= p.pass_rate
    return total
