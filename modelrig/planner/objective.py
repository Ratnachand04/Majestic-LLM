"""The objective and the refusal threshold (§5, §6).

Two separate things live here.

**Ranking** (§5). Once the feasible set is known, order it by

    U(pi, s) = Q(pi,s) - lambda_c C/C_max - lambda_t tau/tau_max - lambda_s M_w/M_max

Quality dominates; cost and time discount it; the size term breaks ties toward
smaller artefacts — cheaper to store, faster to ship, less prone to quantisation
damage.

**Deciding whether to build at all** (§6). This is the theoretical core, and it
has no analogue in AutoML.

The seven hard predicates are not probabilistic: if a model does not fit in RAM
it does not fit, and the refusal is certain. The interesting case is soft — the
plan is feasible, but predicted quality is low. Should you build it?

With ``V`` the value of a working model, ``C`` the build cost and ``kappa`` the
trust damage of shipping a failure:

    E[build] = p(V - C) + (1-p)(-C - kappa) = p(V + kappa) - (C + kappa)

Build when that beats refusing (zero), giving

    theta* = (C + kappa) / (V + kappa)

Three consequences are worth drawing out, and the third is the one that matters.

* **kappa dominates as it grows.** As trust damage rises, ``theta* -> 1``: a
  system whose failures are catastrophic to trust should refuse almost anything
  it is unsure about. In regulated domains the threshold sits near 1.
* **Cheap builds justify optimism, but only so far.** As ``C -> 0`` with kappa
  fixed, ``theta* -> kappa/(V+kappa)``, still bounded away from zero. **Even free
  builds should sometimes be refused**, because the damage of shipping a bad
  model is not the compute you burned. This is why "just build it and see" is
  wrong even where marginal cost is zero.
* **It all rests on calibration.** If ``p`` is badly calibrated near
  ``theta*``, the optimal policy collapses to "build everything" and the
  flagship contribution evaporates. That is claim C2, and it is an empirical
  test of this entire module rather than a nice-to-have experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from modelrig.planner.costmodel import USD, usd


class Tier(str, Enum):
    """Crude customer tiering for ``V`` and ``kappa`` (§11 open question 2).

    Estimating value and trust damage per customer is unvalidated. A three-way
    tiering is probably enough, and being explicit about the guess is better
    than hiding it inside a constant.
    """

    EXPERIMENTAL = "experimental"
    COMMERCIAL = "commercial"
    REGULATED = "regulated"


@dataclass(frozen=True)
class DecisionParams:
    """``V``, ``C`` and ``kappa``, in integer micro-USD."""

    value: int          # V — value of a working model to the customer
    build_cost: int     # C — cost of attempting the build
    trust_damage: int   # kappa — cost of delivering a failure

    def threshold(self) -> float:
        """``theta* = (C + kappa) / (V + kappa)``."""
        denom = self.value + self.trust_damage
        if denom <= 0:
            return 1.0                       # no value to gain: refuse everything
        return min(1.0, (self.build_cost + self.trust_damage) / denom)

    def expected_value(self, p: float) -> int:
        """``E[build] = p(V + kappa) - (C + kappa)``, in micro-USD."""
        return int(p * (self.value + self.trust_damage) - (self.build_cost + self.trust_damage))

    def as_usd(self) -> dict[str, float]:
        return {"V": usd(self.value), "C": usd(self.build_cost), "kappa": usd(self.trust_damage)}


#: Value and trust-damage priors per tier, as ``(V, kappa)`` in micro-USD.
# HYPOTHESIS — these are the (V, kappa) estimates §11 flags as unvalidated.
# SOURCE for the commercial row: §6's worked figures, V=$300, kappa=$200,
# which reproduce theta* = 240/500 = 0.48 exactly.
#
# Note that kappa alone does not order the tiers: theta* = (C+kappa)/(V+kappa)
# rises when V falls too, so a tier that values the model very little becomes
# CONSERVATIVE rather than permissive, however tolerant of failure it is. An
# experimental tier is permissive only because its trust damage is negligible
# AND the model is still worth several builds to it.
_TIER_PRIORS: dict[Tier, tuple[int, int]] = {
    Tier.EXPERIMENTAL: (150 * USD, 5 * USD),      # theta* ~ 0.29 at C = $40
    Tier.COMMERCIAL: (300 * USD, 200 * USD),      # theta* = 0.48 at C = $40
    Tier.REGULATED: (500 * USD, 5_000 * USD),     # theta* ~ 0.92 at C = $40
}


def decision_params(build_cost: int, tier: Tier = Tier.COMMERCIAL) -> DecisionParams:
    """Assemble ``(V, C, kappa)`` for a tier."""
    value, damage = _TIER_PRIORS[tier]
    return DecisionParams(value=value, build_cost=build_cost, trust_damage=damage)


def refusal_threshold(build_cost: int, tier: Tier = Tier.COMMERCIAL) -> float:
    """``theta*`` for this build. Refuse when predicted pass probability is below it."""
    return decision_params(build_cost, tier).threshold()


def should_build(p: float, build_cost: int, tier: Tier = Tier.COMMERCIAL) -> bool:
    """The §6 decision rule."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("pass probability must lie in [0, 1]")
    return p >= refusal_threshold(build_cost, tier)


# --------------------------------------------------------------------------- #
#: Utility weights. Quality dominates by construction.
LAMBDA_COST = 0.15
LAMBDA_TIME = 0.10
LAMBDA_SIZE = 0.05


@dataclass(frozen=True)
class UtilityTerms:
    """The four terms of ``U``, kept separate so a ranking can be explained."""

    quality: float
    cost_penalty: float
    time_penalty: float
    size_penalty: float

    @property
    def total(self) -> float:
        return self.quality - self.cost_penalty - self.time_penalty - self.size_penalty


def utility(
    *,
    quality: float,
    cost_micro: int,
    cost_ceiling_micro: int,
    latency_ms: float,
    latency_budget_ms: float | None,
    weight_bytes: int,
    weight_ceiling_bytes: int,
) -> UtilityTerms:
    """Score one feasible plan.

    Every penalty is normalised into ``[0, 1]`` against its own budget so the
    lambdas are comparable and a change to one budget cannot silently dominate
    the ranking.
    """
    c_norm = _ratio(cost_micro, cost_ceiling_micro)
    t_norm = _ratio(latency_ms, latency_budget_ms) if latency_budget_ms else 0.0
    s_norm = _ratio(weight_bytes, weight_ceiling_bytes)
    return UtilityTerms(
        quality=quality,
        cost_penalty=LAMBDA_COST * c_norm,
        time_penalty=LAMBDA_TIME * t_norm,
        size_penalty=LAMBDA_SIZE * s_norm,
    )


def _ratio(value: float, ceiling: float | None) -> float:
    if not ceiling or ceiling <= 0:
        return 0.0
    return max(0.0, min(1.0, value / ceiling))


# --------------------------------------------------------------------------- #
def quality_prior(
    *, params_b: float, min_params_b: float, seed_count: int, seed_floor: int,
    distil_mode: str, bit_width_effective: float,
) -> float:
    """The rule-based ``Q_prior`` used before any build history exists (§5).

    Deliberately crude and monotone in the things that plainly matter: headroom
    over the primitive's minimum size, seed sufficiency, whether a teacher is
    involved, and how aggressively the artefact is compressed. It exists to be
    *replaced* by the learned predictor, and the shrinkage in
    :mod:`modelrig.planner.metalearn` governs how fast that happens.
    """
    size_headroom = min(params_b / max(min_params_b, 1e-6), 4.0) / 4.0
    seed_headroom = min(seed_count / max(seed_floor, 1), 4.0) / 4.0
    teacher_bonus = {"logit_kd": 0.10, "sequence_kd": 0.05, "none": 0.0}.get(distil_mode, 0.0)
    # Compression damage: 4-bit costs a little, 8-bit and above costs nothing.
    quant_penalty = 0.06 if bit_width_effective < 6.0 else 0.0

    raw = 0.45 + 0.28 * size_headroom + 0.22 * seed_headroom + teacher_bonus - quant_penalty
    return max(0.0, min(1.0, raw))
