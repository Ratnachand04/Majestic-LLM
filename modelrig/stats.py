"""The statistics every customer-facing number passes through (Part 7 §2-§7).

Write this before anything else in the Proving Ground. Majestic does not sell a
model file, it sells a certificate — so **if the statistics are wrong, the
product is a lie with good typography.**

**The correction this module exists for.** Every earlier document wrote the gate
as ``q_hat >= q_gate``. That is wrong: ``q_hat`` is an estimate with variance,
and the comparison ignores it. The gate is a hypothesis test,

    H0: q <= q_gate        H1: q > q_gate

and it passes only when ``H0`` can be *rejected*. The burden of proof sits on the
model, which is the right direction — a certificate asserts something, and an
assertion needs evidence.

**What that costs, concretely.** At ``n = 50`` the Wilson interval around
``q_hat = 0.94`` spans fourteen points. The observed score and a 0.93 gate are
statistically indistinguishable; "0.937 versus 0.93" is not a comparison, it is
noise. Exactly: at ``n = 50`` a **single error** makes a 0.93 claim
uncertifiable, and the Apex build every document in this series has treated as a
success cannot honestly claim its stated gate.

The honest routes out are to collect more data, to certify what ``n`` actually
supports, to narrow with a capped prior, or to ship provisionally and let field
data tighten the interval. Reporting the point estimate and hiding the interval
is not one of them.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Sequence

from majestic.logging_utils import get_logger

logger = get_logger(__name__)

_NORMAL = NormalDist()

#: Default one-sided confidence level for a certificate.
DEFAULT_ALPHA = 0.05

#: Default power target when sizing a held-out set.
DEFAULT_BETA = 0.20


def z(p: float) -> float:
    """Standard normal quantile."""
    if not 0.0 < p < 1.0:
        raise ValueError("quantile probability must lie in (0, 1)")
    return _NORMAL.inv_cdf(p)


# =========================================================================== #
# Intervals
# =========================================================================== #
@dataclass(frozen=True)
class Interval:
    """An estimate with its uncertainty, and the ``n`` that produced it."""

    estimate: float
    low: float
    high: float
    n: int
    method: str = "wilson"

    @property
    def width(self) -> float:
        return self.high - self.low

    def supports(self, threshold: float) -> bool:
        """Whether the *lower bound* clears the threshold. The gate rule (§5)."""
        return self.low >= threshold

    def describe(self, places: int = 3) -> str:
        return (f"{self.estimate:.{places}f} "
                f"[{self.low:.{places}f}, {self.high:.{places}f}] at n={self.n}")

    def plain(self) -> str:
        """§19 — a sentence a clinic owner can act on, not an F1 score."""
        return (f"{self.estimate:.0%} (between {self.low:.0%} and {self.high:.0%} "
                f"— based on {self.n} of your examples)")

    def as_dict(self) -> dict[str, Any]:
        return {
            "estimate": round(self.estimate, 4),
            "low": round(self.low, 4),
            "high": round(self.high, 4),
            "width": round(self.width, 4),
            "n": self.n,
            "method": self.method,
        }


def wilson(successes: int, n: int, alpha: float = DEFAULT_ALPHA) -> Interval:
    """Two-sided Wilson score interval. Well-behaved at small ``n`` and near 1.

    The normal approximation is not: at ``k = n`` it gives a zero-width interval
    at 1.0, which would certify anything on a clean sweep of five examples.
    """
    _check(successes, n)
    if n == 0:
        return Interval(0.0, 0.0, 1.0, 0, "wilson")
    zc = z(1.0 - alpha / 2.0)
    p = successes / n
    denom = 1.0 + zc * zc / n
    centre = (p + zc * zc / (2 * n)) / denom
    half = zc * math.sqrt(p * (1 - p) / n + zc * zc / (4 * n * n)) / denom
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half), n, "wilson")


def clopper_pearson_lcb(successes: int, n: int, alpha: float = DEFAULT_ALPHA) -> float:
    """Exact one-sided lower confidence bound on a proportion.

    The ``p`` for which ``P(X >= k | n, p) = alpha``. Exact rather than
    approximate, because this is the number a certificate rests on and an
    approximation that is optimistic at small ``n`` is exactly the wrong error.

    At ``n = 50`` a perfect sweep certifies 0.942 and a single error drops it to
    0.906 — which is why one mistake makes a 0.93 claim uncertifiable.
    """
    _check(successes, n)
    if n == 0 or successes == 0:
        return 0.0
    if successes == n:
        # P(X >= n) = p^n = alpha  =>  p = alpha^(1/n)
        return alpha ** (1.0 / n)

    lo, hi = 0.0, 1.0
    for _ in range(200):                       # bisection: ~1e-60 precision
        mid = (lo + hi) / 2.0
        if _binom_sf(successes, n, mid) > alpha:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def clopper_pearson(successes: int, n: int, alpha: float = DEFAULT_ALPHA) -> Interval:
    """The two-sided exact interval, reported as an :class:`Interval`."""
    _check(successes, n)
    low = clopper_pearson_lcb(successes, n, alpha / 2.0)
    high = 1.0 - clopper_pearson_lcb(n - successes, n, alpha / 2.0)
    estimate = successes / n if n else 0.0
    return Interval(estimate, low, high, n, "clopper-pearson")


def _binom_sf(k: int, n: int, p: float) -> float:
    """``P(X >= k)`` for ``X ~ Binomial(n, p)``, in log space for stability."""
    if p <= 0.0:
        return 1.0 if k <= 0 else 0.0
    if p >= 1.0:
        return 1.0 if k <= n else 0.0
    total = 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    for i in range(k, n + 1):
        log_term = (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                    + i * log_p + (n - i) * log_q)
        total += math.exp(log_term)
    return min(1.0, total)


def _check(successes: int, n: int) -> None:
    if n < 0 or successes < 0 or successes > n:
        raise ValueError(f"need 0 <= successes <= n, got {successes}/{n}")


# =========================================================================== #
# §5 — the gate rule
# =========================================================================== #
def certifies(successes: int, n: int, threshold: float,
              alpha: float = DEFAULT_ALPHA) -> bool:
    """``LCB_{1-alpha}(q_hat, n) >= q_gate``. Strictly more conservative, and
    strictly more honest, than comparing the point estimate."""
    return clopper_pearson_lcb(successes, n, alpha) >= threshold


def certifiable_threshold(successes: int, n: int, alpha: float = DEFAULT_ALPHA) -> float:
    """The highest gate this evidence *does* support.

    §5's honest escape: rather than refusing a build whose data cannot carry a
    0.93 claim, issue the 0.85 certificate the data does carry. The customer
    gets a true statement instead of no statement.
    """
    return clopper_pearson_lcb(successes, n, alpha)


# =========================================================================== #
# §4 — how much data a gate actually needs
# =========================================================================== #
def required_n(q0: float, q1: float, alpha: float = DEFAULT_ALPHA,
               beta: float = DEFAULT_BETA) -> int:
    """Sample size for a **one-sided** proportion test.

        n >= [ (z_a sqrt(q0(1-q0)) + z_b sqrt(q1(1-q1))) / (q1 - q0) ]^2

    .. note::

       §4's table quotes 380, 150, 81 and 52. Three of those reproduce exactly
       from this one-sided formula; the ``q1 = 0.98`` row does not — it gives
       **116**, and 150 only appears if that row alone used a two-sided
       ``z = 1.96``. The section header specifies a one-sided test, so the
       formula is taken as authoritative and that row as a slip.
    """
    if not 0.0 < q0 < 1.0 or not 0.0 < q1 < 1.0:
        raise ValueError("proportions must lie in (0, 1)")
    if q0 == q1:
        raise ValueError("no difference to detect")
    numerator = (z(1 - alpha) * math.sqrt(q0 * (1 - q0))
                 + z(1 - beta) * math.sqrt(q1 * (1 - q1)))
    return math.ceil((numerator / abs(q1 - q0)) ** 2)


def resolvable_difference(n: int, q0: float, alpha: float = DEFAULT_ALPHA,
                          beta: float = DEFAULT_BETA, step: float = 0.001) -> float:
    """The smallest improvement ``n`` can actually detect.

    **Fifty held-out examples resolves roughly ten-point differences.** It cannot
    resolve the three-point differences gate thresholds are usually written at,
    which is the whole problem: the schema's ``holdout_count >= 20`` minimum
    resolves nothing at all.
    """
    if n < 1:
        raise ValueError("sample size must be positive")

    def smallest(direction: int) -> float:
        delta = step
        while 0.0 < q0 + direction * delta < 1.0:
            if required_n(q0, q0 + direction * delta, alpha, beta) <= n:
                return delta
            delta += step
        return float("inf")

    # Report the HARDER direction. Detecting a model far above the gate is easy
    # because the variance collapses near 1; detecting one below it is not, and
    # a claim about resolution should be the conservative one. At n=50 this is
    # about ten points, which is why a three-point gate cannot be resolved.
    return round(max(smallest(+1), smallest(-1)), 4)


@dataclass(frozen=True)
class PowerReport:
    """What a held-out set of this size can and cannot say."""

    n: int
    threshold: float
    resolves: float
    needed_for_3pt: int
    needed_for_5pt: int

    @property
    def adequate(self) -> bool:
        """Whether this held-out set can resolve a five-point difference.

        Gate thresholds are typically written at three points, so "adequate"
        here is already a generous reading.
        """
        return self.resolves <= 0.05

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "threshold": self.threshold,
            "resolvable_difference": self.resolves,
            "n_for_3_points": self.needed_for_3pt,
            "n_for_5_points": self.needed_for_5pt,
            "adequate": self.adequate,
        }


def power_report(n: int, threshold: float, alpha: float = DEFAULT_ALPHA,
                 beta: float = DEFAULT_BETA) -> PowerReport:
    return PowerReport(
        n=n, threshold=threshold,
        resolves=resolvable_difference(n, threshold, alpha, beta),
        needed_for_3pt=required_n(threshold, min(threshold + 0.03, 0.999), alpha, beta),
        needed_for_5pt=required_n(threshold, min(threshold + 0.05, 0.999), alpha, beta),
    )


# =========================================================================== #
# §6 — Bayesian narrowing, with the two guards
# =========================================================================== #
@dataclass(frozen=True)
class BayesResult:
    """A posterior credible interval, alongside what the data alone supports."""

    posterior_mean: float
    low: float
    high: float
    prior_strength: float
    prior_capped: bool
    frequentist_lcb: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "posterior_mean": round(self.posterior_mean, 4),
            "credible_low": round(self.low, 4),
            "credible_high": round(self.high, 4),
            "prior_strength": round(self.prior_strength, 2),
            "prior_capped": self.prior_capped,
            "frequentist_lcb": round(self.frequentist_lcb, 4),
            "note": (
                "the prior comes from similar past builds, so a genuinely broken "
                "build would inherit credit for its ancestors. Both numbers are "
                "shown: the customer is entitled to know which came from their data"
            ),
        }


def beta_posterior(successes: int, n: int, a0: float = 1.0, b0: float = 1.0,
                   alpha: float = DEFAULT_ALPHA, samples: int = 20_000,
                   seed: int = 0) -> BayesResult:
    """Narrow the interval with a prior from similar builds — carefully.

    Two guards, both mandatory:

    1. **Cap prior strength at ``n``.** The prior may never outvote the
       customer's own data. Without the cap, a strong prior from successful
       ancestors certifies a broken build.
    2. **Report both numbers.** The frequentist bound travels alongside, so the
       customer can see which claim rests on their evidence.
    """
    _check(successes, n)
    strength = a0 + b0
    capped = strength > n and n > 0
    if capped:
        scale = n / strength
        a0, b0 = a0 * scale, b0 * scale
        logger.info(
            "stats: prior strength %.1f capped to %d — a prior may not outvote "
            "the customer's own data", strength, n,
        )

    a, b = a0 + successes, b0 + (n - successes)
    rng = random.Random(seed)
    draws = sorted(rng.betavariate(a, b) for _ in range(samples))
    lo = draws[int(alpha / 2 * samples)]
    hi = draws[min(int((1 - alpha / 2) * samples), samples - 1)]
    return BayesResult(
        posterior_mean=a / (a + b), low=lo, high=hi,
        prior_strength=min(strength, float(n)) if n else strength,
        prior_capped=capped,
        frequentist_lcb=clopper_pearson_lcb(successes, n, alpha),
    )


# =========================================================================== #
# §7 — provisional certification
# =========================================================================== #
def effective_n(n0: int, daily_volume: int, correction_rate: float, days: float) -> int:
    """``n_eff(t) = n0 + r*V*t``. Field usage generates labelled outcomes.

    Corrections are ground truth, so the certificate **strengthens monotonically
    with deployment**. This does not manufacture confidence: it defers the strong
    claim until the evidence exists, and says when that will be.
    """
    if min(n0, daily_volume) < 0 or days < 0 or not 0.0 <= correction_rate <= 1.0:
        raise ValueError("counts must be non-negative and the rate in [0, 1]")
    return int(n0 + daily_volume * correction_rate * days)


def days_to_certify(n0: int, target_n: int, daily_volume: int,
                    correction_rate: float) -> float | None:
    """When a provisional certificate becomes a full one."""
    per_day = daily_volume * correction_rate
    if n0 >= target_n:
        return 0.0
    if per_day <= 0:
        return None
    return round((target_n - n0) / per_day, 1)


# =========================================================================== #
# §11 — judge attenuation
# =========================================================================== #
def judge_usable_n(n: int, swap_inconsistency: float, agreement: float) -> float:
    """``n_usable ~ n (1-f) rho^2`` — noise costs sample size twice over.

    The ``rho^2`` is classical attenuation: measurement error in the instrument
    reduces effective information quadratically. At ``n=50``, ``f=0.15`` and
    ``rho=0.80`` about half the data is gone to instrument noise, which is the
    quantitative reason the Judge is advisory rather than blocking.
    """
    if not 0.0 <= swap_inconsistency <= 1.0 or not 0.0 <= agreement <= 1.0:
        raise ValueError("rates must lie in [0, 1]")
    return n * (1.0 - swap_inconsistency) * agreement ** 2


# =========================================================================== #
# §12 — flip decomposition, and the boundary-mass mechanism
# =========================================================================== #
@dataclass(frozen=True)
class FlipDecomposition:
    """``phi = phi_+ + phi_-`` against ``Delta = phi_- - phi_+``.

    Two models can share an accuracy delta and differ wildly in churn:

        A: phi_+ = 0.01, phi_- = 0.03  ->  Delta = 0.02, phi = 0.04
        B: phi_+ = 0.15, phi_- = 0.17  ->  Delta = 0.02, phi = 0.32

    Identical net change; B has rewritten a third of its answers. ``Delta``
    measures only the net, ``phi`` measures the churn — and churn is what
    predicts field failure.
    """

    improved: int          # wrong -> right
    degraded: int          # right -> wrong
    unchanged: int
    n: int

    @property
    def phi_plus(self) -> float:
        return self.improved / self.n if self.n else 0.0

    @property
    def phi_minus(self) -> float:
        return self.degraded / self.n if self.n else 0.0

    @property
    def phi(self) -> float:
        """Total churn. An estimate of probability mass near the boundary."""
        return self.phi_plus + self.phi_minus

    @property
    def delta(self) -> float:
        """Net accuracy change. Contains no information about fragility."""
        return self.phi_minus - self.phi_plus

    @property
    def boundary_mass(self) -> float:
        """``phi ~ P(|margin| < epsilon)`` — a direct measure of fragility.

        If quantisation perturbs the decision function by ``epsilon``, the
        answers that flip are exactly those whose margin was smaller than the
        perturbation. On the held-out set those perturbations happen to cancel;
        on new data there is no reason they should.
        """
        return self.phi

    def as_dict(self) -> dict[str, Any]:
        return {
            "phi": round(self.phi, 4),
            "phi_plus": round(self.phi_plus, 4),
            "phi_minus": round(self.phi_minus, 4),
            "delta": round(self.delta, 4),
            "n": self.n,
            "why": (
                "delta measures the net, phi measures the churn. phi estimates the "
                "probability mass near the decision boundary, which is what "
                "predicts failure on data the eval set did not see"
            ),
        }


def decompose_flips(reference_correct: Sequence[bool],
                    candidate_correct: Sequence[bool]) -> FlipDecomposition:
    """Split the flips by direction (§12)."""
    if len(reference_correct) != len(candidate_correct):
        raise ValueError("paired comparison needs equal-length sequences")
    improved = sum(1 for r, c in zip(reference_correct, candidate_correct) if not r and c)
    degraded = sum(1 for r, c in zip(reference_correct, candidate_correct) if r and not c)
    n = len(reference_correct)
    return FlipDecomposition(improved, degraded, n - improved - degraded, n)


def bootstrap_variance(correct: Sequence[bool], resamples: int = 2000,
                       seed: int = 0) -> float:
    """Variance of accuracy across bootstrap resamples of the eval set.

    §12's second, cheaper experiment for C4. If ``phi`` really measures boundary
    mass, high-``phi`` models should show **higher bootstrap variance** — and
    that is testable with no additional models and no field data, which turns a
    bare predictive correlation into evidence for the mechanism.
    """
    n = len(correct)
    if n == 0:
        return 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        draw = [correct[rng.randrange(n)] for _ in range(n)]
        means.append(sum(draw) / n)
    mean = sum(means) / len(means)
    return sum((m - mean) ** 2 for m in means) / len(means)


# =========================================================================== #
# §20 — paired comparison
# =========================================================================== #
@dataclass(frozen=True)
class McNemarResult:
    """The paired test, and an honest reading of it at small ``n``."""

    student_only: int      # b: student right, teacher wrong
    teacher_only: int      # c: student wrong, teacher right
    statistic: float
    p_value: float

    @property
    def discordant(self) -> int:
        return self.student_only + self.teacher_only

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def headline(self) -> str:
        """The claim the evidence supports — which is usually the weaker one.

        At ``n = 50`` the discordant pairs might number five to eight, where
        McNemar has almost no power. "We could not detect a difference" and "the
        student matches the teacher" are different claims, and only the first is
        supported.
        """
        if self.discordant < 10:
            return (f"could not detect a difference ({self.discordant} discordant "
                    "pairs is too few to resolve one)")
        if self.significant:
            better = "student" if self.student_only > self.teacher_only else "teacher"
            return f"the {better} is better (p={self.p_value:.3f})"
        return f"no difference detected (p={self.p_value:.3f})"

    def as_dict(self) -> dict[str, Any]:
        return {
            "student_only": self.student_only,
            "teacher_only": self.teacher_only,
            "discordant": self.discordant,
            "chi2": round(self.statistic, 4),
            "p_value": round(self.p_value, 4),
            "headline": self.headline(),
        }


def mcnemar(student_correct: Sequence[bool],
            teacher_correct: Sequence[bool]) -> McNemarResult:
    """Paired comparison. Only the discordant pairs carry information.

    Student and teacher are evaluated on the *same* items, so an unpaired test
    would throw away the pairing and understate the evidence.
    """
    if len(student_correct) != len(teacher_correct):
        raise ValueError("paired comparison needs equal-length sequences")
    b = sum(1 for s, t in zip(student_correct, teacher_correct) if s and not t)
    c = sum(1 for s, t in zip(student_correct, teacher_correct) if t and not s)
    if b + c == 0:
        return McNemarResult(b, c, 0.0, 1.0)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)      # with continuity correction
    p = 2 * (1 - _NORMAL.cdf(math.sqrt(max(chi2, 0.0))))
    return McNemarResult(b, c, chi2, min(1.0, p))


__all__ = [
    "DEFAULT_ALPHA", "DEFAULT_BETA",
    "BayesResult", "FlipDecomposition", "Interval", "McNemarResult", "PowerReport",
    "beta_posterior", "bootstrap_variance", "certifiable_threshold", "certifies",
    "clopper_pearson", "clopper_pearson_lcb", "days_to_certify", "decompose_flips",
    "effective_n", "judge_usable_n", "mcnemar", "power_report", "required_n",
    "resolvable_difference", "wilson", "z",
]
