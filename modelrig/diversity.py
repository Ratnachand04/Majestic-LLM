"""Effective modes, saturation, coverage, and the amplification ceiling (§7-§10, §20).

Counting examples is the wrong measure. Sixty thousand near-identical rows is one
example repeated, so the quantity that matters is the **effective number of
distinct modes**:

    N_eff(S) = exp(H(S))        the Hill number of order 1

.. warning::

   **§19 — the citation this module used to rest on was at the wrong subsystem.**

   Every earlier document justified these guardrails by appeal to model collapse
   (the Curse of Recursion, 2305.17493). That paper describes **recursive**
   training: generation ``k+1`` trains on generation ``k``'s outputs, iterated,
   with distribution tails progressively lost.

   Single-build amplification here is **not recursive**. Every generation call is
   anchored on the customer's real seeds and no generated example is ever fed
   back as a source. The mechanism that produces collapse is structurally absent.

   What actually happens is **saturation**:

       collapse (recursive)          saturation (anchored)
       tails lost, compounding       bounded space fully covered
       more data is HARMFUL          more data is USELESS, not harmful
       drift across generations      flat N_eff against n_s
       remedy: inject real data      remedy: stop generating

   The distinction is not academic. Under a collapse model, over-generating
   *damages* the build and a guardrail must abort. Under saturation it merely
   wastes money, and the right response is to stop. **The existing design already
   did the right thing; the stated reason was wrong**, and the reason matters
   because it determines what happens when a guardrail fires.

   The citation belongs to the *flywheel*, where generation ``g+1`` training on
   generation ``g``'s outputs would be exactly the recursive structure — which is
   why I-03 forbids it and keeps the reference.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from majestic.logging_utils import get_logger

logger = get_logger(__name__)

#: §8 — the one remaining free constant. How much a distinct synthetic mode is
#: worth against a real example.
# HYPOTHESIS — R4 measures this. The old two-constant form (eta_0, rho_0) is
# recovered exactly by eta = eta_0 * N_eff / n_s, but N_eff is MEASURED per
# build where rho_0 was assumed globally, so the speculative constant is gone.
ETA_0 = 0.30

#: §9 — relative marginal gain below which generation stops.
SATURATION_EPSILON = 0.02

#: §10 — embedding distance within which a real example counts as covered.
COVERAGE_TAU = 0.35


# =========================================================================== #
# §7 — effective modes
# =========================================================================== #
def shannon_entropy(counts: Iterable[int]) -> float:
    """``H = -sum p log p`` in nats, over a cluster distribution."""
    counts = [c for c in counts if c > 0]
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in counts)


def effective_modes(cluster_labels: Sequence[Any]) -> float:
    """``N_eff = exp(H)`` — the Hill number of order 1.

    Uniform over ``K`` clusters gives ``K``; everything in one cluster gives 1;
    sixty thousand rows collapsed to a hundred modes gives about a hundred.
    That last case is the one example count cannot see.
    """
    if not len(cluster_labels):
        return 0.0
    return math.exp(shannon_entropy(Counter(cluster_labels).values()))


def vendi_score(similarity: Sequence[Sequence[float]]) -> float:
    """The exponential of the von Neumann entropy of the similarity matrix.

    More principled than clustering — it avoids the arbitrary clustering step
    entirely — and preferred whenever the set is small enough to admit an
    ``n x n`` matrix. Falls back to :func:`effective_modes` above that size.
    """
    n = len(similarity)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a core dependency
        raise RuntimeError("the Vendi score needs numpy; use effective_modes")

    kernel = np.asarray(similarity, dtype=float) / n
    eigenvalues = np.linalg.eigvalsh((kernel + kernel.T) / 2.0)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    if eigenvalues.size == 0:
        return 1.0
    entropy = float(-(eigenvalues * np.log(eigenvalues)).sum())
    return math.exp(entropy)


# =========================================================================== #
# §8 — the effective sample size
# =========================================================================== #
def effective_sample_size(n_real: int, n_eff_synthetic: float,
                          eta_0: float = ETA_0) -> float:
    """``n_eff = n_r + eta_0 * N_eff(S)``.

    **The discount factor is the diversity ratio itself**, measured per build
    rather than assumed. Two sets of sixty thousand rows contribute wildly
    differently:

        diverse   N_eff = 8000  ->  2400 at eta_0 = 0.3
        collapsed N_eff =  100  ->    30

    The old parametric form needed two invented constants; this needs one, and
    the more speculative of the pair disappears.
    """
    if n_real < 0 or n_eff_synthetic < 0:
        raise ValueError("counts must be non-negative")
    return n_real + eta_0 * n_eff_synthetic


def seed_predicate(n_real: int, n_eff_synthetic: float, floor: int,
                   eta_0: float = ETA_0) -> bool:
    """The revised ``g_seed``: does the *effective* set clear the floor?"""
    return effective_sample_size(n_real, n_eff_synthetic, eta_0) >= floor


# =========================================================================== #
# §9 — saturation stopping
# =========================================================================== #
@dataclass
class SaturationTrace:
    """Cumulative diversity per batch, and where it stopped growing."""

    batches: list[int] = field(default_factory=list)
    n_eff: list[float] = field(default_factory=list)
    stopped_at: int | None = None
    epsilon: float = SATURATION_EPSILON

    @property
    def saturated(self) -> bool:
        return self.stopped_at is not None

    @property
    def final(self) -> float:
        return self.n_eff[-1] if self.n_eff else 0.0

    def marginal_gain(self) -> float:
        if len(self.n_eff) < 2 or self.n_eff[-1] <= 0:
            return 1.0
        return (self.n_eff[-1] - self.n_eff[-2]) / self.n_eff[-1]

    def as_dict(self) -> dict[str, Any]:
        return {
            "batches": list(self.batches),
            "n_eff": [round(v, 2) for v in self.n_eff],
            "generated": self.batches[-1] if self.batches else 0,
            "stopped_at": self.stopped_at,
            "saturated": self.saturated,
            "final_n_eff": round(self.final, 2),
            "why": (
                "generation stops when diversity stops growing, not at a guessed "
                "target. The count is a reported measurement rather than an input"
            ),
        }


def generate_until_saturated(
    batch: Callable[[int], Sequence[Any]],
    *,
    batch_size: int = 2_000,
    max_examples: int = 60_000,
    epsilon: float = SATURATION_EPSILON,
    cluster_of: Callable[[Sequence[Any]], Sequence[Any]] | None = None,
) -> tuple[list[Any], SaturationTrace]:
    """Generate in batches, stopping when marginal diversity gain falls below
    ``epsilon`` (§9).

    ``max_examples`` is an **upper bound**, not a target — which is the
    correction: ``synthetic_target`` was a number somebody guessed, and the
    actual count should be determined by saturation and then reported.

    Two benefits. Wall clock, which is the binding constraint: saturating at
    20k instead of a planned 60k cuts generation from about 3.3 hours to 1.1 on
    a rented A100. And correctness — the run stops for a principled reason.

    This subsumes the separate "diversity monitor". Collapse in this regime is
    simply saturation at a low ``N_eff``, which then fails the seed predicate.
    """
    cluster_of = cluster_of or _default_clusters
    pool: list[Any] = []
    trace = SaturationTrace(epsilon=epsilon)

    while len(pool) < max_examples:
        pool.extend(batch(batch_size))
        trace.batches.append(len(pool))
        trace.n_eff.append(effective_modes(cluster_of(pool)))
        if len(trace.n_eff) >= 2 and trace.marginal_gain() < epsilon:
            trace.stopped_at = len(pool)
            logger.info(
                "data factory: diversity saturated at %d examples (N_eff=%.0f); "
                "generating further would cost money and add nothing",
                len(pool), trace.final,
            )
            break
    return pool, trace


def _default_clusters(examples: Sequence[Any]) -> list[str]:
    """A cheap stand-in for embedding + clustering: coarse content shape.

    High-cardinality tokens — identifiers, timestamps, random suffixes — are
    dropped, because keeping them makes every example its own cluster and
    ``N_eff`` degenerates to the row count. That failure is silent and it
    reports maximum diversity for a set that has none, which is precisely the
    opposite of what this measurement is for.

    Substitute a real embedding-and-cluster step whenever one is available; this
    exists so the saturation rule is testable without a model.
    """
    out = []
    for item in examples:
        text = item if isinstance(item, str) else str(getattr(item, "text", item))
        tokens = [t for t in text.lower().split()
                  if t.isalpha() or (len(t) <= 8 and not _is_high_entropy(t))]
        out.append(" ".join(sorted(set(tokens))[:12]))
    return out


def _is_high_entropy(token: str) -> bool:
    """Whether a token looks like an identifier rather than content."""
    digits = sum(c.isdigit() for c in token)
    return digits > 2 or "." in token


# =========================================================================== #
# §10 — coverage, measured without leaking
# =========================================================================== #
def coverage(reference: Sequence[Sequence[float]],
             synthetic: Sequence[Sequence[float]],
             tau: float = COVERAGE_TAU) -> float:
    """Nearest-neighbour coverage of a real reference set.

    ``N_eff`` measures *internal* diversity and says nothing about whether the
    set covers reality: a generator can produce eight thousand diverse modes all
    unlike the deployment distribution.
    """
    if not reference:
        return 0.0
    if not synthetic:
        return 0.0
    covered = 0
    for r in reference:
        nearest = min(_distance(r, s) for s in synthetic)
        covered += nearest < tau
    return covered / len(reference)


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


@dataclass
class CoverageReport:
    folds: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return sum(self.folds) / len(self.folds) if self.folds else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "folds": [round(f, 4) for f in self.folds],
            "mean_coverage": round(self.mean, 4),
            "measured_against": "seed folds",
            "why": (
                "measuring coverage against the HOLDOUT and regenerating until it "
                "improves is fitting to the holdout, and would silently invalidate "
                "every number the Proving Ground produces"
            ),
        }


def kfold_coverage(
    seeds: Sequence[Sequence[float]],
    generate: Callable[[Sequence[Sequence[float]]], Sequence[Sequence[float]]],
    *,
    k: int = 5,
    tau: float = COVERAGE_TAU,
) -> CoverageReport:
    """Coverage measured **inside the seed set**, never against the holdout.

    The leakage trap is real and quiet: measure coverage against the held-out
    set, regenerate until it improves, and you have fitted to the holdout —
    which invalidates the certificate rather than improving the model.

    K-fold inside the seeds avoids it. Every seed example serves both purposes,
    the holdout is never touched, and no third split is needed — which matters,
    because at ``n = 200`` a three-way split leaves nothing usable anywhere.
    """
    if k < 2 or len(seeds) < k:
        raise ValueError("k-fold coverage needs at least k >= 2 folds of seeds")
    report = CoverageReport()
    for fold in range(k):
        held = [s for i, s in enumerate(seeds) if i % k == fold]
        rest = [s for i, s in enumerate(seeds) if i % k != fold]
        if not held or not rest:
            continue
        report.folds.append(coverage(held, generate(rest), tau))
    return report


# =========================================================================== #
# §20 — the amplification ceiling
# =========================================================================== #
def amplification_ceiling(n_real: int, kappa: float) -> float:
    """``n_eff_max = n_r (1 + kappa)``.

    Combining the effective-sample formula with saturation gives a bound worth
    stating plainly: **amplification multiplies effective data by at most
    ``(1 + kappa)``, however much is generated.** Fifty seeds cannot become five
    thousand examples' worth of information; they become at most ``50(1+kappa)``.
    """
    if n_real < 0 or kappa < 0:
        raise ValueError("seed count and kappa must be non-negative")
    return n_real * (1.0 + kappa)


def seeds_required(floor: int, kappa: float) -> int:
    """``n_r >= n_0(p) / (1 + kappa)`` — the seed floor as a derivation.

    This turns a table lookup into an explanation, which is the point: a refusal
    can be stated in the customer's terms rather than as a threshold.
    """
    if kappa < 0:
        raise ValueError("kappa must be non-negative")
    return math.ceil(floor / (1.0 + kappa))


def explain_refusal(n_real: int, floor: int, kappa: float) -> dict[str, Any]:
    """The most useful sentence the system can produce on a failed build.

    Not "below the floor" but *"forty forms cannot reach this bar for this task
    — you need about ninety, and generating more will not substitute."*
    """
    needed = seeds_required(floor, kappa)
    ceiling = amplification_ceiling(n_real, kappa)
    return {
        "n_real": n_real,
        "seeds_required": needed,
        "amplification_ceiling": round(ceiling, 1),
        "floor": floor,
        "sufficient": n_real >= needed,
        "message": (
            f"{n_real} real examples amplify to at most {ceiling:.0f} effective "
            f"examples for this task, and the bar needs {floor}. You need about "
            f"{needed} real examples — **more seeds, not more generation**"
        ) if n_real < needed else (
            f"{n_real} real examples amplify to about {ceiling:.0f} effective, "
            f"which clears the {floor} the task needs"
        ),
    }


__all__ = [
    "COVERAGE_TAU", "ETA_0", "SATURATION_EPSILON",
    "CoverageReport", "SaturationTrace",
    "amplification_ceiling", "coverage", "effective_modes",
    "effective_sample_size", "explain_refusal", "generate_until_saturated",
    "kfold_coverage", "seed_predicate", "seeds_required", "shannon_entropy",
    "vendi_score",
]
