"""Measuring refusal quality without inflating it (§16, §2.2).

§16 is a correction to how the flagship claim must be evaluated, and it needs
code rather than prose because the failure mode is silent and flattering.

**The problem.** Partition the predicates by soundness. ``P_ram``, ``P_tok``,
``P_lic`` and ``P_off`` are sound *by construction*: a model that does not fit in
memory does not fit, undefined KL is undefined, an illegal licence chain is
illegal, and a reachability check over a finite graph is exact. Their precision
is exactly 1 and **no experiment can inform it**. So aggregate refusal precision
decomposes as

    Prec = (n_hard * 1 + n_soft * p_soft) / (n_hard + n_soft)

which is **inflatable by construction**. Padding an evaluation corpus with
memory-infeasible specs drives the aggregate toward 1 while measuring nothing at
all. A reviewer will spot it immediately, and it would undermine the flagship
claim rather than support it.

**What this module does.** It computes the honest number — ``p_soft``, stratified
by which predicate fired — refuses to report a headline that mixes the two, and
warns when the corpus itself is composed in a way that makes the result
meaningless. The corpus-composition guard is the part that matters: it is what
stops a well-meaning run from producing an impressive and worthless figure.

The framing that follows: **the novelty is that the hard predicates are checked
at all — nobody does that. The research question is whether the soft ones can be
calibrated well enough to be worth checking.** Two different claims, reported
separately.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from majestic.logging_utils import get_logger
from modelrig.planner.predicates import HARD, SOFT
from modelrig.planner.refusal import Refusal

logger = get_logger(__name__)

#: Below this share of soft-only refusals, the corpus cannot support a claim
#: about soft-predicate precision — whatever the arithmetic returns.
# HYPOTHESIS — a corpus-design guard, not a measurement. Half is the point at
# which the hard predicates stop dominating the aggregate.
MIN_SOFT_SHARE = 0.50

#: C2's bar for soft-predicate refusal precision.
C2_PRECISION_TARGET = 0.60


@dataclass(frozen=True)
class RefusalOutcome:
    """One evaluated refusal: what fired, and whether refusing was right.

    ``would_have_failed`` is the ground truth, and it is expensive: it means
    somebody built the thing anyway and observed whether it cleared the gate. A
    corpus of these is the only way to say anything about ``p_soft``.
    """

    spec_hash: str
    witness: tuple[str, ...]
    would_have_failed: bool
    note: str = ""

    @property
    def hard_fired(self) -> tuple[str, ...]:
        return tuple(p for p in self.witness if p in HARD)

    @property
    def soft_fired(self) -> tuple[str, ...]:
        return tuple(p for p in self.witness if p in SOFT)

    @property
    def carried_by_hard(self) -> bool:
        """A refusal any hard predicate carries is certain, whatever else fired."""
        return bool(self.hard_fired)

    @property
    def soft_only(self) -> bool:
        """The only refusals that carry information about calibration."""
        return bool(self.soft_fired) and not self.hard_fired

    @classmethod
    def of(cls, refusal: Refusal, would_have_failed: bool, note: str = "") -> RefusalOutcome:
        return cls(
            spec_hash=refusal.spec_hash,
            witness=tuple(refusal.witness),
            would_have_failed=would_have_failed,
            note=note,
        )


@dataclass(frozen=True)
class PrecisionEstimate:
    """A precision with its support and a Wilson interval.

    The interval is not decoration. At the corpus sizes available here a point
    estimate of 0.62 over 13 refusals is not evidence of clearing a 0.60 bar,
    and reporting it without the interval would be the same species of error
    §16 is correcting.
    """

    correct: int
    total: int
    label: str = ""

    @property
    def precision(self) -> float | None:
        return self.correct / self.total if self.total else None

    def wilson(self, z: float = 1.96) -> tuple[float, float] | None:
        """95% Wilson score interval — well-behaved at small n and near 1."""
        if not self.total:
            return None
        n, p = self.total, self.correct / self.total
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        halfwidth = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return max(0.0, centre - halfwidth), min(1.0, centre + halfwidth)

    def clears(self, target: float = C2_PRECISION_TARGET) -> bool:
        """Whether the *lower bound* clears the bar. A point estimate does not."""
        interval = self.wilson()
        return interval is not None and interval[0] >= target

    def as_dict(self) -> dict[str, Any]:
        interval = self.wilson()
        return {
            "label": self.label,
            "precision": round(self.precision, 4) if self.precision is not None else None,
            "correct": self.correct,
            "total": self.total,
            "wilson_95": [round(interval[0], 4), round(interval[1], 4)] if interval else None,
        }


@dataclass
class RefusalAudit:
    """The §16 report. Stratified by construction, so it cannot be blurred."""

    outcomes: list[RefusalOutcome] = field(default_factory=list)

    def add(self, outcome: RefusalOutcome) -> RefusalOutcome:
        self.outcomes.append(outcome)
        return outcome

    def record(self, refusal: Refusal, would_have_failed: bool,
               note: str = "") -> RefusalOutcome:
        return self.add(RefusalOutcome.of(refusal, would_have_failed, note))

    # -- the partition ---------------------------------------------------- #
    @property
    def hard_carried(self) -> list[RefusalOutcome]:
        return [o for o in self.outcomes if o.carried_by_hard]

    @property
    def soft_only(self) -> list[RefusalOutcome]:
        return [o for o in self.outcomes if o.soft_only]

    @property
    def soft_share(self) -> float:
        return len(self.soft_only) / len(self.outcomes) if self.outcomes else 0.0

    # -- the numbers ------------------------------------------------------ #
    def aggregate_precision(self) -> PrecisionEstimate:
        """The number §16 forbids as a headline. Computed so it can be *shown*
        to be inflated, next to the honest one."""
        correct = sum(o.would_have_failed for o in self.outcomes)
        return PrecisionEstimate(correct, len(self.outcomes), "aggregate (INFLATED)")

    def soft_precision(self) -> PrecisionEstimate:
        """``p_soft`` — the primary result, and the only informative one."""
        soft = self.soft_only
        return PrecisionEstimate(
            sum(o.would_have_failed for o in soft), len(soft), "soft-only (primary)"
        )

    def by_predicate(self) -> dict[str, PrecisionEstimate]:
        """Per-predicate precision over soft-only refusals.

        ``P_seed`` is the scientifically interesting row: it is the only refusal
        grounded in a statistical claim about learnability rather than in physics
        or law.
        """
        correct: Counter[str] = Counter()
        total: Counter[str] = Counter()
        for outcome in self.soft_only:
            for name in outcome.soft_fired:
                total[name] += 1
                correct[name] += outcome.would_have_failed
        return {
            name: PrecisionEstimate(correct[name], total[name], name)
            for name in sorted(total)
        }

    def hard_precision_is_unmeasurable(self) -> dict[str, Any]:
        """Hard predicates are sound by construction. State it; never compute it."""
        fired: Counter[str] = Counter()
        for outcome in self.outcomes:
            fired.update(outcome.hard_fired)
        return {
            "predicates": list(HARD),
            "precision": 1.0,
            "measured": False,
            "fired": dict(fired),
            "why": (
                "sound by construction: a model that does not fit in memory does "
                "not fit, undefined KL is undefined, an illegal licence chain is "
                "illegal. No experiment can inform this number"
            ),
        }

    # -- the guard that makes the rest worth reading ---------------------- #
    def corpus_warnings(self, min_soft_share: float = MIN_SOFT_SHARE) -> list[str]:
        """Whether this corpus can support a claim at all.

        §16 step 3: design the corpus so soft-only refusals dominate — specs that
        are memory-feasible, licence-clean and tokenizer-consistent, but marginal
        on seed sufficiency or achievable quality.
        """
        warnings: list[str] = []
        if not self.outcomes:
            return ["the corpus is empty: nothing can be concluded"]
        if self.soft_share < min_soft_share:
            warnings.append(
                f"only {self.soft_share:.0%} of refusals are soft-only, below the "
                f"{min_soft_share:.0%} floor: the aggregate is dominated by predicates "
                "whose precision is 1 by construction, so it measures corpus "
                "composition rather than calibration"
            )
        soft = self.soft_precision()
        if soft.total < 30:
            warnings.append(
                f"only {soft.total} soft-only refusals: too few to separate "
                f"{C2_PRECISION_TARGET:.2f} from chance — report the interval, not "
                "the point estimate"
            )
        if not any(o.soft_fired for o in self.outcomes):
            warnings.append(
                "no soft predicate fired anywhere in this corpus: it tests the "
                "hard predicates only, which no experiment can inform"
            )
        return warnings

    def inflation(self) -> float | None:
        """How much the aggregate overstates the honest number, in points."""
        agg, soft = self.aggregate_precision().precision, self.soft_precision().precision
        return None if agg is None or soft is None else agg - soft

    # -- the report ------------------------------------------------------- #
    def report(self, min_soft_share: float = MIN_SOFT_SHARE) -> dict[str, Any]:
        soft = self.soft_precision()
        warnings = self.corpus_warnings(min_soft_share)
        return {
            "n_refusals": len(self.outcomes),
            "n_hard_carried": len(self.hard_carried),
            "n_soft_only": len(self.soft_only),
            "soft_share": round(self.soft_share, 4),
            "primary": soft.as_dict(),
            "clears_c2": soft.clears(),
            "by_predicate": {k: v.as_dict() for k, v in self.by_predicate().items()},
            "hard": self.hard_precision_is_unmeasurable(),
            "aggregate_do_not_report": self.aggregate_precision().as_dict(),
            "inflation_points": (
                round(i, 4) if (i := self.inflation()) is not None else None
            ),
            "corpus_warnings": warnings,
            "usable": not warnings,
        }

    def render(self) -> str:
        """The §16 report as text, with the two claims kept apart."""
        r = self.report()
        lines = [
            "REFUSAL AUDIT (§16)",
            "",
            "  Claim 1 — the hard predicates are checked at all. Nobody does that.",
            f"    {', '.join(HARD)}: sound by construction, precision 1, unmeasurable.",
            f"    fired on {r['n_hard_carried']} of {r['n_refusals']} refusals.",
            "",
            "  Claim 2 — can the soft predicates be calibrated well enough to be worth",
            "  checking? This is the research question, and the only measurable one.",
        ]
        primary = r["primary"]
        if primary["precision"] is None:
            lines.append("    no soft-only refusals: nothing measured.")
        else:
            lo, hi = primary["wilson_95"]
            verdict = "clears" if r["clears_c2"] else "does NOT clear"
            lines.append(
                f"    p_soft = {primary['precision']:.3f} "
                f"[{lo:.3f}, {hi:.3f}] over n={primary['total']} "
                f"— {verdict} the {C2_PRECISION_TARGET:.2f} bar on the lower bound."
            )
        if r["by_predicate"]:
            lines += ["", "  Stratified by which predicate fired:"]
            for name, est in r["by_predicate"].items():
                interval = est["wilson_95"]
                lines.append(
                    f"    {name:<9} {est['precision']:.3f} "
                    f"[{interval[0]:.3f}, {interval[1]:.3f}]  n={est['total']}"
                )
        agg = r["aggregate_do_not_report"]
        if agg["precision"] is not None and r["inflation_points"]:
            lines += [
                "",
                f"  The aggregate would read {agg['precision']:.3f} — "
                f"{r['inflation_points']:+.3f} against the honest figure. It is "
                "inflatable by",
                "  corpus composition and must not be reported as the headline.",
            ]
        if r["corpus_warnings"]:
            lines += ["", "  CORPUS WARNINGS — the numbers above are not yet usable:"]
            lines += [f"    - {w}" for w in r["corpus_warnings"]]
        return "\n".join(lines)


def audit(outcomes: Iterable[RefusalOutcome]) -> RefusalAudit:
    return RefusalAudit(outcomes=list(outcomes))


# =========================================================================== #
# §2.2 — the structural claim, made checkable
# =========================================================================== #
def plan_space_size(catalog: Any = None) -> dict[str, Any]:
    """Compute ``|Pi|`` over the live catalogue, rather than asserting it.

    §2.2 rests the entire algorithm on one number: the plan space is about
    ``1.4e6`` points, which is enumerable with pruning, where NAS operates over
    ``1e18`` and is forced into search. That claim justifies rejecting search, so
    it should be *computed* — a catalogue that grew by an order of magnitude
    without anyone noticing would quietly invalidate it.

    Note that §12 makes the stronger argument for the dominant dimension: base
    selection is a maximum over a chain, so there is nothing to enumerate there
    at all. This number is the fallback justification, not the main one.
    """
    from modelrig.planner.catalog import default_catalog
    from modelrig.planner.core import PEFT_METHODS, QUANTISERS, RANKS, TARGETS

    catalog = catalog or default_catalog()
    dims = {
        "base": max(len(catalog.bases()), 1),
        "teacher": max(len(catalog.teachers()), 1),
        "distil_mode": 3,          # logit_kd | sequence_kd | none
        "peft_method": len(PEFT_METHODS),
        "rank": len(RANKS),
        "data_recipe": 8,          # strategy subset x volume
        "quantiser": len(QUANTISERS),
        "bit_width": 5,
        "target": len(TARGETS),
    }
    size = math.prod(dims.values())
    return {
        "dimensions": dims,
        "size": size,
        "log10": round(math.log10(size), 2) if size else None,
        "enumerable": size <= 1e8,
        "nas_log10": 18,
        "why": (
            "at 1e6 with pruning, exhaustive enumeration terminates in "
            "milliseconds; at NAS's 1e18 it does not. Rejecting search is the "
            "correct algorithm for this regime, not a shortcut taken for cost"
        ),
    }


def enumeration_economy(
    considered: int, predicate_evaluations: int, n_predicates: int = len(HARD) + len(SOFT)
) -> dict[str, Any]:
    """What the §4 ordering actually bought on a real run.

    Predicates form a conjunction, so ordering cannot affect correctness — only
    cost. Sorting ascending on ``c_i/(1-rho_i)`` minimises expected evaluations,
    and the saving is the gap between this and checking all of them.
    """
    if considered <= 0:
        return {"considered": 0, "evaluations_per_candidate": None}
    per = predicate_evaluations / considered
    return {
        "considered": considered,
        "predicate_evaluations": predicate_evaluations,
        "evaluations_per_candidate": round(per, 2),
        "without_early_exit": n_predicates,
        "saving": round(1.0 - per / n_predicates, 4) if n_predicates else None,
    }


def summarise_witnesses(refusals: Sequence[Refusal]) -> dict[str, Any]:
    """Which predicates carry refusals in practice, split by soundness.

    Useful for corpus design: a corpus whose witnesses are all ``P_ram`` is a
    corpus that cannot say anything about calibration.
    """
    hard: Counter[str] = Counter()
    soft: Counter[str] = Counter()
    for refusal in refusals:
        for name in refusal.witness:
            (hard if name in HARD else soft)[name] += 1
    return {
        "hard": dict(hard.most_common()),
        "soft": dict(soft.most_common()),
        "n_refusals": len(refusals),
        "soft_only": sum(1 for r in refusals if not r.sound),
    }


__all__ = [
    "C2_PRECISION_TARGET", "MIN_SOFT_SHARE",
    "PrecisionEstimate", "RefusalAudit", "RefusalOutcome",
    "audit", "enumeration_economy", "plan_space_size", "summarise_witnesses",
]
