"""Refusal as a first-class output (§2.3, §7 line 16).

`Refusal` sits alongside `BuildPlanIR` in the Planner's return type. Every AutoML
system, every fine-tuning platform, always returns *something*. Majestic returns
nothing when nothing will work — and explains why in language the customer can
act on.

The explanation is the product. A refusal that says "infeasible" is worthless; a
refusal that says *"your tablet has 4 GB and this accuracy target needs a model
requiring 5.5 GB — either accept 0.88 accuracy or use an 8 GB device"* is the
thing being sold.

:func:`minimal_cover` is what makes that possible. Enumeration produces one
witness per eliminated candidate, which for a saturated space is thousands of
entries saying the same few things. The cover finds the **smallest set of
predicates that explains every elimination**, so the customer is told the two
real problems rather than forty redundant ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from modelrig.planner.predicates import HARD, SOFT, PredicateResult


@dataclass(frozen=True)
class Witness:
    """One eliminated candidate: which predicate killed it, and why."""

    predicate: str
    candidate: str          # a stable identifier for the rejected plan
    reason: str
    remedy: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def of(cls, candidate: str, result: PredicateResult) -> Witness:
        return cls(
            predicate=result.predicate, candidate=candidate,
            reason=result.reason, remedy=result.remedy, detail=dict(result.detail),
        )


@dataclass
class WitnessSet:
    """Every elimination recorded during enumeration."""

    witnesses: list[Witness] = field(default_factory=list)

    def add(self, candidate: str, result: PredicateResult) -> None:
        self.witnesses.append(Witness.of(candidate, result))

    def __len__(self) -> int:
        return len(self.witnesses)

    @property
    def candidates(self) -> set[str]:
        return {w.candidate for w in self.witnesses}

    @property
    def predicates(self) -> set[str]:
        return {w.predicate for w in self.witnesses}

    def by_predicate(self, name: str) -> list[Witness]:
        return [w for w in self.witnesses if w.predicate == name]

    def coverage(self) -> dict[str, set[str]]:
        """``predicate -> the set of candidates it eliminated``."""
        out: dict[str, set[str]] = {}
        for w in self.witnesses:
            out.setdefault(w.predicate, set()).add(w.candidate)
        return out


def minimal_cover(witnesses: WitnessSet) -> list[str]:
    """The smallest set of predicates explaining every elimination.

    This is exactly set cover: the universe is the eliminated candidates, and each
    predicate is the set it killed. Set cover is NP-hard, so this uses the
    standard greedy algorithm — repeatedly take the predicate covering the most
    still-uncovered candidates. Greedy is a ``ln n``-approximation, which is the
    best achievable in polynomial time unless P = NP, and at our scale (at most
    seven predicates) it is very often exactly optimal.

    Ties are broken toward **hard** predicates first, then by name. That is
    deliberate: a hard failure is certain and actionable, whereas a soft one
    rests on a claim that could be wrong, so when both explain the same
    eliminations the customer should be shown the certain one.
    """
    coverage = witnesses.coverage()
    if not coverage:
        return []

    uncovered = set(witnesses.candidates)
    chosen: list[str] = []
    remaining = dict(coverage)

    while uncovered and remaining:
        best = max(
            remaining,
            key=lambda p: (len(remaining[p] & uncovered), p in HARD, _neg_alpha(p)),
        )
        gain = remaining[best] & uncovered
        if not gain:
            break
        chosen.append(best)
        uncovered -= gain
        del remaining[best]

    # Present in evaluation order (hard first), not discovery order.
    return sorted(chosen, key=lambda p: (p not in HARD, p))


def _neg_alpha(name: str) -> tuple[int, ...]:
    """Reverse-alphabetical key so ``max`` breaks final ties alphabetically."""
    return tuple(-ord(c) for c in name)


@dataclass
class Refusal:
    """The Planner's other return type.

    ``witness`` names the predicates that eliminated the last surviving
    candidates. ``remedies`` are ordered by expected quality, so the first is the
    one the customer should probably take.
    """

    spec_hash: str
    witness: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    remedies: list[str] = field(default_factory=list)
    candidates_considered: int = 0
    candidates_eliminated: int = 0
    hard_failures: list[str] = field(default_factory=list)
    soft_failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def admitted(self) -> bool:
        return False

    @property
    def sound(self) -> bool:
        """True when at least one HARD predicate fired.

        A refusal carried by a hard predicate is certain (§16). One carried only
        by soft predicates rests on a calibration claim and should be reported as
        such rather than blended into an aggregate precision number.
        """
        return bool(self.hard_failures)

    def render(self) -> str:
        """The customer-facing account. This is what is actually sold."""
        lines = ["REFUSED — no feasible plan", ""]
        for name, reason in zip(self.witness, self.reasons):
            tag = "hard" if name in HARD else "soft"
            lines.append(f"  {name:<7} [{tag}] {reason}")
        if self.remedies:
            lines += ["", "  Remedies, in order of expected quality:"]
            lines += [f"  {i}. {r}" for i, r in enumerate(self.remedies, 1)]
        for note in self.notes:
            lines += ["", f"  NOTE: {note}"]
        lines += [
            "",
            f"  {self.candidates_eliminated} of {self.candidates_considered} candidate plans "
            "eliminated before any GPU was allocated.",
        ]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "spec_hash": self.spec_hash,
            "witness": list(self.witness),
            "reasons": list(self.reasons),
            "remedies": list(self.remedies),
            "hard_failures": list(self.hard_failures),
            "soft_failures": list(self.soft_failures),
            "sound": self.sound,
            "candidates_considered": self.candidates_considered,
            "candidates_eliminated": self.candidates_eliminated,
            "notes": list(self.notes),
        }


def build_refusal(
    spec_hash: str,
    witnesses: WitnessSet,
    considered: int,
    extra_notes: Iterable[str] = (),
) -> Refusal:
    """Turn a witness set into a refusal the customer can act on."""
    cover = minimal_cover(witnesses)
    reasons: list[str] = []
    remedies: list[str] = []
    for name in cover:
        group = witnesses.by_predicate(name)
        # The most informative witness for a predicate is the one whose candidate
        # got furthest — approximated by the longest reason, which carries the
        # most numeric detail.
        best = max(group, key=lambda w: len(w.reason))
        reasons.append(best.reason)
        if best.remedy and best.remedy not in remedies:
            remedies.append(best.remedy)

    return Refusal(
        spec_hash=spec_hash,
        witness=cover,
        reasons=reasons,
        remedies=remedies,
        candidates_considered=considered,
        candidates_eliminated=len(witnesses.candidates),
        hard_failures=[p for p in cover if p in HARD],
        soft_failures=[p for p in cover if p in SOFT],
        notes=list(extra_notes),
    )


def quality_refusal(
    spec_hash: str, predicted: float, threshold: float, considered: int,
    remedies: Sequence[str] = (),
) -> Refusal:
    """Refusal on predicted quality rather than on a hard constraint (§6).

    This is the *soft* case: the plan is feasible, but the model probably will
    not clear its gate. Whether refusing here is right depends entirely on
    whether the predictor is calibrated.
    """
    return Refusal(
        spec_hash=spec_hash,
        witness=["P_qual"],
        reasons=[
            f"predicted pass probability {predicted:.2f} is below the decision "
            f"threshold {threshold:.2f}; building is negative expected value"
        ],
        remedies=list(remedies),
        candidates_considered=considered,
        candidates_eliminated=0,
        soft_failures=["P_qual"],
        notes=[
            "this refusal rests on a calibration claim, not on physics or law — "
            "its correctness is exactly the open question in C2"
        ],
    )
