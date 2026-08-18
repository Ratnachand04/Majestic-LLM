"""Information gain by plan sampling — the Planner as the oracle (Part 4 §3).

**The novel piece.** Classical active learning ranks questions by the model's own
uncertainty: ask about whatever you are least sure of. That is the wrong
criterion here, and the failure is not subtle. Tone can be maximally ambiguous in
an extraction description and it is worth exactly zero questions, because no
coordinate of an extraction plan reads it. Entropy would put it near the top.

MAJESTIC has something active learning normally lacks: a **deterministic
downstream consumer**. The Planner is a pure function from spec to plan. So
instead of guessing which uncertainties matter, push each candidate answer
through the Planner and look at what comes out:

    IG(sigma_i)  =  H( Pi | sigma_i ~ p_hat_i )

and that identity is exact rather than a heuristic. The information the answer
carries about the plan is the mutual information ``I(A; Pi)``; the plan is a
*deterministic* function of the answer once everything else is fixed, so
``H(Pi | A) = 0`` and ``I(A; Pi) = H(Pi)``. The entropy of the induced
distribution over plans *is* the information gain.

Two consequences fall straight out:

* **All values give the same plan => IG = 0 => never ask.** However uncertain the
  slot is. This is the tone case, and it is a proof rather than a rule of thumb.
* **The values straddle admitted and refused => the highest gain there is.** That
  answer decides whether the customer gets anything at all, so it is worth the
  whole of ``V + kappa`` rather than the difference between two working models.

The compiler tells the interview what to ask. Nothing else in the pipeline gets
to reuse a component this way, and it is the reason FORGE terminates in four
questions instead of forty.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Sequence

from majestic.logging_utils import get_logger
from modelrig.forge import slots as slot_table
from modelrig.forge.posterior import ParsePosterior, candidate_values
from modelrig.ir import SpecIR

logger = get_logger(__name__)

#: What a plan *change* is worth, as a share of what a plan's *existence* is
#: worth. An admissibility flip risks the entire ``V + kappa``; swapping one
#: feasible base for another risks the quality difference between them.
#: HYPOTHESIS — the ratio is unvalidated; it is a stake weight, not a measurement.
DELTA_SHARE = 0.25

#: How many values of a slot to push through the Planner. Three covers the
#: binary slots exactly and the enum slots well enough; the cost is one full
#: plan() per value per slot per round.
DEFAULT_MAX_VALUES = 3


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlanSignature:
    """The coordinates of an outcome a customer would notice changing.

    Coarse where the customer cannot tell the difference and fine where they
    can. Two plans differing only in LoRA rank are the *same* answer, and
    counting them apart would manufacture gain out of implementation detail.
    But ``grammar_ref`` and ``eval_suite_ref`` stay in: they decide what shape
    comes out and what the model is certified against, which is the most visible
    difference there is. Drop them and the task primitive — the single
    highest-gain slot in the table — measures as worth nothing to ask.
    """

    admitted: bool
    base_ref: str | None = None
    quantiser: str | None = None
    target: str | None = None
    distil_mode: str | None = None
    grammar_ref: str | None = None
    eval_suite_ref: str | None = None
    #: The refusal's minimal witness set, as a tuple so the signature stays
    #: hashable — refusals are grouped by *why* they refused, and two refusals
    #: with different witnesses are genuinely different answers.
    witness: tuple[str, ...] = ()

    @classmethod
    def of(cls, outcome: Any) -> PlanSignature:
        plan_ir = getattr(outcome, "plan", None)
        if plan_ir is not None:
            return cls(
                admitted=True, base_ref=plan_ir.base_ref, quantiser=plan_ir.quantiser,
                target=plan_ir.target, distil_mode=plan_ir.distil_mode,
                grammar_ref=plan_ir.grammar_ref, eval_suite_ref=plan_ir.eval_suite_ref,
            )
        refusal = getattr(outcome, "refusal", None)
        return cls(admitted=False, witness=_witness(getattr(refusal, "witness", None)))

    def describe(self) -> str:
        if not self.admitted:
            return f"refused ({', '.join(self.witness) or 'no witness'})"
        suite = (self.eval_suite_ref or "").removeprefix("eval:")
        return f"{self.base_ref} @ {self.quantiser} -> {self.target} [{suite}]"


def _witness(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(sorted(str(v) for v in value))


@dataclass
class InfoGain:
    """What one slot's answer is worth, measured through the Planner."""

    slot: str
    gain: float                                   # H(Pi) / log2(n), in [0, 1]
    entropy_bits: float = 0.0
    values_tried: list[Any] = field(default_factory=list)
    outcomes: dict[Any, PlanSignature] = field(default_factory=dict)
    flips_admissibility: bool = False
    error: str | None = None

    @property
    def distinct_plans(self) -> int:
        return len(set(self.outcomes.values()))

    @property
    def stake(self) -> float:
        """The multiplier on ``Lambda`` this slot's answer carries (§4)."""
        return 1.0 if self.flips_admissibility else DELTA_SHARE

    @property
    def decision_relevant(self) -> bool:
        return self.gain > 0.0

    def reason(self) -> str:
        if self.error:
            return f"could not be scored: {self.error}"
        if not self.decision_relevant:
            return (
                f"every candidate answer produces the same plan "
                f"({next(iter(self.outcomes.values())).describe()}) "
                if self.outcomes else "no candidate answers to try "
            ) + "— the answer cannot change the build, so asking is pure attrition"
        if self.flips_admissibility:
            return (
                "the answers straddle admitted and refused — this one decides "
                "whether there is a model at all"
            )
        return (
            f"the answers produce {self.distinct_plans} different plans "
            f"({', '.join(sorted({s.describe() for s in self.outcomes.values()}))})"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "gain": round(self.gain, 4),
            "entropy_bits": round(self.entropy_bits, 4),
            "distinct_plans": self.distinct_plans,
            "flips_admissibility": self.flips_admissibility,
            "stake": self.stake,
            "values_tried": [_plain(v) for v in self.values_tried],
            "reason": self.reason(),
        }


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    return value


# --------------------------------------------------------------------------- #
SpecBuilder = Callable[[dict[str, Any]], SpecIR]
Oracle = Callable[[SpecIR], Any]


class PlanOracle:
    """A memoised ``plan()``. The interview asks it dozens of times a round.

    Memoisation is on the spec hash, which is sound precisely because the
    Planner is deterministic — the same property that makes it usable as an
    oracle at all.
    """

    def __init__(self, catalog: Any = None, planner: Callable[..., Any] | None = None) -> None:
        self._catalog = catalog
        self._planner = planner
        self._cache: dict[str, Any] = {}
        self.calls = 0
        self.hits = 0

    def _plan_fn(self) -> Callable[..., Any]:
        if self._planner is None:
            from modelrig.planner.core import plan as _plan

            self._planner = _plan
        return self._planner

    def _default_catalog(self) -> Any:
        if self._catalog is None:
            from modelrig.planner.catalog import default_catalog

            self._catalog = default_catalog()
        return self._catalog

    def __call__(self, spec: SpecIR) -> Any:
        key = spec.hash
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        outcome = self._plan_fn()(spec, self._default_catalog())
        self._cache[key] = outcome
        return outcome

    @property
    def stats(self) -> dict[str, int]:
        return {"plan_calls": self.calls, "cache_hits": self.hits,
                "cached_specs": len(self._cache)}


# --------------------------------------------------------------------------- #
def induced_plans(
    slot: str,
    values: Sequence[Any],
    build_spec: SpecBuilder,
    oracle: Oracle,
) -> tuple[dict[Any, PlanSignature], str | None]:
    """Push each candidate value through the Planner and collect the outcomes."""
    outcomes: dict[Any, PlanSignature] = {}
    error: str | None = None
    for value in values:
        try:
            spec = build_spec({slot: value})
            outcomes[_hashable(value)] = PlanSignature.of(oracle(spec))
        except Exception as exc:  # noqa: BLE001 - an unbuildable value is an outcome
            # A value that cannot even be lowered into a spec is not a silent
            # skip: it is a distinct, and maximally bad, outcome.
            error = f"{type(exc).__name__}: {exc}"
            outcomes[_hashable(value)] = PlanSignature(
                admitted=False, witness=("spec_invalid",)
            )
    return outcomes, error


def _hashable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(sorted(str(v) for v in value))
    if isinstance(value, dict):
        return tuple(sorted((str(k), str(v)) for k, v in value.items()))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def information_gain(
    slot: str,
    posterior: ParsePosterior | None,
    build_spec: SpecBuilder,
    oracle: Oracle,
    *,
    max_values: int = DEFAULT_MAX_VALUES,
    weights: dict[Any, float] | None = None,
) -> InfoGain:
    """``IG(sigma_i) = H(Pi | sigma_i)`` — measured, not guessed.

    ``weights`` lets a caller push the *posterior's* mass through instead of a
    uniform prior over the candidate values; by default the values are weighted
    uniformly, which is the honest reading of "we do not know which is meant".
    """
    entry = posterior.get(slot) if posterior is not None else None
    values = candidate_values(slot, entry, limit=max_values)
    if len(values) < 2:
        # Nothing to disagree about: one reading, or none we can enumerate.
        return InfoGain(slot=slot, gain=0.0, values_tried=list(values))

    outcomes, error = induced_plans(slot, values, build_spec, oracle)
    if not outcomes:
        return InfoGain(slot=slot, gain=0.0, values_tried=list(values), error=error)

    mass: dict[PlanSignature, float] = {}
    total = 0.0
    for value in values:
        key = _hashable(value)
        sig = outcomes.get(key)
        if sig is None:
            continue
        w = float((weights or {}).get(key, 1.0))
        mass[sig] = mass.get(sig, 0.0) + w
        total += w

    entropy = 0.0
    if total > 0:
        for w in mass.values():
            p = w / total
            if p > 0:
                entropy -= p * math.log2(p)
    ceiling = math.log2(len(values)) if len(values) > 1 else 0.0
    gain = entropy / ceiling if ceiling > 0 else 0.0

    admitted = {sig.admitted for sig in outcomes.values()}
    return InfoGain(
        slot=slot, gain=gain, entropy_bits=entropy, values_tried=list(values),
        outcomes=outcomes, flips_admissibility=len(admitted) > 1, error=error,
    )


def rank(
    names: Iterable[str],
    posterior: ParsePosterior | None,
    build_spec: SpecBuilder,
    oracle: Oracle,
    *,
    max_values: int = DEFAULT_MAX_VALUES,
) -> list[InfoGain]:
    """Score every candidate slot and order it by what its answer is worth.

    Ties break toward the slot the *table* already believed was high-gain, and
    then by name, so the interview is reproducible question for question.
    """
    scored = [
        information_gain(n, posterior, build_spec, oracle, max_values=max_values)
        for n in names
    ]
    scored.sort(key=lambda g: (-g.gain * g.stake, -_prior(g.slot), g.slot))
    return scored


def _prior(name: str) -> float:
    slot = slot_table.find(name)
    return slot.ig_class.prior if slot is not None else 0.0


def zero_gain(scored: Iterable[InfoGain]) -> list[str]:
    """Slots whose answers provably cannot move the plan. Never ask these."""
    return sorted(g.slot for g in scored if not g.decision_relevant)


__all__ = [
    "DEFAULT_MAX_VALUES", "DELTA_SHARE",
    "InfoGain", "Oracle", "PlanOracle", "PlanSignature", "SpecBuilder",
    "induced_plans", "information_gain", "rank", "zero_gain",
]
