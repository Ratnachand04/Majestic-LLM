"""``parse_K`` — a posterior over slot values, not a point estimate (Part 4 §2).

A single parse gives one reading of the description and no way to tell a
confident reading from a lucky one. Parse **K** times and the spread over the
samples becomes measurable, which separates two failure modes that a point
estimate conflates:

* **Absence** — no sample produced a value. The description is silent. Whether
  to ask is a pure information-gain question.
* **Ambiguity** — the samples produced *different* values, confidently. The
  description says something and it admits more than one reading. "Works
  offline" is the canonical case, and silently defaulting it is the single most
  expensive thing FORGE can get wrong.

Entropy alone cannot tell those apart, which is why both ``H_i`` and ``A_i`` are
reported. Neither of them decides whether to ask: that is
:mod:`modelrig.forge.infogain`'s job, and §3's whole point is that a slot can be
maximally ambiguous and still have exactly zero information gain.

**Constrained decoding.** Every sample is projected onto the slot's declared
domain (:mod:`modelrig.forge.slots`), so an out-of-domain value is structurally
impossible and the posterior is a genuine distribution over the domain rather
than over strings.

**Sampling a deterministic parser.** The rule parser has no temperature, so
resampling it K times would give K identical draws and a spuriously confident
posterior of zero entropy. Instead we resample the *evidence*: each clause of
the description is dropped with probability ``ABLATION_RATE`` and the remainder
re-parsed. A slot supported by several redundant cues survives; a slot resting
on one ambiguous phrase flips. That measures ambiguity *in the text*, which is
exactly what ``A_i`` is supposed to be — and it degrades gracefully to the LLM
front end, where temperature sampling replaces the ablation and nothing
downstream changes.
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Protocol

from modelrig.forge import slots as slot_table

#: How much of the description each resample hides. High enough that a
#: single-cue slot flips, low enough that the description still parses.
ABLATION_RATE = 0.30

#: Default number of resamples. Beyond about 8 the entropy estimate stops
#: moving and the extra parses are wasted.
DEFAULT_K = 8

#: A slot is ambiguous when the modal reading holds less than this share of the
#: samples that produced a value at all.
AMBIGUITY_THRESHOLD = 0.75

#: Continuous slots are bucketed before counting, so 30 s and 31 s are the same
#: reading rather than two competing ones.
#:
#: The bucket width has to suit the slot, and getting it wrong is silent. Most
#: of these span orders of magnitude, so half-decade (sqrt-10) buckets are
#: right. ``quality_gate`` lives in ``[0, 1]`` and every plausible value —
#: 0.80, 0.90, 0.98 — falls in a *single* half-decade bucket, which would erase
#: the difference between a lenient gate and one nothing can meet.
_CONTINUOUS: dict[str, Callable[[float], str]] = {
    "latency_budget_ms": lambda x: _decade(x),
    "seed_data_count": lambda x: _decade(x),
    "budget_ceiling_usd": lambda x: _decade(x),
    "expected_input_tokens": lambda x: _decade(x),
    "expected_output_tokens": lambda x: _decade(x),
    "expected_daily_volume": lambda x: _decade(x),
    "quality_gate": lambda x: f"q{round(x, 2):.2f}",
}


class _Parser(Protocol):
    def parse(self, description: str, **known: Any) -> Any: ...


# --------------------------------------------------------------------------- #
def canonical_key(name: str, value: Any) -> Any:
    """A hashable, comparison-stable identity for a sampled slot value.

    Two samples share a key exactly when they are the *same reading*. Getting
    this wrong in either direction corrupts the posterior: too coarse and real
    disagreement disappears, too fine and rounding noise looks like ambiguity.
    """
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and name in _CONTINUOUS:
        return _CONTINUOUS[name](float(value))
    if isinstance(value, (list, tuple, set, frozenset)):
        return ("seq", tuple(sorted(str(v) for v in value)))
    if isinstance(value, dict):
        return ("map", tuple(sorted((str(k), str(v)) for k, v in value.items())))
    try:
        hash(value)
    except TypeError:
        return ("repr", repr(value))
    return value


def _decade(x: float) -> str:
    """Half-decade buckets, so nearby magnitudes count as one reading."""
    if x <= 0:
        return "0"
    return f"e{round(math.log10(x) * 2)}"


def _clauses(description: str) -> list[str]:
    """Split into the units an ablation can drop: sentences, then clauses."""
    parts = [p for p in re.split(r"(?<=[.!?;])\s+|\s*\n+\s*", description) if p.strip()]
    out: list[str] = []
    for part in parts:
        out.extend(p for p in re.split(r",\s+(?=\w)", part) if p.strip())
    return out or ([description] if description.strip() else [])


def ablate(description: str, rng: random.Random, rate: float = ABLATION_RATE) -> str:
    """Drop clauses at random, always keeping at least one."""
    parts = _clauses(description)
    if len(parts) <= 1:
        return description
    kept = [p for p in parts if rng.random() >= rate]
    if not kept:
        kept = [rng.choice(parts)]
    return " ".join(kept)


# --------------------------------------------------------------------------- #
@dataclass
class SlotPosterior:
    """The empirical distribution over one slot's readings across K samples."""

    name: str
    counts: Counter = field(default_factory=Counter)   # canonical key -> n
    representative: dict[Any, Any] = field(default_factory=dict)  # key -> a value
    samples: int = 0
    #: The parser flagged this reading as ambiguous outright (it knows about
    #: "when the internet dies" and does not need K samples to notice).
    declared_ambiguous: bool = False
    #: Mean confidence the parser reported, over the samples that produced a value.
    confidence: float = 0.0

    # -- shape ------------------------------------------------------------ #
    @property
    def filled_samples(self) -> int:
        return self.samples - self.counts.get(None, 0)

    @property
    def support(self) -> list[Any]:
        """The distinct non-null readings, most frequent first."""
        return [self.representative[k] for k, _ in self.counts.most_common() if k is not None]

    @property
    def probabilities(self) -> dict[Any, float]:
        if self.samples == 0:
            return {}
        return {k: n / self.samples for k, n in self.counts.items()}

    @property
    def mode(self) -> Any:
        """The most frequent reading, or ``None`` when the slot stayed empty."""
        for key, _ in self.counts.most_common():
            if key is not None:
                return self.representative[key]
        return None

    @property
    def mode_mass(self) -> float:
        """The modal reading's share of the samples that produced a value."""
        if self.filled_samples == 0:
            return 0.0
        top = max((n for k, n in self.counts.items() if k is not None), default=0)
        return top / self.filled_samples

    # -- the two numbers §2 keeps apart ----------------------------------- #
    @property
    def entropy(self) -> float:
        """``H_i`` in bits, over the full support *including* absence.

        Absence is a genuine outcome of the parse, so it belongs in ``H``. That
        is what makes ``H`` the wrong thing to threshold on its own: a slot the
        description never mentions has high ``H`` and may still be irrelevant.
        """
        if self.samples == 0:
            return 0.0
        h = 0.0
        for n in self.counts.values():
            p = n / self.samples
            if p > 0:
                h -= p * math.log2(p)
        return h

    @property
    def max_entropy(self) -> float:
        distinct = len(self.counts)
        return math.log2(distinct) if distinct > 1 else 0.0

    @property
    def normalised_entropy(self) -> float:
        """``H_i`` scaled into ``[0, 1]``, so slots with different-sized
        domains are comparable."""
        return self.entropy / self.max_entropy if self.max_entropy > 0 else 0.0

    @property
    def ambiguity(self) -> float:
        """``A_i`` — disagreement *conditional on a value being produced*.

        This is the half of the uncertainty the customer can actually resolve.
        A slot the description never mentions has ``A = 0``: there is nothing to
        disambiguate, only something to supply.
        """
        if self.filled_samples == 0:
            return 0.0
        return 1.0 - self.mode_mass

    @property
    def is_empty(self) -> bool:
        return self.filled_samples == 0

    @property
    def is_ambiguous(self) -> bool:
        return self.declared_ambiguous or (
            self.filled_samples > 1 and self.mode_mass < AMBIGUITY_THRESHOLD
        )

    @property
    def stable(self) -> bool:
        """Every sample that spoke, said the same thing."""
        return self.filled_samples > 0 and self.mode_mass >= 1.0 - 1e-9

    def as_dict(self) -> dict[str, Any]:
        return {
            "slot": self.name,
            "mode": _plain(self.mode),
            "support": [_plain(v) for v in self.support],
            "samples": self.samples,
            "filled_samples": self.filled_samples,
            "entropy_bits": round(self.entropy, 4),
            "normalised_entropy": round(self.normalised_entropy, 4),
            "ambiguity": round(self.ambiguity, 4),
            "confidence": round(self.confidence, 4),
            "empty": self.is_empty,
            "ambiguous": self.is_ambiguous,
        }


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    return value


# --------------------------------------------------------------------------- #
@dataclass
class ParsePosterior:
    """The K-sample posterior over the whole slot table."""

    description: str
    k: int
    slots: dict[str, SlotPosterior] = field(default_factory=dict)

    def __getitem__(self, name: str) -> SlotPosterior:
        return self.slots[name]

    def get(self, name: str) -> SlotPosterior | None:
        return self.slots.get(name)

    def modes(self) -> dict[str, Any]:
        """The point estimate a single parse would have produced."""
        return {n: p.mode for n, p in self.slots.items() if p.mode is not None}

    def empty(self) -> list[str]:
        """Slots the description never determined. *Supply* them."""
        return sorted(n for n, p in self.slots.items() if p.is_empty)

    def ambiguous(self) -> list[str]:
        """Slots read more than one way. *Disambiguate* them."""
        return sorted(n for n, p in self.slots.items() if p.is_ambiguous)

    def stable(self) -> list[str]:
        """Slots every sample agreed on. Do not ask about these."""
        return sorted(n for n, p in self.slots.items() if p.stable)

    def by_entropy(self) -> list[SlotPosterior]:
        """Most uncertain first. Deliberately *not* the ask order — §3."""
        return sorted(self.slots.values(), key=lambda p: -p.entropy)

    def as_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "empty": self.empty(),
            "ambiguous": self.ambiguous(),
            "stable": self.stable(),
            "slots": [p.as_dict() for p in self.slots.values()],
        }


# --------------------------------------------------------------------------- #
def project_onto_domain(name: str, value: Any) -> Any:
    """Constrained decoding: keep only values the slot's domain admits.

    Returns ``None`` for an out-of-domain sample, which counts as absence rather
    than as a competing reading — a hallucinated value must never be able to
    win a vote.
    """
    if value is None:
        return None
    slot = slot_table.find(name)
    if slot is None or not slot.domain:
        return value
    if value in slot.domain:
        return value
    raw = value.value if isinstance(value, Enum) else value
    if raw in slot.domain:
        return value
    if isinstance(raw, str) and raw.lower() in {str(d).lower() for d in slot.domain}:
        return value
    return None


def parse_K(
    parser: _Parser,
    description: str,
    *,
    k: int = DEFAULT_K,
    seed: int = 0,
    ablation_rate: float = ABLATION_RATE,
    known: dict[str, Any] | None = None,
    sampler: Callable[[str, random.Random], str] | None = None,
) -> ParsePosterior:
    """Parse ``k`` times over ablated evidence and return the posterior.

    ``seed`` is fixed by default: the interview must be reproducible, because a
    build that cannot be replayed cannot be audited. ``sampler`` exists so an
    LLM front end can substitute temperature sampling for the ablation without
    touching anything downstream.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    known = dict(known or {})
    rng = random.Random(seed)
    draw = sampler or (lambda text, r: ablate(text, r, ablation_rate))

    posterior = ParsePosterior(description=description, k=k)
    conf_totals: dict[str, list[float]] = {}

    for i in range(k):
        # Sample 0 is always the full description, so the posterior's mode can
        # never disagree with what a single honest parse would have said.
        text = description if i == 0 else draw(description, rng)
        state = parser.parse(text, **known)
        for name, s in state.slots.items():
            value = project_onto_domain(name, getattr(s, "value", None))
            entry = posterior.slots.setdefault(name, SlotPosterior(name=name))
            key = canonical_key(name, value)
            entry.counts[key] += 1
            entry.samples += 1
            if key is not None:
                entry.representative.setdefault(key, value)
                conf_totals.setdefault(name, []).append(float(getattr(s, "confidence", 0.0)))
            if getattr(s, "ambiguous", False):
                entry.declared_ambiguous = True

    for name, confs in conf_totals.items():
        if confs:
            posterior.slots[name].confidence = sum(confs) / len(confs)
    # A caller-supplied answer is not a sample; it is ground truth.
    for name in known:
        entry = posterior.slots.get(name)
        if entry is not None:
            entry.declared_ambiguous = False
    return posterior


def summarise(posterior: ParsePosterior) -> dict[str, Any]:
    """A compact report: what is missing, what is contested, what is settled."""
    return {
        "k": posterior.k,
        "n_slots": len(posterior.slots),
        "n_empty": len(posterior.empty()),
        "n_ambiguous": len(posterior.ambiguous()),
        "n_stable": len(posterior.stable()),
        "empty": posterior.empty(),
        "ambiguous": posterior.ambiguous(),
        "note": (
            "entropy ranks these; it does not decide them. A slot can be "
            "maximally ambiguous and have zero information gain (§3)."
        ),
    }


def candidate_values(name: str, entry: SlotPosterior | None, limit: int = 3) -> list[Any]:
    """The values worth pushing through the Planner when scoring this slot.

    Sampled readings first — they are what the description actually supports —
    then the declared domain, so a slot the parser never populated can still be
    scored. Without the fallback an empty slot would score zero information gain
    for the circular reason that nothing was sampled from it.
    """
    values: list[Any] = list(entry.support) if entry is not None else []
    slot = slot_table.find(name)
    fallback = (slot.domain or slot.probe_values) if slot is not None else ()
    if fallback:
        seen = {canonical_key(name, v) for v in values}
        for option in fallback:
            key = canonical_key(name, option)
            if key not in seen:
                seen.add(key)
                values.append(option)
    return values[:limit]


def iter_posteriors(posterior: ParsePosterior, names: Iterable[str]) -> Iterable[SlotPosterior]:
    for name in names:
        entry = posterior.get(name)
        if entry is not None:
            yield entry


__all__ = [
    "ABLATION_RATE", "AMBIGUITY_THRESHOLD", "DEFAULT_K",
    "ParsePosterior", "SlotPosterior",
    "ablate", "canonical_key", "candidate_values", "iter_posteriors",
    "parse_K", "project_onto_domain", "summarise",
]
