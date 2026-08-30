"""The append-only corpus, and what it costs (Part 5 §14-§16).

Invariant **I-03**, stated formally:

    R_g subset-of R_{g+1}   for all g

The real corpus at generation ``g+1`` contains everything from generation ``g``.
Nothing is ever removed. That is what makes the collapse guarantee hold: if the
corpus can shrink, a later generation can lose the real examples anchoring it,
and the flywheel becomes a drift.

**The flywheel, quantified.** Corrections arrive at rate ``r`` per request over
volume ``V``, so ``|R_g|`` grows linearly in deployment time and ``n_eff`` grows
at least linearly with it. Quality improvement is therefore driven by ``r*V*t``
— the compounding rate is set by **deployment volume and correction rate**, not
by anything clever in the algorithm. There is no shortcut around shipping.

**§15 — the cost nobody had written down.** Append-only means unbounded storage
growth, and there is no code path that reduces it:

    Storage(t) = |R_0| + r*V*t

Negligible per customer, real at scale, and *permanent*. Three mitigations
preserve I-03 and are implemented here; three superficially similar ones do not
and are refused, because each would let the corpus shrink.

**§16 — synthetic data is a recipe, not an artefact.** The largest storage win
available, and it is free: store the four arguments that generate a synthetic
set rather than the set itself. Sixty thousand examples might be 200 MB; the
tuple that produces them is a few hundred bytes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

from majestic.logging_utils import get_logger

logger = get_logger(__name__)

KB = 1_000
MB = 1_000_000

#: Bytes per stored correction. Text plus provenance.
BYTES_PER_CORRECTION = 4 * KB

#: Compression achieved on cold generations. They are text.
# SOURCE: routine range for general-purpose text compression.
COLD_COMPRESSION = 7.0

#: Tolerance on regenerated synthetic statistics (§16). Regeneration is not
#: bit-identical across hardware, so equivalence is the achievable standard.
REGEN_TOLERANCE = 0.02


class CorpusError(ValueError):
    """An operation that would violate I-03."""


# =========================================================================== #
# §14 — monotone growth
# =========================================================================== #
@dataclass
class Generation:
    """One generation of the real corpus."""

    index: int
    record_ids: frozenset[str] = field(default_factory=frozenset)

    @property
    def size(self) -> int:
        return len(self.record_ids)


def check_monotone(generations: Sequence[Generation]) -> list[str]:
    """Verify ``R_g subset-of R_{g+1}`` across a history. Empty means sound.

    This is the one invariant the flywheel rests on, so it is checkable rather
    than assumed. A violation names the records that went missing, because
    "the corpus shrank" is not actionable and "these 12 ids were dropped between
    generation 4 and 5" is.
    """
    problems: list[str] = []
    for earlier, later in zip(generations, generations[1:]):
        lost = earlier.record_ids - later.record_ids
        if lost:
            problems.append(
                f"generation {later.index} dropped {len(lost)} record(s) present in "
                f"{earlier.index} (e.g. {sorted(lost)[:3]}): I-03 requires the real "
                "corpus to be append-only, and the collapse guarantee is void without it"
            )
    return problems


def append(generation: Generation, record_ids: Iterable[str]) -> Generation:
    """The only permitted mutation. Returns the next generation."""
    return Generation(
        index=generation.index + 1,
        record_ids=generation.record_ids | frozenset(record_ids),
    )


def effective_size(real: int, synthetic: int, rho_min: float = 0.10) -> dict[str, Any]:
    """``n_eff`` and whether the real:synthetic floor still holds.

    ``rho = |R| / |S|`` is enforced, so synthetic volume may grow proportionally
    with real corrections but never faster. That is what makes ``n_eff`` grow at
    least linearly in real data rather than in generated data.
    """
    rho = real / synthetic if synthetic else float("inf")
    # Amplification is worth less than real data, and worth less the thinner the
    # real anchor gets. eta is the discount that encodes it.
    eta = min(1.0, rho / rho_min) * 0.25 if rho_min > 0 else 0.0
    return {
        "real": real,
        "synthetic": synthetic,
        "rho": round(rho, 4) if synthetic else None,
        "rho_min": rho_min,
        "holds": rho >= rho_min,
        "n_eff": int(real + eta * synthetic),
    }


# =========================================================================== #
# §15 — the unstated cost, and what may be done about it
# =========================================================================== #
def growth(daily_volume: int, correction_rate: float, days: int,
           initial_bytes: int = 0,
           bytes_per_correction: int = BYTES_PER_CORRECTION) -> dict[str, Any]:
    """``Storage(t) = |R_0| + r*V*t``. Linear, and permanent.

    At 200 requests/day and a 6% correction rate that is 12 corrections a day,
    about 17 MB a year per customer. Individually negligible; the point is that
    it never goes down.
    """
    if daily_volume < 0 or days < 0 or not 0.0 <= correction_rate <= 1.0:
        raise ValueError("volume and days must be non-negative, rate in [0, 1]")
    corrections = daily_volume * correction_rate * days
    total = initial_bytes + corrections * bytes_per_correction
    return {
        "corrections_per_day": round(daily_volume * correction_rate, 2),
        "corrections": int(corrections),
        "bytes": int(total),
        "mb": round(total / MB, 2),
        "mb_per_year": round(daily_volume * correction_rate * 365
                             * bytes_per_correction / MB, 2),
        "note": "monotone: no code path reduces this",
    }


#: Mitigations that preserve I-03. The corpus keeps every record; only its
#: representation changes.
PERMITTED_MITIGATIONS: dict[str, str] = {
    "compress": "cold generations are text and compress 5-10x; every record survives",
    "content_dedup": "near-identical corrections collapse by MinHash — the CONTENT "
                     "dedups, the RECORD does not",
    "tier": "move old generations to cold storage; slower to read, still present",
}

#: Mitigations that look similar and are forbidden, because each lets the corpus
#: shrink. If it can shrink, the collapse guarantee is void.
FORBIDDEN_MITIGATIONS: dict[str, str] = {
    "delete": "removes records outright",
    "sample": "keeps a subset, which is deletion with extra steps",
    "window": "drops generations older than N, so R_g is not a subset of R_{g+1}",
    "ttl": "expiry is deletion on a timer",
}


def mitigation_allowed(name: str) -> bool:
    return name in PERMITTED_MITIGATIONS


def apply_mitigation(name: str, current_bytes: int) -> dict[str, Any]:
    """Project the saving, refusing anything that would violate I-03."""
    if name in FORBIDDEN_MITIGATIONS:
        raise CorpusError(
            f"{name!r} is forbidden under I-03: {FORBIDDEN_MITIGATIONS[name]}. The "
            "real corpus is append-only, and a corpus that can shrink voids the "
            "collapse guarantee"
        )
    if name not in PERMITTED_MITIGATIONS:
        raise CorpusError(f"unknown mitigation {name!r}")
    factor = COLD_COMPRESSION if name in ("compress", "tier") else 1.0
    return {
        "mitigation": name,
        "why_permitted": PERMITTED_MITIGATIONS[name],
        "bytes_before": current_bytes,
        "bytes_after": int(current_bytes / factor),
        "records_lost": 0,
    }


# =========================================================================== #
# §16 — synthetic data is a recipe
# =========================================================================== #
@dataclass(frozen=True)
class SyntheticRecipe:
    """The four arguments that generate a synthetic set. Store these, not it.

    ``S_g = generate(R_g, recipe, teacher_hash, rng_seed)``

    A 200 MB generated corpus reduces to a few hundred bytes, and the reduction
    is lossless in the only sense that matters: the set can be produced again.
    """

    real_corpus_hash: str
    recipe_name: str
    teacher_hash: str
    rng_seed: int
    n_generated: int = 0
    #: §16's honest caveat. Regeneration is NOT bit-identical across hardware,
    #: because floating-point reduction order differs between GPU models. So the
    #: original set's content hash and summary statistics are kept, and a
    #: regenerated set is accepted on *eval equivalence* rather than bit
    #: equality — the same standard already adopted for build reproducibility.
    content_hash: str = ""
    statistics: dict[str, float] = field(default_factory=dict)

    @property
    def ref(self) -> str:
        payload = json.dumps(
            [self.real_corpus_hash, self.recipe_name, self.teacher_hash, self.rng_seed],
            sort_keys=True,
        )
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()

    def stored_bytes(self) -> int:
        """What the recipe costs on disk, against what the set would have."""
        return len(json.dumps(asdict(self)).encode("utf-8"))

    def saving_against(self, generated_bytes: int) -> dict[str, Any]:
        stored = self.stored_bytes()
        return {
            "recipe_bytes": stored,
            "generated_bytes": generated_bytes,
            "saving": round(1.0 - stored / generated_bytes, 6) if generated_bytes else 0.0,
        }

    def equivalent(self, other_stats: dict[str, float],
                   tolerance: float = REGEN_TOLERANCE) -> tuple[bool, list[str]]:
        """Is a regenerated set close enough to the original to substitute?

        Bit equality is unavailable and demanding it would make the recipe
        useless on any machine but the one that ran it. Statistical equivalence
        within tolerance is the achievable standard.
        """
        drift: list[str] = []
        for key, expected in self.statistics.items():
            actual = other_stats.get(key)
            if actual is None:
                drift.append(f"{key}: missing from the regenerated set")
                continue
            scale = max(abs(expected), 1e-9)
            if abs(actual - expected) / scale > tolerance:
                drift.append(
                    f"{key}: {actual:.4f} against {expected:.4f}, outside "
                    f"{tolerance:.0%}"
                )
        return not drift, drift

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ref"] = self.ref
        return data


def storage_plan(real_bytes: int, synthetic_bytes: int,
                 recipe: SyntheticRecipe | None = None) -> dict[str, Any]:
    """What the registry actually stores, and what it declines to.

    The real corpus is kept forever because I-03 requires it. The synthetic set
    is not kept at all, because it is reproducible from four arguments — which
    is the largest available win and costs nothing.
    """
    recipe_bytes = recipe.stored_bytes() if recipe else 0
    stored = real_bytes + recipe_bytes
    naive = real_bytes + synthetic_bytes
    return {
        "real_bytes": real_bytes,
        "synthetic_bytes_not_stored": synthetic_bytes,
        "recipe_bytes": recipe_bytes,
        "stored_bytes": stored,
        "naive_bytes": naive,
        "saving": round(1.0 - stored / naive, 4) if naive else 0.0,
        "why": (
            "the real corpus is append-only and kept forever (I-03); the "
            "synthetic set is regenerable from four arguments and is not stored"
        ),
    }


__all__ = [
    "BYTES_PER_CORRECTION", "COLD_COMPRESSION", "FORBIDDEN_MITIGATIONS",
    "PERMITTED_MITIGATIONS", "REGEN_TOLERANCE",
    "CorpusError", "Generation", "SyntheticRecipe",
    "append", "apply_mitigation", "check_monotone", "effective_size", "growth",
    "mitigation_allowed", "storage_plan",
]
