"""DATA FACTORY — amplification with collapse guardrails (B-06).

Solves the single largest failure risk in the product: the customer has two
hundred examples and the build needs sixty thousand.

    real corpus -> IMMUTABLE SPLIT (seed | LOCKED held-out)
                -> amplify (backtranslate, evolve hard cases, rationale traces)
                -> QA GATES (each BLOCKS the build; none merely warns)
                -> curated training set

Two invariants carry the whole design:

**The held-out set is locked and never trained on.** It goes straight to the
Proving Ground. Everything the customer is later shown is measured on it.

**Accumulate, do not replace.** Every retraining generation keeps the whole real
corpus and mixes it in. This is what makes the flywheel safe
(accumulate-don't-replace, 2404.01413).

The bounding constraint is the Curse of Recursion (2305.17493): training on
generated data causes irreversible tail loss. Below a minimum real-seed floor the
factory REFUSES the build rather than shipping a model that will quietly
degrade.

GAP-05 (open): nobody has characterised what happens to a narrow task cartridge
retrained across twenty generations on accumulating human corrections. The
real:synthetic ratio floor used here
(:data:`MAX_SYNTHETIC_RATIO`) is a conservative guess, not a measured result.
"""
from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from majestic.logging_utils import get_logger
from majestic.perception.encoders import HashingTextEncoder
from modelrig.primitives import TaskPrimitive, spec_for

logger = get_logger(__name__)

Example = tuple[str, str]  # (text, label-or-target)

#: Share of the real corpus locked away for evaluation and never trained on.
HELD_OUT_FRACTION = 0.25
#: Ceiling on synthetic share of the training mix (GAP-05: not yet measured).
MAX_SYNTHETIC_RATIO = 0.95
#: Minimum Shannon entropy (nats) of the generated pool before collapse is called.
MIN_DIVERSITY_ENTROPY = 2.0
#: Minimum share of generated rows that must survive dedup. A generator whose
#: output collapses to a handful of distinct rows has degenerated, even when the
#: amplification frames keep raw token entropy superficially high.
MIN_SURVIVAL_RATE = 0.10

_TOKEN_RE = re.compile(r"[a-z0-9']+")

# PII patterns scrubbed BEFORE training — weights ship to devices and
# memorisation is measurable (B-06, B-07 privacy audit).
_PII_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("EMAIL", re.compile(r"\b[\w.%-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,3}[ -]?)?\d{10}\b")),
    ("AADHAAR", re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")),
    ("CARD", re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("IP", re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")),
)


class DataRefusal(ValueError):
    """Raised when the factory refuses to build rather than degrade quietly."""


@dataclass
class QAReport:
    """What each blocking gate removed or measured."""

    raw_count: int = 0
    after_exact_dedup: int = 0
    after_minhash_dedup: int = 0
    after_semantic_dedup: int = 0
    after_quality_filter: int = 0
    pii_redactions: int = 0
    diversity_entropy: float = 0.0
    survival_rate: float = 1.0
    synthetic_ratio: float = 0.0
    blocked: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.blocked


@dataclass
class DataBundle:
    """The factory's output: what to train on, and what was locked away."""

    train: list[Example]
    held_out: list[Example]
    labels: list[str]
    report: QAReport
    real_seed_count: int = 0
    synthetic_count: int = 0


# --------------------------------------------------------------------------- #
def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def scrub_pii(text: str) -> tuple[str, int]:
    """Redact PII before training. Returns ``(clean_text, redaction_count)``."""
    count = 0
    for tag, pattern in _PII_PATTERNS:
        text, n = pattern.subn(f"[{tag}]", text)
        count += n
    return text, count


def _minhash(text: str, num_perm: int = 32) -> tuple[int, ...]:
    """A small MinHash signature over token shingles."""
    toks = _tokens(text)
    shingles = {" ".join(toks[i : i + 3]) for i in range(max(len(toks) - 2, 1))} or set(toks)
    sig = []
    for seed in range(num_perm):
        best = min(
            (
                int.from_bytes(
                    hashlib.blake2b(f"{seed}:{s}".encode(), digest_size=8).digest(), "little"
                )
                for s in shingles
            ),
            default=0,
        )
        sig.append(best)
    return tuple(sig)


def _jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    return sum(1 for x, y in zip(a, b) if x == y) / len(a) if a else 0.0


def diversity_entropy(examples: Iterable[Example]) -> float:
    """Shannon entropy (nats) over the token distribution of the pool.

    The monitor that kills a run on distribution collapse. Without it,
    amplification degenerates silently.
    """
    counts: Counter[str] = Counter()
    for text, _ in examples:
        counts.update(_tokens(text))
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in counts.values())


# --------------------------------------------------------------------------- #
class DataFactory:
    """Seed-anchored amplification with blocking QA gates."""

    def __init__(
        self,
        held_out_fraction: float = HELD_OUT_FRACTION,
        target_size: int = 600,
        seed: int = 0,
        minhash_threshold: float = 0.85,
        semantic_threshold: float = 0.97,
    ) -> None:
        self.held_out_fraction = held_out_fraction
        self.target_size = target_size
        self.seed = seed
        self.minhash_threshold = minhash_threshold
        self.semantic_threshold = semantic_threshold
        self._encoder = HashingTextEncoder(dim=256)

    # -- the immutable split --------------------------------------------- #
    def split(self, rows: list[Example]) -> tuple[list[Example], list[Example]]:
        """Deterministically split into (seed, LOCKED held-out).

        The held-out rows are never returned to training by any later step.
        """
        rng = random.Random(self.seed)
        shuffled = rows[:]
        rng.shuffle(shuffled)
        n_held = max(1, int(len(shuffled) * self.held_out_fraction))
        return shuffled[n_held:], shuffled[:n_held]

    # -- amplification ---------------------------------------------------- #
    def _backtranslate(self, seeds: list[Example], rng: random.Random) -> list[Example]:
        """Generate instruction variants FOR the customer's real documents.

        The response side stays the customer's REAL data, so the teacher cannot
        hallucinate the answer. This is the primary path.
        """
        out: list[Example] = []
        frames = ("please handle: {}", "{} — what is this?", "input: {}",
                  "review the following. {}", "record says {}")
        for text, label in seeds:
            for frame in rng.sample(frames, k=min(3, len(frames))):
                out.append((frame.format(text), label))
        return out

    def _evolve(self, seeds: list[Example], rng: random.Random) -> list[Example]:
        """Deliberately evolve HARD cases: occlusion, drift, adversarial noise.

        A flat generator never produces these, and they are exactly the rows the
        deployed model will meet.
        """
        out: list[Example] = []
        for text, label in seeds:
            toks = text.split()
            if len(toks) > 3:
                # occlusion: a smudged scan drops a token
                drop = rng.randrange(len(toks))
                out.append((" ".join(t for i, t in enumerate(toks) if i != drop), label))
                # format drift: casing and separators change
                out.append((" | ".join(toks).upper(), label))
                # truncation: a missing field
                out.append((" ".join(toks[: max(2, len(toks) // 2)]), label))
        return out

    def rationale_trace(self, text: str, label: str) -> str:
        """A training-time-only reasoning trace.

        Traces transfer capability rather than style, and are STRIPPED from the
        deployed output contract (B-06).
        """
        return f"The salient terms point to {label}. Therefore: {label}."

    # -- the blocking QA gates -------------------------------------------- #
    def _qa(self, pool: list[Example], held_out: list[Example], report: QAReport) -> list[Example]:
        report.raw_count = len(pool)

        # 0. PII scrub — before training, not after.
        scrubbed: list[Example] = []
        for text, label in pool:
            clean, n = scrub_pii(text)
            report.pii_redactions += n
            scrubbed.append((clean, label))
        pool = scrubbed

        # 1. exact dedup, and train/eval leakage check
        held_keys = {t.strip().lower() for t, _ in held_out}
        seen: set[str] = set()
        exact: list[Example] = []
        for text, label in pool:
            key = text.strip().lower()
            if key in seen or key in held_keys:
                continue
            seen.add(key)
            exact.append((text, label))
        pool = exact
        report.after_exact_dedup = len(pool)

        # 2. MinHash near-duplicate removal
        kept: list[Example] = []
        signatures: list[tuple[int, ...]] = []
        for text, label in pool:
            sig = _minhash(text)
            if any(_jaccard(sig, s) >= self.minhash_threshold for s in signatures):
                continue
            signatures.append(sig)
            kept.append((text, label))
        pool = kept
        report.after_minhash_dedup = len(pool)

        # 3. semantic dedup — lexically varied, semantically identical rows.
        #    Vectorised: embeddings are L2-normalised, so a dot product is cosine.
        if pool:
            embedded = np.asarray(
                [self._encoder.encode(t) for t, _ in pool], dtype=np.float32
            )
            kept_idx: list[int] = []
            for i in range(len(pool)):
                if kept_idx:
                    sims = embedded[kept_idx] @ embedded[i]
                    if float(sims.max()) >= self.semantic_threshold:
                        continue
                kept_idx.append(i)
            pool = [pool[i] for i in kept_idx]
        report.after_semantic_dedup = len(pool)

        # 4. quality filter — drop degenerate rows
        pool = [(t, lab) for t, lab in pool if len(_tokens(t)) >= 2 and lab]
        report.after_quality_filter = len(pool)

        # 5. diversity monitors — these KILL the run; they do not warn.
        report.diversity_entropy = round(diversity_entropy(pool), 4)
        if report.diversity_entropy < MIN_DIVERSITY_ENTROPY:
            report.blocked.append(
                f"distribution collapse: diversity entropy "
                f"{report.diversity_entropy:.2f} < {MIN_DIVERSITY_ENTROPY}"
            )
        # Raw token entropy can be propped up by the amplification frames' own
        # vocabulary, so survival through dedup is the sharper collapse signal:
        # a generator emitting the same row a thousand ways has degenerated.
        report.survival_rate = (
            round(len(pool) / report.raw_count, 4) if report.raw_count else 0.0
        )
        if report.survival_rate < MIN_SURVIVAL_RATE:
            report.blocked.append(
                f"generator collapse: only {report.survival_rate:.1%} of generated "
                f"rows survived dedup (floor {MIN_SURVIVAL_RATE:.0%})"
            )
        return pool

    # -- the public entry point ------------------------------------------- #
    def build(
        self,
        rows: list[Example],
        primitive: TaskPrimitive | str,
        previous_real: list[Example] | None = None,
        with_rationales: bool = False,
    ) -> DataBundle:
        """Split, amplify, and QA a real corpus into a training bundle.

        ``previous_real`` carries the accumulated real corpus from earlier
        generations — it is MIXED IN, never replaced, which is what keeps the
        flywheel safe across generations.

        Raises :class:`DataRefusal` when the real-seed floor is not met or a
        blocking gate fires.
        """
        prim = spec_for(primitive)
        real = list(previous_real or []) + list(rows)

        if len(real) < prim.seed_floor:
            raise DataRefusal(
                f"only {len(real)} real examples for primitive "
                f"{prim.primitive.value!r}; the floor is {prim.seed_floor}. "
                "Refusing to amplify: training on generated data below the floor "
                "causes irreversible tail loss (2305.17493)."
            )

        seeds, held_out = self.split(real)
        labels = sorted({label for _, label in real})
        rng = random.Random(self.seed)

        # Scrub the REAL seeds too: they are trained on, and weights ship to
        # devices where memorisation is measurable. The held-out set is scrubbed
        # as well so the eval contract matches what the model was shown.
        seed_redactions = 0
        scrubbed_seeds: list[Example] = []
        for text, label in seeds:
            clean, n = scrub_pii(text)
            seed_redactions += n
            scrubbed_seeds.append((clean, label))
        seeds = scrubbed_seeds
        held_out = [(scrub_pii(t)[0], lab) for t, lab in held_out]

        generated = self._backtranslate(seeds, rng) + self._evolve(seeds, rng)
        if with_rationales:
            generated += [
                (f"{t}\n{self.rationale_trace(t, lab)}", lab) for t, lab in seeds
            ]

        report = QAReport()
        curated = self._qa(generated, held_out, report)
        report.pii_redactions += seed_redactions

        # Accumulate: the real seeds always ride along with the synthetic pool.
        train = seeds + curated
        rng.shuffle(train)
        if len(train) > self.target_size:
            # Never drop a real seed when trimming.
            synthetic = [row for row in train if row not in set(seeds)]
            keep_n = max(self.target_size - len(seeds), 0)
            train = seeds + synthetic[:keep_n]
            rng.shuffle(train)

        synthetic_count = max(len(train) - len(seeds), 0)
        report.synthetic_ratio = round(synthetic_count / len(train), 4) if train else 0.0
        if report.synthetic_ratio > MAX_SYNTHETIC_RATIO:
            report.blocked.append(
                f"synthetic ratio {report.synthetic_ratio:.2f} exceeds the ceiling "
                f"{MAX_SYNTHETIC_RATIO} (GAP-05: real floor is unmeasured)"
            )

        if report.blocked:
            raise DataRefusal("; ".join(report.blocked))

        logger.info(
            "data factory: %d real -> %d train (%d synthetic), %d LOCKED held-out, "
            "entropy %.2f, %d PII redactions",
            len(real), len(train), synthetic_count, len(held_out),
            report.diversity_entropy, report.pii_redactions,
        )
        return DataBundle(
            train=train,
            held_out=held_out,
            labels=labels,
            report=report,
            real_seed_count=len(seeds),
            synthetic_count=synthetic_count,
        )
