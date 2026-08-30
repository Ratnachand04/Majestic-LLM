"""Augmentation: the operation that was hiding inside "synthetic data" (§1-§4).

**The reframe.** Every earlier document called this subsystem "synthetic data
generation", which conflates three operations with radically different risk:

    augmentation      (d, y) -> (T(d), y)    label REAL by construction
    backtranslation   y -> (G(y), y)         output real, input generated
    synthesis         {} -> (G1, G2)         both generated

Priority order is augmentation, then backtranslation, then synthesis. Use
synthesis only when the first two are unavailable.

**Why this matters more than it looks.** For ``extract`` the customer usually
already has ``(document, label)`` pairs and does not know it: a clinic that has
processed lab requisitions manually for years has the extracted fields sitting in
its operational database. The document is the input, the database record is the
label. So the task is not to invent pairs — it is to generate realistic
variations of real inputs *whose labels are known to be unchanged*.

Three consequences follow, and all three are practical:

* **Collapse risk is primitive-dependent**, because label provenance is. This
  explains the seed-floor differences as a mechanism rather than an observation.
* **Augmentation costs approximately nothing** — no teacher inference is needed
  for a label-preserving transform. The dollar figures quoted for generation
  apply to synthesis-dominated primitives, not to extraction.
* **No quality filter is needed.** Labels are correct by construction, so a
  filter can only discard valid examples.
"""
from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Sequence

from majestic.logging_utils import get_logger
from modelrig.primitives import TaskPrimitive

logger = get_logger(__name__)

#: §2 — composition depth cap. Beyond about three, ``T^k(d)`` drifts off the
#: manifold of realistic inputs: a document blurred, rotated, occluded and
#: re-encoded five times is not a document anyone will ever scan.
MAX_DEPTH = 3

#: §21 — per-source multiplicity cap. Memorisation probability rises sharply
#: with duplication count, and Majestic ships weights to devices, so the
#: attacker has white-box access. Twenty transforms of one seed are twenty
#: near-copies of that seed's content.
# HYPOTHESIS — the extraction-rate/multiplicity curve is not measured here.
MAX_MULTIPLICITY = 8

#: §3 — how many augmented samples a human must eyeball per build. The one
#: place in the pipeline where human judgement is not replaceable, and it takes
#: about ten minutes.
SPOT_CHECK_SAMPLES = 100


class Operation(str, Enum):
    """Which of the three operations produced an example (§1)."""

    REAL = "real"
    AUGMENTATION = "augmentation"
    BACKTRANSLATION = "backtranslation"
    SYNTHESIS = "synthesis"

    @property
    def label_is_real(self) -> bool:
        """Whether the LABEL came from the customer rather than a teacher."""
        return self in (Operation.REAL, Operation.AUGMENTATION,
                        Operation.BACKTRANSLATION)

    @property
    def collapse_risk(self) -> str:
        return {
            "real": "none", "augmentation": "very low",
            "backtranslation": "medium", "synthesis": "high",
        }[self.value]

    @property
    def needs_quality_filter(self) -> bool:
        """§12 — augmented data needs none: its labels are correct by
        construction, so a filter can only discard valid examples."""
        return self in (Operation.BACKTRANSLATION, Operation.SYNTHESIS)

    @property
    def teacher_tokens_per_example(self) -> int:
        """§1 — augmentation requires no teacher inference at all."""
        return 0 if self in (Operation.REAL, Operation.AUGMENTATION) else 300


#: §1 — the dominant operation per primitive, and hence its collapse risk. The
#: seed floors differ because LABEL PROVENANCE differs, which is a mechanism
#: rather than an observation.
DOMINANT_OPERATION: dict[TaskPrimitive, Operation] = {
    TaskPrimitive.EXTRACT: Operation.AUGMENTATION,
    TaskPrimitive.CLASSIFY: Operation.AUGMENTATION,
    TaskPrimitive.ROUTE: Operation.AUGMENTATION,
    TaskPrimitive.ANSWER: Operation.BACKTRANSLATION,
    TaskPrimitive.SUMMARISE: Operation.BACKTRANSLATION,
    TaskPrimitive.REWRITE: Operation.SYNTHESIS,
    TaskPrimitive.GENERATE: Operation.SYNTHESIS,
    TaskPrimitive.TOOLCALL: Operation.SYNTHESIS,
}


# =========================================================================== #
# §2 — the operator monoid
# =========================================================================== #
@dataclass(frozen=True)
class Operator:
    """A label-preserving transform.

    Admissible iff ``y(T(d)) = y(d)`` for every ``d``. Admissible operators are
    closed under composition, so a small set yields combinatorial coverage — but
    only up to :data:`MAX_DEPTH`, past which the output stops resembling
    anything a scanner would produce.
    """

    name: str
    apply: Callable[[str, random.Random], str]
    primitives: tuple[TaskPrimitive, ...] = ()
    note: str = ""
    #: §3 — realism comes from parameterising the operator with domain
    #: knowledge, not from proximity to the seeds. Proximity is the WRONG test:
    #: the point of hard-case generation is to produce inputs the seed set does
    #: not contain.
    parameter_source: str = "domain knowledge"

    def applies_to(self, primitive: TaskPrimitive) -> bool:
        return not self.primitives or primitive in self.primitives


def _occlude(text: str, rng: random.Random) -> str:
    """A staple mark or thumb over part of the page."""
    words = text.split()
    if len(words) < 4:
        return text
    i = rng.randrange(len(words) - 1)
    words[i] = "[...]"
    return " ".join(words)


def _whitespace_drift(text: str, rng: random.Random) -> str:
    return re.sub(r"\s+", lambda _m: " " * rng.randint(1, 3), text)


def _typo(text: str, rng: random.Random) -> str:
    words = text.split()
    if not words:
        return text
    i = rng.randrange(len(words))
    w = words[i]
    if len(w) > 3:
        j = rng.randrange(len(w) - 1)
        words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def _reorder_fields(text: str, rng: random.Random) -> str:
    """Field reordering — a genuinely hard case, generated for free.

    It exercises exactly the invariance the behavioural tests check, which is
    why augmentation operators and behavioural test generators are the same
    objects seen from two directions. Build them once.
    """
    parts = [p for p in text.split("\n") if p.strip()]
    if len(parts) < 2:
        return text
    rng.shuffle(parts)
    return "\n".join(parts)


def _ocr_noise(text: str, rng: random.Random) -> str:
    """Character confusions a real OCR pass produces."""
    confusions = {"0": "O", "1": "l", "5": "S", "8": "B", "rn": "m"}
    for src, dst in confusions.items():
        if src in text and rng.random() < 0.3:
            text = text.replace(src, dst, 1)
    return text


def _abbreviate(text: str, rng: random.Random) -> str:
    forms = {"Street": "St", "Doctor": "Dr", "Laboratory": "Lab",
             "Department": "Dept", "Number": "No"}
    for long, short in forms.items():
        if long in text and rng.random() < 0.5:
            text = text.replace(long, short)
    return text


DOCUMENT_OPERATORS: tuple[Operator, ...] = (
    Operator("occlusion", _occlude, (TaskPrimitive.EXTRACT,),
             "staple marks and thumbs, from real page photographs",
             parameter_source="observed occlusion shapes"),
    Operator("ocr_noise", _ocr_noise, (TaskPrimitive.EXTRACT,),
             "character confusions a real OCR pass produces",
             parameter_source="scanner MTF and OCR confusion matrix"),
    Operator("field_reorder", _reorder_fields, (TaskPrimitive.EXTRACT,),
             "layout drift; also the behavioural invariance test"),
    Operator("whitespace_drift", _whitespace_drift, (), "format drift"),
    Operator("abbreviation", _abbreviate, (), "abbreviation expansion/contraction"),
    Operator("typo", _typo, (TaskPrimitive.CLASSIFY, TaskPrimitive.ROUTE),
             "keyboard transposition"),
)


def operators_for(primitive: TaskPrimitive) -> list[Operator]:
    return [op for op in DOCUMENT_OPERATORS if op.applies_to(primitive)]


def compose(operators: Sequence[Operator], max_depth: int = MAX_DEPTH) -> Operator:
    """Compose operators. Admissible by closure — but bounded in depth.

    The monoid guarantees the composition is still label-preserving. It does not
    guarantee the result is *realistic*, which is what the depth cap is for.
    """
    if len(operators) > max_depth:
        raise ValueError(
            f"composition depth {len(operators)} exceeds {max_depth}: past this "
            "the output drifts off the manifold of realistic inputs, and a "
            "document transformed five times is not one anyone will ever scan"
        )

    def applied(text: str, rng: random.Random) -> str:
        for op in operators:
            text = op.apply(text, rng)
        return text

    return Operator(
        name=" o ".join(op.name for op in operators),
        apply=applied,
        note=f"composition of depth {len(operators)}",
    )


# =========================================================================== #
# §4 — pseudonymisation is privacy mitigation AND augmentation
# =========================================================================== #
#: Replacement pools. Consistent within a document, varied across variants.
_SURNAMES = ("Sharma", "Okafor", "Nakamura", "Kowalski", "Mbeki", "Duarte",
             "Andersen", "Rahman", "Silva", "Novak")
_GIVEN = ("Priya", "Chidi", "Yuki", "Marta", "Thabo", "Ines", "Lars", "Aisha",
          "Bruno", "Eva")


@dataclass
class Pseudonymiser:
    """Consistent entity replacement, with the label updated in step (§4).

    **The hard tension this resolves:** for extraction tasks the PII *is* the
    label. You cannot redact patient identifiers from a patient-identifier
    extractor's training data — redaction destroys the task.

    Consistent pseudonymisation resolves it, and does two jobs at once:

    * it is **label-preserving under relabelling** — only surface tokens change;
    * it is **an augmentation operator** — one real document yields ``m``
      variants, each correctly labelled.

    So privacy mitigation and data amplification are the same operation, and it
    directly lowers verbatim-extraction rate because no single real identity
    appears often enough to be memorised. **This should be the default for every
    extraction build carrying PII, not an option.**
    """

    seed: int = 0
    mapping: dict[str, str] = field(default_factory=dict)

    def name_for(self, original: str, variant: int = 0) -> str:
        key = f"{original}|{variant}"
        if key not in self.mapping:
            rng = random.Random(f"{self.seed}:{key}")
            self.mapping[key] = f"{rng.choice(_GIVEN)} {rng.choice(_SURNAMES)}"
        return self.mapping[key]

    def apply(self, text: str, label: dict[str, Any], entities: Sequence[str],
              variant: int = 0) -> tuple[str, dict[str, Any]]:
        """Replace ``entities`` consistently, updating the label in step."""
        out_text, out_label = text, dict(label)
        for entity in entities:
            if not entity:
                continue
            replacement = self.name_for(entity, variant)
            out_text = out_text.replace(entity, replacement)
            for key, value in out_label.items():
                if isinstance(value, str) and entity in value:
                    out_label[key] = value.replace(entity, replacement)
        return out_text, out_label


# =========================================================================== #
# Generating a batch, with provenance
# =========================================================================== #
@dataclass(frozen=True)
class AugmentedExample:
    """One generated example, carrying provenance back to its real source.

    ``source_id`` is required rather than optional: §14 needs it to check
    contamination at the *source-document* level, and §21 needs it to cap
    per-source multiplicity. An augmented variant of a held-out document leaks
    even though the two examples differ byte for byte, and only provenance
    catches that.
    """

    text: str
    label: Any
    source_id: str
    operation: Operation = Operation.AUGMENTATION
    operator: str = ""
    depth: int = 1

    @property
    def label_is_real(self) -> bool:
        return self.operation.label_is_real


def source_id_of(text: str) -> str:
    """A stable identifier for an originating real document."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


def augment(
    seeds: Sequence[tuple[str, Any]],
    primitive: TaskPrimitive,
    *,
    per_seed: int = 4,
    max_depth: int = MAX_DEPTH,
    max_multiplicity: int = MAX_MULTIPLICITY,
    seed: int = 0,
) -> list[AugmentedExample]:
    """Generate label-preserving variations of real inputs.

    No teacher, no GPU, and the label is real by construction. ``per_seed`` is
    clamped to ``max_multiplicity`` because twenty transforms of one document
    are twenty near-copies of its content, which is a memorisation risk rather
    than data (§21).
    """
    if per_seed > max_multiplicity:
        logger.info(
            "augment: capping %d variants per seed at %d — duplication count "
            "drives verbatim extractability, and the weights ship to devices",
            per_seed, max_multiplicity,
        )
        per_seed = max_multiplicity

    ops = operators_for(primitive)
    if not ops:
        return []
    rng = random.Random(seed)
    out: list[AugmentedExample] = []

    for text, label in seeds:
        sid = source_id_of(text)
        for i in range(per_seed):
            depth = rng.randint(1, min(max_depth, len(ops)))
            chosen = rng.sample(ops, depth)
            operator = compose(chosen, max_depth)
            out.append(AugmentedExample(
                text=operator.apply(text, rng), label=label, source_id=sid,
                operation=Operation.AUGMENTATION, operator=operator.name,
                depth=depth,
            ))
    return out


def multiplicity(examples: Iterable[AugmentedExample]) -> Counter:
    """How many examples each real source produced (§21)."""
    return Counter(e.source_id for e in examples)


def multiplicity_report(examples: Sequence[AugmentedExample],
                        max_multiplicity: int = MAX_MULTIPLICITY) -> dict[str, Any]:
    """``max_i k_i`` — a privacy predictor, reported before the audit runs.

    A build at ``k_max = 40`` should expect a worse extraction rate than one at
    ``k_max = 5``. Knowing that in advance turns a surprise into a prediction.
    """
    counts = multiplicity(examples)
    worst = max(counts.values(), default=0)
    return {
        "sources": len(counts),
        "examples": len(examples),
        "max_multiplicity": worst,
        "mean_multiplicity": round(len(examples) / len(counts), 2) if counts else 0.0,
        "within_cap": worst <= max_multiplicity,
        "why": (
            "memorisation probability rises sharply with duplication count, and "
            "Majestic ships weights to devices — the attacker has white-box "
            "access, which is the strongest setting"
        ),
    }


def cost_estimate(examples: Sequence[AugmentedExample],
                  usd_per_million_tokens: float = 0.4) -> dict[str, Any]:
    """§1, §15 — what a batch actually costs to generate.

    Augmentation requires no teacher inference, so an extraction-dominated build
    costs approximately nothing. The dollar figures quoted for generation apply
    to synthesis-dominated primitives.
    """
    tokens = sum(e.operation.teacher_tokens_per_example for e in examples)
    return {
        "examples": len(examples),
        "teacher_tokens": tokens,
        "usd": round(tokens / 1_000_000 * usd_per_million_tokens, 4),
        "free_share": round(
            sum(1 for e in examples
                if e.operation.teacher_tokens_per_example == 0) / len(examples), 4
        ) if examples else 0.0,
    }


def spot_check_sample(examples: Sequence[AugmentedExample],
                      n: int = SPOT_CHECK_SAMPLES,
                      seed: int = 0) -> list[AugmentedExample]:
    """The sample a human must actually look at (§3).

    Realism cannot be checked by distance to the seeds — that would reject
    exactly the hard cases augmentation exists to produce. It is enforced by
    parameterising operators from domain knowledge, and then verified by
    somebody looking at a hundred of them for ten minutes.
    """
    rng = random.Random(seed)
    return rng.sample(list(examples), min(n, len(examples)))


__all__ = [
    "DOCUMENT_OPERATORS", "DOMINANT_OPERATION", "MAX_DEPTH", "MAX_MULTIPLICITY",
    "SPOT_CHECK_SAMPLES",
    "AugmentedExample", "Operation", "Operator", "Pseudonymiser",
    "augment", "compose", "cost_estimate", "multiplicity", "multiplicity_report",
    "operators_for", "source_id_of", "spot_check_sample",
]
