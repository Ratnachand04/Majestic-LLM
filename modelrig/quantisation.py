"""Post-training quantisation — where a good fine-tune quietly dies (A-05).

    FP16 model -> calibration set -> quantiser -> INT4 -> MANDATORY re-eval

Three things here are the difference between a working product and a silent
failure:

**The calibration set is drawn from the CUSTOMER's task distribution**, never
from generic web text. Off-distribution calibration is the most common cause of a
fine-tune evaporating at 4-bit, and it is the one modification that matters most.

**Re-evaluation is mandatory, not optional.** A quantised model can hold
identical aggregate accuracy while changing a large fraction of individual
answers (2407.09141). Aggregate parity is an illusion, so the answer-flip rate is
computed and reported.

**Confidence must be recalibrated.** Quantisation shifts the probability
distribution, so a temperature fitted before compression no longer holds — and
routing, abstention and human review all depend on that confidence being
trustworthy.

On failure the escalation ladder is explicit: try a higher-fidelity quantiser
(SpQR preserves outliers), then fall back to a larger base or a wider bit width.
Outlier features above roughly 6.7B parameters are the specific mechanism of
quantisation failure (LLM.int8()); below that scale the failure is subtler and
easier to miss, which is exactly why the gate is unconditional.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

from majestic.logging_utils import get_logger

logger = get_logger(__name__)

Example = tuple[str, str]

#: Parameter count above which outlier features dominate quantisation error.
OUTLIER_THRESHOLD_B = 6.7
#: Default number of representative samples in a calibration set.
DEFAULT_CALIBRATION_SIZE = 256
#: Escalation ladder applied when the post-quantisation gate fails.
ESCALATION_LADDER: tuple[str, ...] = ("awq", "gptq", "spqr", "int8", "larger_base")


@dataclass
class CalibrationSet:
    """Representative samples used to fit the quantiser."""

    samples: list[str] = field(default_factory=list)
    source: str = "customer_task_distribution"
    on_distribution: bool = True

    def __len__(self) -> int:
        return len(self.samples)


def build_calibration_set(
    train: Sequence[Example],
    size: int = DEFAULT_CALIBRATION_SIZE,
    seed: int = 0,
    generic_fallback: Sequence[str] = (),
) -> CalibrationSet:
    """Draw a calibration set from the customer's own task distribution.

    Sampling is stratified by label so every class the model must serve is
    represented — an unstratified draw can leave a rare-but-important class out
    of calibration entirely and is how a class silently degrades at 4-bit.

    Falling back to generic text is supported but flagged ``on_distribution=False``
    so the caller can surface the risk rather than absorb it silently.
    """
    if not train:
        return CalibrationSet(
            samples=list(generic_fallback)[:size],
            source="generic_text",
            on_distribution=False,
        )

    by_label: dict[str, list[str]] = {}
    for text, label in train:
        by_label.setdefault(str(label), []).append(text)

    rng = random.Random(seed)
    per_label = max(1, size // max(len(by_label), 1))
    samples: list[str] = []
    for label in sorted(by_label):
        pool = by_label[label]
        rng.shuffle(pool)
        samples.extend(pool[:per_label])

    # Top up from the whole pool if stratification under-fills the quota.
    if len(samples) < size:
        rest = [t for t, _ in train if t not in set(samples)]
        rng.shuffle(rest)
        samples.extend(rest[: size - len(samples)])

    return CalibrationSet(samples=samples[:size])


# --------------------------------------------------------------------------- #
@dataclass
class TemperatureScaler:
    """Post-quantisation confidence recalibration (Guo et al. 1706.04599).

    A single temperature fitted on held-out data. Quantisation shifts the
    probability distribution, so this must be refitted AFTER compression — a
    temperature fitted on the FP16 model is not valid for the INT4 one.

    GAP-03: calibration below 2B parameters is unvalidated territory, so the
    fitted temperature is diagnostic rather than trustworthy at that scale.
    """

    temperature: float = 1.0
    fitted: bool = False
    n_samples: int = 0

    def fit(
        self,
        confidences: Sequence[float],
        correct: Sequence[bool],
        grid: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0),
    ) -> TemperatureScaler:
        """Grid-search the temperature that minimises calibration error."""
        if not confidences:
            return self
        best_t, best_err = 1.0, float("inf")
        for t in grid:
            scaled = [self._apply(c, t) for c in confidences]
            err = _calibration_error(scaled, correct)
            if err < best_err:
                best_t, best_err = t, err
        self.temperature = best_t
        self.fitted = True
        self.n_samples = len(confidences)
        logger.info(
            "recalibrated confidence after quantisation: T=%.2f (error %.4f)",
            best_t, best_err,
        )
        return self

    @staticmethod
    def _apply(confidence: float, temperature: float) -> float:
        """Temperature-scale a probability through the logit domain."""
        p = min(max(float(confidence), 1e-6), 1 - 1e-6)
        logit = math.log(p / (1 - p)) / max(temperature, 1e-6)
        return 1.0 / (1.0 + math.exp(-logit))

    def __call__(self, confidence: float) -> float:
        return self._apply(confidence, self.temperature)


def _calibration_error(confidences: Sequence[float], correct: Sequence[bool],
                       bins: int = 10) -> float:
    if not confidences:
        return 0.0
    total, error = len(confidences), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if (lo < c <= hi) or (b == 0 and c <= lo)]
        if not idx:
            continue
        avg_conf = sum(confidences[i] for i in idx) / len(idx)
        acc = sum(1 for i in idx if correct[i]) / len(idx)
        error += (len(idx) / total) * abs(avg_conf - acc)
    return error


# --------------------------------------------------------------------------- #
@dataclass
class QuantisationOutcome:
    """What compression did, and whether it is allowed through."""

    quantiser: str
    bit_width: str
    passed: bool
    answer_flip_rate: float
    flip_bound: float
    calibration_on_distribution: bool
    calibration_size: int
    temperature: float = 1.0
    escalation: str | None = None
    reasons: list[str] = field(default_factory=list)
    outlier_risk: bool = False


def next_escalation(current_quantiser: str, params_b: float) -> str | None:
    """The next rung when post-quantisation evaluation fails (A-05).

    SpQR is reached sooner for large models because outlier features above
    ~6.7B are the specific failure mechanism there.
    """
    ladder = list(ESCALATION_LADDER)
    if current_quantiser not in ladder:
        return ladder[0]

    remaining = ladder[ladder.index(current_quantiser) + 1:]
    if not remaining:
        return None
    # Above the outlier threshold, jump straight to the outlier-preserving
    # method rather than walking rungs that fail for the same reason.
    if params_b >= OUTLIER_THRESHOLD_B and "spqr" in remaining:
        return "spqr"
    return remaining[0]


def evaluate_quantisation(
    *,
    quantiser: str,
    bit_width: str,
    params_b: float,
    reference_predictions: Sequence[str],
    quantised_predictions: Sequence[str],
    calibration: CalibrationSet,
    flip_bound: float = 0.10,
    confidences: Sequence[float] | None = None,
    correct: Sequence[bool] | None = None,
    scorer: Callable[[Sequence[str]], float] | None = None,
    quality_gate: float | None = None,
) -> QuantisationOutcome:
    """Run the mandatory post-quantisation gate and recalibrate confidence."""
    n = min(len(reference_predictions), len(quantised_predictions))
    flips = sum(
        1 for i in range(n) if reference_predictions[i] != quantised_predictions[i]
    )
    flip_rate = round(flips / n, 4) if n else 0.0

    reasons: list[str] = []
    if flip_rate > flip_bound:
        reasons.append(
            f"answer-flip rate {flip_rate:.3f} exceeds bound {flip_bound:.3f}: "
            "aggregate parity is an illusion when individual answers change"
        )
    if not calibration.on_distribution:
        reasons.append(
            "calibration used generic text rather than the customer's task "
            "distribution — the most common cause of a fine-tune evaporating at 4-bit"
        )
    if quality_gate is not None and scorer is not None:
        score = scorer(quantised_predictions)
        if score < quality_gate:
            reasons.append(
                f"post-quantisation score {score:.3f} fell below the gate {quality_gate:.2f}"
            )

    scaler = TemperatureScaler()
    if confidences is not None and correct is not None:
        scaler.fit(confidences, correct)

    outlier_risk = params_b >= OUTLIER_THRESHOLD_B and bit_width == "int4"
    if outlier_risk:
        reasons.append(
            f"base is {params_b}B: above ~{OUTLIER_THRESHOLD_B}B outlier features "
            "drive quantisation failure, so int4 needs outlier-preserving methods"
        )

    passed = not reasons
    outcome = QuantisationOutcome(
        quantiser=quantiser,
        bit_width=bit_width,
        passed=passed,
        answer_flip_rate=flip_rate,
        flip_bound=flip_bound,
        calibration_on_distribution=calibration.on_distribution,
        calibration_size=len(calibration),
        temperature=scaler.temperature,
        escalation=None if passed else next_escalation(quantiser, params_b),
        reasons=reasons,
        outlier_risk=outlier_risk,
    )
    logger.info(
        "quantisation %s/%s: %s (flip %.3f, calib %d on-dist=%s)",
        quantiser, bit_width, "PASS" if passed else "FAIL",
        flip_rate, len(calibration), calibration.on_distribution,
    )
    return outcome
