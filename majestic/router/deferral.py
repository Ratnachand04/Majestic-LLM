"""Learned deferral: decide locally, escalate only the hard tail (A-08, B-10).

    request -> deferral rule -> [local cartridge | cloud teacher] -> response

Majestic inverts the usual cascade economics: **tier 1 runs on the user's own
device**, so its marginal cost is zero and no data leaves the building. The
privacy argument is as strong as the cost argument.

Two rules are non-negotiable:

**Offline mode hard-locks the threshold.** When the spec demands offline
operation, escalation is architecturally disabled — not merely discouraged. A
customer who bought offline gets offline.

**Every escalation is logged as a hard case.** The queries the local model could
not handle become the next generation's training data; the router is what closes
the flywheel.

The rule is LEARNED, not a softmax threshold. Confidence-based deferral is
provably suboptimal for specialist models under distribution shift (2307.02764)
— precisely Majestic's setting — and the deferral rule's accuracy caps
whole-system accuracy. Following Hybrid LLM (2404.14618), it routes on predicted
quality GAP rather than absolute difficulty, and exposes a tunable cost-quality
knob the customer sees as a business control: how often may this call the cloud?

GAP-03 (open): the confidence feature feeding this rule is unvalidated below 2B
parameters, where Majestic operates. Miscalibrated confidence breaks the router,
the abstention behaviour and the human-review workflow simultaneously.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from majestic.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DeferralDecision:
    """Why this request was answered locally or escalated."""

    escalate: bool
    predicted_quality_gap: float
    confidence: float
    reason: str
    locked_offline: bool = False


@dataclass
class HardCase:
    """One escalated query — the raw material of the flywheel."""

    query: str
    confidence: float
    predicted_quality_gap: float
    capability: str = "unknown"
    resolved_by: str = "cloud"
    correction: str | None = None


def _sigmoid(x: float) -> float:
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


class LearnedDeferralRule:
    """Logistic model over routing features, fitted from observed outcomes.

    Features are deliberately cheap to compute on-device:

    ``confidence``      the cartridge's own confidence (unreliable below 2B)
    ``length``          normalised input length; long inputs shift distribution
    ``oov_ratio``       share of tokens unseen in the training vocabulary — the
                        most direct distribution-shift signal available locally
    ``retrieval_score`` best retrieval similarity, when the graph retrieves

    The model predicts the QUALITY GAP: how much better the cloud teacher would
    be. Escalate when the predicted gap exceeds ``gap_threshold``.
    """

    FEATURES = ("bias", "confidence", "length", "oov_ratio", "retrieval_score")

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        gap_threshold: float = 0.35,
        learning_rate: float = 0.2,
    ) -> None:
        # Sensible priors: low confidence, long inputs, high OOV and weak
        # retrieval all raise the predicted gap.
        self.weights = weights or {
            "bias": -0.4, "confidence": -2.2, "length": 0.6,
            "oov_ratio": 2.0, "retrieval_score": -1.0,
        }
        self.gap_threshold = gap_threshold
        self.learning_rate = learning_rate
        self.n_updates = 0

    def features(
        self,
        query: str,
        confidence: float,
        vocabulary: Sequence[str] = (),
        retrieval_score: float = 0.0,
        max_length: int = 512,
    ) -> dict[str, float]:
        tokens = query.lower().split()
        vocab = set(vocabulary)
        oov = (
            sum(1 for t in tokens if t not in vocab) / len(tokens)
            if tokens and vocab else 0.0
        )
        return {
            "bias": 1.0,
            "confidence": float(confidence),
            "length": min(len(tokens) / max_length, 1.0),
            "oov_ratio": oov,
            "retrieval_score": float(retrieval_score),
        }

    def predict_gap(self, features: dict[str, float]) -> float:
        """Predicted quality gap in [0, 1]: how much better the cloud would be."""
        z = sum(self.weights.get(k, 0.0) * features.get(k, 0.0) for k in self.FEATURES)
        return _sigmoid(z)

    def update(self, features: dict[str, float], cloud_was_better: bool) -> None:
        """One online logistic step from an observed escalation outcome."""
        predicted = self.predict_gap(features)
        error = (1.0 if cloud_was_better else 0.0) - predicted
        for k in self.FEATURES:
            self.weights[k] = self.weights.get(k, 0.0) + (
                self.learning_rate * error * features.get(k, 0.0)
            )
        self.n_updates += 1

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {"weights": self.weights, "gap_threshold": self.gap_threshold,
                 "n_updates": self.n_updates},
                indent=2,
            ),
            encoding="utf-8",
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> LearnedDeferralRule:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rule = cls(weights=data["weights"], gap_threshold=data.get("gap_threshold", 0.35))
        rule.n_updates = int(data.get("n_updates", 0))
        return rule


class DeferralRouter:
    """Applies the learned rule, honouring the offline hard-lock."""

    def __init__(
        self,
        rule: LearnedDeferralRule | None = None,
        offline_required: bool = False,
        vocabulary: Sequence[str] = (),
        max_escalation_rate: float | None = None,
    ) -> None:
        self.rule = rule or LearnedDeferralRule()
        self.offline_required = offline_required
        self.vocabulary = list(vocabulary)
        #: The customer-facing cost-quality knob: how often may this call the cloud?
        self.max_escalation_rate = max_escalation_rate
        self.hard_cases: list[HardCase] = []
        self._decisions = 0
        self._escalations = 0

    @property
    def escalation_rate(self) -> float:
        return round(self._escalations / self._decisions, 4) if self._decisions else 0.0

    def decide(
        self,
        query: str,
        confidence: float,
        retrieval_score: float = 0.0,
        capability: str = "unknown",
    ) -> DeferralDecision:
        """Route one request, logging every escalation as a hard case."""
        self._decisions += 1
        feats = self.rule.features(query, confidence, self.vocabulary, retrieval_score)
        gap = self.rule.predict_gap(feats)

        if self.offline_required:
            # Architecturally disabled, not merely discouraged.
            return DeferralDecision(
                escalate=False, predicted_quality_gap=gap, confidence=confidence,
                reason="offline_required: escalation is architecturally disabled",
                locked_offline=True,
            )

        if (
            self.max_escalation_rate is not None
            and self.escalation_rate >= self.max_escalation_rate
        ):
            return DeferralDecision(
                escalate=False, predicted_quality_gap=gap, confidence=confidence,
                reason=(
                    f"escalation budget spent ({self.escalation_rate:.2f} >= "
                    f"{self.max_escalation_rate:.2f})"
                ),
            )

        escalate = gap > self.rule.gap_threshold
        if escalate:
            self._escalations += 1
            self.hard_cases.append(
                HardCase(query=query, confidence=confidence,
                         predicted_quality_gap=gap, capability=capability)
            )
            logger.info("deferral: escalating (gap %.2f > %.2f)", gap, self.rule.gap_threshold)
        return DeferralDecision(
            escalate=escalate,
            predicted_quality_gap=round(gap, 4),
            confidence=confidence,
            reason=(
                f"predicted quality gap {gap:.2f} "
                f"{'>' if escalate else '<='} threshold {self.rule.gap_threshold:.2f}"
            ),
        )

    def observe(self, query: str, confidence: float, cloud_was_better: bool,
                retrieval_score: float = 0.0) -> None:
        """Teach the rule from an observed outcome (the learned part)."""
        feats = self.rule.features(query, confidence, self.vocabulary, retrieval_score)
        self.rule.update(feats, cloud_was_better)

    def drain_hard_cases(self) -> list[HardCase]:
        """Hand escalated cases to the flywheel and clear the local queue."""
        cases, self.hard_cases = self.hard_cases, []
        return cases


@dataclass
class RouterTelemetry:
    """Aggregate view the customer sees as a business control."""

    decisions: int = 0
    escalations: int = 0
    hard_cases: list[HardCase] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "escalations": self.escalations,
            "escalation_rate": round(self.escalations / self.decisions, 4)
            if self.decisions else 0.0,
            "hard_cases": len(self.hard_cases),
        }
