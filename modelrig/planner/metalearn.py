"""Build-outcome meta-learning with empirical-Bayes shrinkage (§5).

    Q = (1 - gamma) * Q_prior + gamma * Q_learn,      gamma = |H| / (|H| + h0)

``h0`` is the prior's effective sample size — the amount of history at which the
rule-based prior and the learned model are trusted equally. Blending this way is
what makes the Planner improve **monotonically** instead of oscillating while
history is thin: with no history ``gamma = 0`` and the learner is inert; it earns
influence in proportion to evidence.

The learner itself is deliberately simple — ridge regression on a small,
hand-chosen feature vector. At the scale where this matters (|H| in the low
hundreds) anything with more capacity overfits, and a linear model has the
practical virtue that its coefficients can be read and argued with.

**Claim C6 is exactly the question of whether ``Q_learn`` beats ``Q_prior`` at
|H| = 200.** Until that is measured, this module's contribution to a live plan is
bounded by ``gamma``, which is the point.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from majestic.logging_utils import get_logger

logger = get_logger(__name__)

#: History size at which prior and learner are weighted equally.
# HYPOTHESIS — h0 is a judgement about how much the crude prior is worth.
# 50 says "fifty observed builds outweigh the hand-written rule".
H0 = 50

#: Below this, the learner is not fitted at all (§17 step 9).
MIN_HISTORY = 50


@dataclass(frozen=True)
class Observation:
    """One finished build: what was asked, what was planned, what happened."""

    spec_hash: str
    plan_hash: str
    features: tuple[float, ...]
    outcome: float          # y in [0,1]: the observed gate margin
    primitive: str = ""
    passed: bool = False


def features_of(
    *, params_b: float, seed_ratio: float, bits_effective: float,
    distil_mode: str, context: int, quality_gate: float,
) -> tuple[float, ...]:
    """``phi(s) x psi(pi)`` — the feature vector, kept small on purpose.

    Six features against a hundred-odd observations is already a generous
    parameter budget. Adding more would fit noise.
    """
    return (
        1.0,                                        # bias
        math.log1p(params_b),                       # scale, compressed
        min(seed_ratio, 4.0),                       # data sufficiency headroom
        bits_effective / 16.0,                      # compression aggressiveness
        1.0 if distil_mode == "logit_kd" else (0.5 if distil_mode == "sequence_kd" else 0.0),
        quality_gate,                               # how hard the bar is
    )


class OutcomePredictor:
    """Ridge regression over build history, blended with the prior by shrinkage."""

    def __init__(self, h0: int = H0, min_history: int = MIN_HISTORY, ridge: float = 1.0) -> None:
        self.h0 = h0
        self.min_history = min_history
        self.ridge = ridge
        self.history: list[Observation] = []
        self._weights: np.ndarray | None = None
        self._fitted_at = 0

    # -- history ---------------------------------------------------------- #
    def record(self, obs: Observation) -> None:
        self.history.append(obs)
        self._weights = None      # invalidate; refit lazily

    def __len__(self) -> int:
        return len(self.history)

    @property
    def gamma(self) -> float:
        """``gamma = |H| / (|H| + h0)`` — how much the learner is trusted."""
        n = len(self.history)
        return n / (n + self.h0) if n else 0.0

    @property
    def active(self) -> bool:
        """The learner is inert until there is enough history to fit it."""
        return len(self.history) >= self.min_history

    # -- fitting ---------------------------------------------------------- #
    def fit(self) -> bool:
        """Fit ridge regression. Returns False while history is too thin."""
        if not self.active:
            return False
        X = np.asarray([o.features for o in self.history], dtype=np.float64)
        y = np.asarray([o.outcome for o in self.history], dtype=np.float64)
        d = X.shape[1]
        # Closed form: (X'X + lambda I)^-1 X'y. Ridge keeps this invertible even
        # when features are collinear, which they will be with a small catalogue.
        gram = X.T @ X + self.ridge * np.eye(d)
        try:
            self._weights = np.linalg.solve(gram, X.T @ y)
        except np.linalg.LinAlgError:
            logger.warning("metalearn: gram matrix singular; learner stays inert")
            self._weights = None
            return False
        self._fitted_at = len(self.history)
        logger.info("metalearn: fitted on %d observations (gamma=%.2f)",
                    self._fitted_at, self.gamma)
        return True

    def _learned(self, features: Sequence[float]) -> float | None:
        if self._weights is None or self._fitted_at != len(self.history):
            if not self.fit():
                return None
        assert self._weights is not None
        raw = float(np.asarray(features, dtype=np.float64) @ self._weights)
        return max(0.0, min(1.0, raw))

    # -- the blended prediction ------------------------------------------- #
    def predict(self, prior: float, features: Sequence[float]) -> float:
        """``Q = (1-gamma) Q_prior + gamma Q_learn``.

        Falls back to the prior entirely when the learner cannot be fitted, so a
        cold start degrades to the rule-based path rather than to zero.
        """
        learned = self._learned(features)
        if learned is None:
            return prior
        g = self.gamma
        return (1.0 - g) * prior + g * learned

    def explain(self, prior: float, features: Sequence[float]) -> dict[str, Any]:
        """Both halves and the weight between them, for auditability."""
        learned = self._learned(features)
        return {
            "prior": round(prior, 4),
            "learned": round(learned, 4) if learned is not None else None,
            "gamma": round(self.gamma, 4),
            "history": len(self.history),
            "blended": round(self.predict(prior, features), 4),
            "active": self.active,
        }

    # -- persistence ------------------------------------------------------- #
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps([
            {"spec_hash": o.spec_hash, "plan_hash": o.plan_hash,
             "features": list(o.features), "outcome": o.outcome,
             "primitive": o.primitive, "passed": o.passed}
            for o in self.history
        ], indent=2), encoding="utf-8")
        return p

    def load(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            return
        self.history = [
            Observation(
                spec_hash=r["spec_hash"], plan_hash=r["plan_hash"],
                features=tuple(r["features"]), outcome=float(r["outcome"]),
                primitive=r.get("primitive", ""), passed=bool(r.get("passed", False)),
            )
            for r in json.loads(p.read_text(encoding="utf-8"))
        ]
        self._weights = None


# --------------------------------------------------------------------------- #
@dataclass
class Precedent:
    """A past build retrievable by spec shape (§7 lines 1-3)."""

    shape: tuple
    plan: Any
    passed: bool
    margin: float = 0.0


class PrecedentIndex:
    """Nearest-precedent retrieval for the deterministic warm start.

    §11 open question 3: precedent reuse can lock in early mistakes, because
    nearest-neighbour retrieval propagates whatever the first builds did. The
    mitigation is an **exploration floor** — occasionally ignore the precedent —
    which is an explore/exploit problem this design does not claim to solve. The
    floor is exposed here rather than hidden so it can be measured.
    """

    def __init__(self, exploration_rate: float = 0.0) -> None:
        self.records: list[Precedent] = []
        self.exploration_rate = exploration_rate
        self._counter = 0

    def add(self, shape: tuple, plan: Any, passed: bool, margin: float = 0.0) -> None:
        self.records.append(Precedent(shape, plan, passed, margin))

    def nearest(self, shape: tuple) -> Precedent | None:
        """Exact shape match among passing builds, most recent first.

        Exact matching rather than embedding similarity, deliberately: it keeps
        the common path fully deterministic, which is what makes plans cacheable
        and auditable. Similarity search belongs with the learned predictor.
        """
        self._counter += 1
        if self.exploration_rate > 0 and self._counter % max(
            1, int(round(1 / self.exploration_rate))
        ) == 0:
            return None      # deterministic exploration: skip the precedent
        for record in reversed(self.records):
            if record.shape == shape and record.passed:
                return record
        return None
