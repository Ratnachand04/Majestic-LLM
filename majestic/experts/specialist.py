"""A small specialist model expert (the kind ModelRig builds).

When a ``model_id`` is given and ``transformers`` is installed, it loads a small
local text-classification pipeline. Otherwise it falls back to a deterministic,
dependency-free heuristic classifier so the pipeline works offline and in tests.
"""
from __future__ import annotations

from typing import Any

from majestic.experts.base import Expert
from majestic.logging_utils import get_logger

logger = get_logger(__name__)

_POSITIVE = {
    "good", "great", "excellent", "love", "loved", "wonderful", "amazing",
    "happy", "best", "fantastic", "nice", "awesome", "positive", "like",
}
_NEGATIVE = {
    "bad", "terrible", "awful", "hate", "hated", "worst", "horrible", "poor",
    "sad", "angry", "negative", "broken", "useless", "disappointing",
}


class SpecialistExpert(Expert):
    """Text classifier. Real pipeline when available; heuristic otherwise."""

    name = "specialist"
    capabilities = ("classify", "extract", "summarize")

    def __init__(
        self,
        task: str = "sentiment",
        model_id: str | None = None,
        labels: tuple[str, ...] = ("positive", "negative"),
    ) -> None:
        self.task = task
        self.model_id = model_id
        self.labels = labels
        self._pipeline: Any | None = None
        if model_id:
            try:
                self._load(model_id)
            except Exception as exc:  # noqa: BLE001 - degrade to heuristic
                logger.warning("specialist model load failed (%s); using heuristic", exc)

    def _load(self, model_id: str) -> None:
        from transformers import pipeline

        logger.info("Loading specialist pipeline: %s", model_id)
        self._pipeline = pipeline("text-classification", model=model_id)

    def run(self, **kwargs: Any) -> dict[str, Any]:
        text = str(kwargs.get("text", ""))
        if self._pipeline is not None:
            out = self._pipeline(text)[0]
            return {"label": str(out["label"]), "score": float(out["score"]), "backend": "model"}
        return self._heuristic(text)

    def _heuristic(self, text: str) -> dict[str, Any]:
        tokens = {t.strip(".,!?;:").lower() for t in text.split()}
        pos = len(tokens & _POSITIVE)
        neg = len(tokens & _NEGATIVE)
        if pos == neg:
            label, score = "neutral", 0.5
        elif pos > neg:
            label, score = "positive", pos / (pos + neg)
        else:
            label, score = "negative", neg / (pos + neg)
        return {"label": label, "score": round(score, 3), "backend": "heuristic"}

    def estimate_cost(self, **kwargs: Any) -> float:
        return 1.0
