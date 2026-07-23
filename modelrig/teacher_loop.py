"""Teacher -> factory loop: the idea that unifies the whole system.

The Majestic core is the TEACHER. It distills capabilities into small device
specialists via the factory. Deployed specialists escalate the hard cases they
cannot handle; that telemetry improves routing and mints new adapters.
"""
from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from majestic.logging_utils import get_logger
from modelrig.buildspec import BuildSpec, DeviceProfile, TrainingMethod
from modelrig.factory import BuildResult, Factory
from modelrig.feasibility import HeuristicFeasibilityEngine, YamlDeviceProfiler

logger = get_logger(__name__)

# Anchor vocabularies used to synthesize (curate) tiny training sets. A real
# teacher core would generate and label these examples; this deterministic
# generator stands in so the loop runs fully offline.
_ANCHORS = {
    "sentiment": {
        "positive": ["good", "great", "love", "happy", "nice", "wonderful"],
        "negative": ["bad", "hate", "awful", "sad", "terrible", "poor"],
    }
}


class TeacherFactoryLoop(ABC):
    @abstractmethod
    def distill(self, capability: str, device: str) -> BuildSpec:
        """Turn a slice of teacher capability into a build for a device."""
        raise NotImplementedError

    @abstractmethod
    def ingest_telemetry(self, events: list[Any]) -> None:
        """Feed escalated hard cases back to improve routing / mint adapters."""
        raise NotImplementedError


class DistillationLoop(TeacherFactoryLoop):
    """Concrete loop: curate data -> BuildSpec -> feasibility-checked factory build.

    Also accumulates escalation telemetry and surfaces capabilities that escalate
    often enough to warrant a new specialist (a simple routing-update signal).
    """

    def __init__(
        self,
        factory: Optional[Factory] = None,
        profiler: Optional[YamlDeviceProfiler] = None,
        feasibility: Optional[HeuristicFeasibilityEngine] = None,
        data_dir: str | Path = "./registry/datasets",
        n_examples: int = 40,
    ) -> None:
        self.factory = factory or Factory()
        self.profiler = profiler or YamlDeviceProfiler()
        self.feasibility = feasibility or HeuristicFeasibilityEngine(profiler=self.profiler)
        self.data_dir = Path(data_dir)
        self.n_examples = n_examples
        self.telemetry: list[dict[str, Any]] = []
        self.escalation_counts: dict[str, int] = {}

    # -- data curation --------------------------------------------------- #
    def _curate(self, capability: str, seed: int) -> str:
        """Synthesize a small labeled dataset; return its file path."""
        anchors = _ANCHORS.get(capability, _ANCHORS["sentiment"])
        rng = random.Random(seed)
        labels = list(anchors)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        out = self.data_dir / f"{capability}.jsonl"
        lines = []
        for _ in range(self.n_examples):
            label = rng.choice(labels)
            words = rng.sample(anchors[label], k=3)
            lines.append(json.dumps({"text": " ".join(words), "label": label}))
        out.write_text("\n".join(lines), encoding="utf-8")
        logger.info("curated %d examples for %r -> %s", self.n_examples, capability, out)
        return str(out)

    # -- distill --------------------------------------------------------- #
    def _resolve_device(self, device: str | DeviceProfile) -> DeviceProfile:
        return device if isinstance(device, DeviceProfile) else self.profiler.profile(device)

    def distill(
        self, capability: str, device: str | DeviceProfile, seed: int = 0
    ) -> BuildSpec:
        dev = self._resolve_device(device)
        dataset = self._curate(capability, seed)
        spec = BuildSpec(
            task=capability,
            base_model="centroid",
            method=TrainingMethod.CENTROID,
            quantization="int8",
            runtime="npz",
            device=dev,
            dataset=dataset,
            target_score=0.7,
            seed=seed,
        )
        verdict = self.feasibility.evaluate(spec)
        spec.extras["feasibility"] = {
            "feasible": verdict.feasible,
            "ram_mb": verdict.estimate.ram_mb,
            "reasons": verdict.reasons,
        }
        if not verdict.feasible:
            raise ValueError(
                f"distilled spec infeasible on {dev.name}: {verdict.reasons}"
            )
        return spec

    def distill_and_build(
        self, capability: str, device: str | DeviceProfile, seed: int = 0
    ) -> BuildResult:
        """Convenience: distill a spec and run it through the factory."""
        spec = self.distill(capability, device, seed)
        return self.factory.build(spec)

    # -- telemetry ------------------------------------------------------- #
    def ingest_telemetry(self, events: list[Any]) -> None:
        for event in events:
            record = event if isinstance(event, dict) else {"capability": str(event)}
            self.telemetry.append(record)
            cap = record.get("capability", "unknown")
            self.escalation_counts[cap] = self.escalation_counts.get(cap, 0) + 1
        logger.info("ingested %d telemetry events", len(events))

    def routing_updates(self, threshold: int = 3) -> list[str]:
        """Capabilities escalating >= threshold times: candidates for new adapters."""
        return sorted(c for c, n in self.escalation_counts.items() if n >= threshold)
