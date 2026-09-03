"""The build-and-ship service behind the web UI.

Framework-agnostic, like :mod:`api.routes` — the FastAPI wiring lives in
:mod:`api.app` so this stays testable without a server.

One request from the browser walks the whole compiler: FORGE turns a
description into a typed spec, the Planner admits or refuses it before anything
is spent, the Data Factory amplifies the seeds, the Trainer fits a real model,
the Proving Ground certifies it against held-out data, and the Registry stores
the manifest beside the weights. The UI then serves that model and, on request,
freezes it into a standalone binary.

A refusal is a first-class result here, not an error. The panel-facing point of
this system is that it declines to build things that cannot work, and says why —
so the API returns a refusal with the same status code as a success and the UI
renders it as an outcome rather than a failure.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from majestic.logging_utils import get_logger

logger = get_logger(__name__)

REGISTRY_PATH = Path("./registry")
DIST_PATH = Path("./dist")

#: Seed floors are per-primitive and enforced by the Data Factory. Classify is
#: the lowest at 80, so anything below that cannot be built by any route — the
#: UI says so up front rather than letting the request travel and fail.
MIN_SEEDS_HINT = 80


@dataclass
class BuildOutcome:
    """What one build produced — an admitted cartridge or an explained refusal."""

    admitted: bool = False
    cartridge_id: str | None = None
    stage_reached: str = ""
    refusal: str = ""
    gates: list[dict[str, Any]] = field(default_factory=list)
    axes: list[dict[str, Any]] = field(default_factory=list)
    status: str = ""
    plain_summary: str = ""
    plan: dict[str, Any] = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)
    #: Real examples the user supplied, minus what was locked away for testing.
    n_seeds: int = 0
    #: Rows actually trained on — larger than ``n_seeds`` once the Data Factory
    #: has amplified. Reporting only one of the two invites the reading that
    #: amplification did not happen, or that it inflated the evidence; it did
    #: the first and never the second, and the split is what proves it.
    n_train: int = 0
    n_holdout: int = 0
    questions: list[dict[str, Any]] = field(default_factory=list)
    weights_bytes: int = 0
    #: Per-stage telemetry, in the order the compiler ran them.
    stages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return round(sum(s.get("elapsed_ms", 0.0) for s in self.stages), 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "cartridge_id": self.cartridge_id,
            "stage_reached": self.stage_reached,
            "refusal": self.refusal,
            "gates": self.gates,
            "axes": self.axes,
            "status": self.status,
            "plain_summary": self.plain_summary,
            "plan": self.plan,
            "labels": self.labels,
            "n_seeds": self.n_seeds,
            "n_train": self.n_train,
            "n_holdout": self.n_holdout,
            "questions": self.questions,
            "weights_bytes": self.weights_bytes,
            "stages": self.stages,
            "total_ms": self.total_ms,
        }


def _probe_profile() -> dict[str, Any]:
    """A measured device profile.

    Without one ``P_lat`` refuses outright rather than promising a latency it
    never measured, so a UI build with no probe would always be refused. These
    are the figures from the reference mid-range Android unit.
    """
    from modelrig.probe import DeviceProfile
    from modelrig.probe import ProfileSource as ProbeSource

    return DeviceProfile(
        device_id="studio-reference-a536b", ram_total_mb=6000, ram_free_mb=4400,
        bw_eff_gbps=9.24, overhead_ms_per_token=2.16, thermal_derate_180s=0.6,
        prefill_ref_tok_s=50.0, reference_params=1_720_000_000,
        probe_lo_mb=400, probe_hi_mb=1_200, storage_free_mb=24_000,
        power_draw_w=3.5, simd=("neon", "dotprod"), source=ProbeSource.PROBE,
    ).to_dict()


def build(
    description: str,
    examples: list[tuple[str, str]],
    *,
    quality_gate: float = 0.80,
    offline: bool = True,
    registry_path: str | Path = REGISTRY_PATH,
    progress: Any = None,
) -> BuildOutcome:
    """Run one description plus its labelled examples through the compiler.

    ``progress`` is forwarded to the compiler and receives a ``StageEvent`` as
    each stage starts and finishes. Purely observational.
    """
    from modelrig.forge import Interviewer
    from modelrig.pipeline import MajesticCompiler

    outcome = BuildOutcome()
    if not description.strip():
        outcome.refusal = "a description is required"
        return outcome
    if len(examples) < 2:
        outcome.refusal = "at least two labelled examples are required"
        return outcome

    labels = sorted({label for _text, label in examples})
    outcome.labels = labels
    if len(labels) < 2:
        outcome.refusal = (
            f"all {len(examples)} examples carry the label {labels[0]!r}; a "
            "classifier needs at least two distinct labels to separate"
        )
        return outcome

    interview = Interviewer().conduct(
        description,
        device_profile=_probe_profile(),
        answers={
            "data_rights": "customer_owned",
            "quality_gate": quality_gate,
            "seed_data_count": len(examples),
            "offline_required": offline,
            "seed_data_ref": "studio://uploaded",
            "latency_budget_ms": 30_000,
            "expected_input_tokens": 120,
            "io_schema": {"label": "str"},
        },
    )

    # Questions FORGE would still ask are shown rather than hidden: the
    # interview is the part a customer sees, and its reasoning is the product.
    outcome.questions = [q.as_dict() for q in interview.pending]

    spec = interview.spec
    if spec is None:
        outcome.refusal = (
            "FORGE could not emit a specification from that description — it is "
            "missing something it refuses to guess"
        )
        return outcome

    result = MajesticCompiler(base_path=str(registry_path)).compile(
        spec, examples, progress=progress
    )

    outcome.stages = [e.as_dict() for e in result.stages]
    outcome.admitted = result.admitted
    outcome.cartridge_id = result.cartridge_id
    outcome.stage_reached = result.stage_reached
    outcome.refusal = result.refusal
    outcome.gates = [
        {"gate": g.gate, "passed": g.passed, "reasons": list(g.reasons)}
        for g in result.gates
    ]
    if result.plan is not None:
        outcome.plan = {
            "base_ref": result.plan.base_ref,
            "peft_method": result.plan.peft_method,
            "quantiser": result.plan.quantiser,
            "bit_width": result.plan.bit_width,
            "target": result.plan.target,
        }
    if result.scorecard is not None:
        card = result.scorecard
        outcome.axes = [
            {
                "name": a.name, "score": round(a.score, 4),
                "threshold": round(a.threshold, 4),
                "passed": a.passed, "blocking": a.blocking,
            }
            for a in card.axes
        ]
        outcome.status = card.status.value
        outcome.plain_summary = card.plain_summary()
        outcome.n_holdout = card.n_held_out
    elif result.cache_hit and result.cartridge is not None:
        # A cache hit skips the Proving Ground, so there is no fresh scorecard —
        # but the certificate was stored precisely so it need not be re-earned.
        # Reading it back is what makes reuse auditable instead of merely fast.
        _hydrate_from_cartridge(outcome, result.cartridge)

    # The Data Factory is the authority on the split — it did it. Deriving the
    # training count from the input size instead would report the seed count
    # under a heading that means "rows trained on", and quietly contradict the
    # factory's own telemetry.
    data_stage = next(
        (s for s in outcome.stages if s["stage"] == "data" and s["status"] == "ok"), None,
    )
    if data_stage is not None:
        outcome.n_train = int(data_stage["data"].get("n_train", 0))
        outcome.n_holdout = int(data_stage["data"].get("n_holdout", outcome.n_holdout))
    else:
        outcome.n_train = len(examples) - outcome.n_holdout
    outcome.n_seeds = len(examples) - outcome.n_holdout

    weights_dir = (
        Path(result.weights_path) if result.weights_path is not None
        else Path(registry_path) / "weights" / (result.cartridge_id or "")
    )
    if weights_dir.is_dir():
        outcome.weights_bytes = sum(f.stat().st_size for f in weights_dir.iterdir())

    logger.info(
        "studio: build %s (%s)",
        "admitted" if result.admitted else "refused", result.stage_reached,
    )
    return outcome


#: Fallback ordering for certificates written before ``axis_order`` was stored.
#: Blocking axes first, since those are the ones that decide admission.
_AXIS_FALLBACK_ORDER = (
    "task_metric", "safety", "privacy", "contamination",
    "calibrated_judge", "behavioural", "regression", "calibration",
)


def build_stream(
    description: str,
    examples: list[tuple[str, str]],
    *,
    quality_gate: float = 0.80,
    offline: bool = True,
    registry_path: str | Path = REGISTRY_PATH,
) -> Iterator[dict[str, Any]]:
    """The same build, yielded stage by stage as it happens.

    ``build`` runs on a worker thread and pushes events into a queue; this
    generator drains it. The compile is identical either way — the stream is a
    window onto it, not a second code path, so the sequence a viewer watches is
    the sequence that actually ran.

    Yields ``{"event": "stage", ...}`` per stage and one final
    ``{"event": "done", "outcome": ...}``. A crash arrives as
    ``{"event": "error"}`` rather than a truncated stream, so the page can say
    what went wrong instead of hanging on a step that will never complete.
    """
    import queue
    import threading

    events: queue.Queue[Any] = queue.Queue()
    _END = object()

    def run() -> None:
        try:
            outcome = build(
                description, examples, quality_gate=quality_gate, offline=offline,
                registry_path=registry_path,
                progress=lambda e: events.put(("stage", e.as_dict())),
            )
            events.put(("done", outcome.as_dict()))
        except Exception as exc:                     # noqa: BLE001
            logger.exception("studio: streaming build failed")
            events.put(("error", {"message": f"{type(exc).__name__}: {exc}"}))
        finally:
            events.put(_END)

    worker = threading.Thread(target=run, daemon=True, name="majestic-build")
    worker.start()

    while True:
        item = events.get()
        if item is _END:
            break
        kind, payload = item
        yield {"event": kind, **payload}
    worker.join(timeout=5.0)


def _labels_on_disk(weights_dir: Path) -> list[str]:
    """The classes the stored model can actually emit."""
    import json

    manifest = weights_dir / "model.json"
    if not manifest.is_file():
        return []
    try:
        return list(json.loads(manifest.read_text(encoding="utf-8")).get("labels", []))
    except (OSError, ValueError) as exc:
        logger.warning("studio: unreadable weights manifest %s: %s", manifest, exc)
        return []


def _hydrate_from_cartridge(outcome: BuildOutcome, cartridge: Any) -> None:
    """Rebuild the evidence view from a cartridge's stored certificate."""
    cert = cartridge.eval_certificate or {}
    axes = cert.get("axes") or {}
    order = [n for n in cert.get("axis_order", ()) if n in axes]
    order += sorted(
        (n for n in axes if n not in order),
        key=lambda n: (
            _AXIS_FALLBACK_ORDER.index(n) if n in _AXIS_FALLBACK_ORDER else 99, n
        ),
    )
    ordered = [(n, axes[n]) for n in order]
    outcome.axes = [
        {
            "name": name,
            "score": round(float(a.get("score", 0.0)), 4),
            # Older certificates predate the stored threshold; fall back to the
            # top-level gate rather than inventing a bar that was never applied.
            "threshold": round(float(a.get("threshold", cert.get("threshold", 0.0))), 4),
            "passed": bool(a.get("passed")),
            "blocking": bool(a.get("blocking")),
        }
        for name, a in ordered
    ]
    outcome.status = str(cert.get("status", ""))
    outcome.plain_summary = str(cert.get("plain_summary", ""))
    outcome.n_holdout = int(cert.get("n_test", 0))

    card = cartridge.model_card or {}
    outcome.plan = {
        "base_ref": cartridge.base_ref,
        "peft_method": card.get("training_method", ""),
        "quantiser": card.get("quantisation", ""),
        "bit_width": "",
        "target": card.get("target_runtime", ""),
    }


def list_models(registry_path: str | Path = REGISTRY_PATH) -> list[dict[str, Any]]:
    """Every built cartridge, with whether it is servable and packaged."""
    from modelrig.registry import CartridgeRegistry

    base = Path(registry_path)
    if not base.exists():
        return []

    registry = CartridgeRegistry(str(base))
    out: list[dict[str, Any]] = []
    for cid in registry.list():
        try:
            cart = registry.get(cid)
        except Exception as exc:  # noqa: BLE001 - a broken row must not hide the rest
            logger.warning("studio: could not load cartridge %s: %s", cid, exc)
            continue
        weights = base / "weights" / cid
        exe = DIST_PATH / cid[:12] / "majestic-model.exe"
        card = cart.model_card or {}
        out.append({
            "cartridge_id": cid,
            "base_ref": cart.base_ref,
            "task": card.get("task_primitive", ""),
            "intended_use": card.get("intended_use", ""),
            # From the weights, not the IO contract: ``output_schema`` is
            # ``{"label": "str"}`` — the shape of a response, not the set of
            # classes the model can emit.
            "labels": _labels_on_disk(weights),
            "certified": cart.certified,
            "servable": cart.servable and weights.is_dir(),
            "has_weights": weights.is_dir(),
            "packaged": exe.exists(),
            "exe_size_mb": round(exe.stat().st_size / 1_000_000, 1) if exe.exists() else None,
            "status": cart.status.value,
        })
    return out


def predict(
    cartridge_id: str, texts: list[str], registry_path: str | Path = REGISTRY_PATH
) -> dict[str, Any]:
    """Serve a built cartridge. The end of the pipeline, made visible."""
    from modelrig.pipeline import predict_with_cartridge

    cleaned = [t for t in (s.strip() for s in texts) if t]
    if not cleaned:
        return {"predictions": [], "error": "no input text"}
    try:
        preds = predict_with_cartridge(cartridge_id, cleaned, registry_path)
    except FileNotFoundError as exc:
        return {"predictions": [], "error": str(exc)}
    return {
        "predictions": [{"text": t, "label": p} for t, p in zip(cleaned, preds)],
        "error": "",
    }


def package(
    cartridge_id: str,
    registry_path: str | Path = REGISTRY_PATH,
    dist_dir: str | Path = DIST_PATH,
) -> dict[str, Any]:
    """Freeze a built cartridge into a standalone executable."""
    from modelrig.package_exe import package as _package

    result = _package(cartridge_id, registry_path=registry_path, dist_dir=dist_dir)
    return result.as_dict()


def exe_path(cartridge_id: str, dist_dir: str | Path = DIST_PATH) -> Path | None:
    """Where a packaged binary lives, if it has been built."""
    import sys

    name = "majestic-model.exe" if sys.platform == "win32" else "majestic-model"
    candidate = Path(dist_dir) / cartridge_id[:12] / name
    return candidate if candidate.exists() else None


#: A ready-made corpus so the UI is demonstrable in one click. Real enough to
#: train on and large enough to clear the classify seed floor.
_SAMPLE_POS = [
    "excellent quality throughout", "love this service", "the staff were helpful",
    "happy with the fast result", "wonderful and nice experience",
    "great work, very pleased", "friendly and quick support",
    "resolved my issue perfectly", "brilliant response time", "very satisfied overall",
]
_SAMPLE_NEG = [
    "terrible broken service", "hate this experience", "sad and useless outcome",
    "awful quality throughout", "unhelpful and rude staff",
    "far too slow to respond", "the issue was never fixed", "very disappointed overall",
    "waste of my time", "bad and frustrating process",
]


def sample_dataset(n: int = 120) -> list[dict[str, str]]:
    """A labelled sentiment corpus for the demo path."""
    rows: list[dict[str, str]] = []
    for i in range(n // 2):
        rows.append({"text": f"{_SAMPLE_POS[i % len(_SAMPLE_POS)]} (ticket {i})",
                     "label": "positive"})
        rows.append({"text": f"{_SAMPLE_NEG[i % len(_SAMPLE_NEG)]} (ticket {i})",
                     "label": "negative"})
    return rows


__all__ = [
    "MIN_SEEDS_HINT", "BuildOutcome",
    "build", "build_stream", "exe_path", "list_models", "package", "predict",
    "sample_dataset",
]
