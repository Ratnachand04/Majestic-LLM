"""Per-device certification: what is recorded, and what it may claim (§11).

The problem this closes. Compilation and format conversion introduce numerical
differences. An artefact that quantised cleanly on your workstation can produce
**different outputs** through a different runtime on a different SoC — same
weights, same grammar, different answers. Nothing upstream catches that, because
everything upstream ran somewhere else.

So the last step of a build is to run the packaged artefact plus a small eval
subset **on the target device** and record what happened. Twenty held-out
examples is enough, and it is the only check that closes the loop between "we
built it" and "it works there".

Two properties are insisted on here, and both are the difference between a
certificate and a decoration:

**Certification is per device, not per artefact.** The same cartridge may be
certified for one SoC and uncertified for another. The manifest holds an array,
and a device absent from that array is reported as unverified rather than assumed
to be fine. A ledger that answers "yes" for a device it never saw is worse than
no ledger at all.

**Certification is per artefact KIND.** A merged model and a base-plus-adapter
pair are different artefacts with different numerics (§13), so a certificate
earned by one says nothing about the other. Certify whichever one actually ships.

The throughput number is the part customers ask about. The eval subset is the
part that matters.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence

from majestic.logging_utils import get_logger
from modelrig.probe import DeviceProfile, ProfileSource, Tier

logger = get_logger(__name__)

MB = 1_000_000

#: §11's figure. Twenty is small enough to run on a phone in a couple of minutes
#: and large enough to catch a runtime that changed the numerics.
# HYPOTHESIS — the subset size at which parity divergence becomes reliably
# detectable is not measured. Twenty is the spec's number and a deliberate
# minimum, not an optimum.
MIN_EVAL_SUBSET = 20

#: Below this share of on-device outputs matching the workstation, the artefact
#: is not the artefact that was evaluated and must not inherit its certificate.
# HYPOTHESIS — a policy threshold, not a measurement.
MIN_OUTPUT_PARITY = 0.95

#: How far the on-device eval score may fall below the workstation score before
#: the certificate is refused, in absolute points.
MAX_SCORE_REGRESSION = 0.03


class CertificationError(ValueError):
    """A certificate was requested that the evidence does not support."""


class ArtefactKind(str, Enum):
    """Which physical artefact was verified (§4, §13).

    Not a formality. The merged model and the base-plus-adapter pair take
    different code paths through the runtime and quantise differently, so their
    numerics differ. A certificate names the one it measured.
    """

    MERGED = "merged"
    SEPARATE = "separate"


class VerificationSource(str, Enum):
    """Where the certifying run happened. Ordered by what it licenses."""

    CUSTOMER_DEVICE = "customer_device"   # the unit the cartridge will live on
    DEVICE_LAB = "device_lab"             # our own reference unit for this tier
    EMULATED = "emulated"                 # cgroup-constrained: OOM only, never speed
    INTERPOLATED = "interpolated"         # a regression's guess. Not a measurement.

    @property
    def tier(self) -> Tier:
        if self in (VerificationSource.CUSTOMER_DEVICE, VerificationSource.DEVICE_LAB):
            return Tier.MEASURED
        if self is VerificationSource.EMULATED:
            return Tier.EMULATED
        return Tier.ANALYTICAL

    @property
    def may_promise(self) -> bool:
        return self.tier.may_promise

    @classmethod
    def of(cls, source: ProfileSource) -> VerificationSource:
        return {
            ProfileSource.PROBE: cls.CUSTOMER_DEVICE,
            ProfileSource.DEVICE_LAB: cls.DEVICE_LAB,
        }.get(source, cls.INTERPOLATED)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# =========================================================================== #
# The on-device run
# =========================================================================== #
@dataclass(frozen=True)
class OnDeviceRun:
    """Raw results of executing the packaged artefact on the target.

    ``reference_outputs`` are what the *same* eval examples produced on the
    workstation, before packaging. Without them the parity check is impossible
    and the run can only report throughput — which is the number that matters
    least.
    """

    device_id: str
    artefact_kind: ArtefactKind
    tokens_per_s_burst: float
    tokens_per_s_sustained: float
    peak_ram_mb: int
    outputs: tuple[str, ...] = ()             # produced ON the device
    reference_outputs: tuple[str, ...] = ()   # produced on the workstation
    scores: tuple[float, ...] = ()            # per-example, scored on the device
    reference_score: float | None = None      # the workstation's eval score
    power_draw_w: float | None = None
    context: int = 2048
    runtime: str = "llama.cpp"
    ran_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if self.tokens_per_s_burst <= 0 or self.tokens_per_s_sustained <= 0:
            raise CertificationError(f"{self.device_id}: throughput must be positive")
        if self.tokens_per_s_sustained > self.tokens_per_s_burst * 1.001:
            # A device that speeds up as it heats is a broken benchmark, not a
            # discovery. Fitting it would launder the error into a promise.
            raise CertificationError(
                f"{self.device_id}: sustained {self.tokens_per_s_sustained:.1f} tok/s "
                f"exceeds burst {self.tokens_per_s_burst:.1f} — thermal state, caching "
                "or contention corrupted the run"
            )
        if self.peak_ram_mb <= 0:
            raise CertificationError(f"{self.device_id}: peak RAM must be positive")
        if (self.outputs and self.reference_outputs
                and len(self.outputs) != len(self.reference_outputs)):
            raise CertificationError(
                f"{self.device_id}: {len(self.outputs)} on-device outputs against "
                f"{len(self.reference_outputs)} reference outputs — parity needs pairs"
            )

    @property
    def n_examples(self) -> int:
        return max(len(self.outputs), len(self.scores))

    @property
    def eval_subset_score(self) -> float | None:
        return sum(self.scores) / len(self.scores) if self.scores else None

    @property
    def output_parity(self) -> float | None:
        """Share of on-device outputs identical to the workstation's.

        This is the check §11 exists for. Aggregate score can hold steady while
        individual answers change — the same failure the answer-flip rate catches
        after quantisation, reappearing at the runtime boundary.
        """
        if not self.outputs or not self.reference_outputs:
            return None
        same = sum(a == b for a, b in zip(self.outputs, self.reference_outputs))
        return same / len(self.outputs)

    @property
    def thermal_derate(self) -> float:
        return self.tokens_per_s_sustained / self.tokens_per_s_burst

    def energy_per_1k_tokens_j(self) -> float | None:
        """``E = P_draw * 1000 / tok/s`` at the sustained rate.

        .. note::

           §11's illustrative record pairs ``tokens_per_s_sustained: 5.4`` with
           ``energy_per_1k_tokens_j: 41.2``. Those two cannot both hold: 1000
           tokens at 5.4 tok/s takes 185 seconds, so 41.2 J implies a package
           draw of **0.22 W**, roughly 16x below the 3.5 W Part 4 §13.6
           establishes for sustained mobile inference. The same pair at 3.5 W is
           648 J.

           This computes ``E = P * t`` from the *measured* draw, which is
           consistent with §13.6 and with the physics. The spec's figure is
           treated as a typo in an example rather than a constant to reproduce.
        """
        if self.power_draw_w is None:
            return None
        return self.power_draw_w * 1000.0 / self.tokens_per_s_sustained


# =========================================================================== #
# The certificate
# =========================================================================== #
@dataclass(frozen=True)
class DeviceCertificate:
    """One entry of the manifest's ``measured_performance`` array."""

    device_id: str
    source: VerificationSource
    artefact_kind: ArtefactKind
    tokens_per_s: float
    tokens_per_s_sustained: float
    peak_ram_mb: int
    verified_at: str
    eval_subset_score: float | None = None
    eval_subset_size: int = 0
    output_parity: float | None = None
    energy_per_1k_tokens_j: float | None = None
    soc: str = ""
    runtime: str = ""
    context: int = 0
    probe_version: str = ""
    notes: tuple[str, ...] = ()

    @property
    def may_promise(self) -> bool:
        """A commitment needs a measurement AND evidence the outputs survived."""
        return self.source.may_promise and self.closes_the_loop

    @property
    def closes_the_loop(self) -> bool:
        """Whether the eval subset actually ran on the device.

        Throughput alone certifies nothing. A cartridge that generates fast and
        answers differently is a worse outcome than one that is merely slow.
        """
        return (
            self.eval_subset_size >= MIN_EVAL_SUBSET
            and self.eval_subset_score is not None
            and self.output_parity is not None
        )

    def covers(self, device_id: str, kind: ArtefactKind | None = None) -> bool:
        if self.device_id != device_id:
            return False
        return kind is None or self.artefact_kind is kind

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source"] = self.source.value
        data["artefact_kind"] = self.artefact_kind.value
        data["notes"] = list(self.notes)
        data["may_promise"] = self.may_promise
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeviceCertificate:
        data = dict(data)
        data.pop("may_promise", None)
        data["source"] = VerificationSource(str(data.get("source", "interpolated")))
        data["artefact_kind"] = ArtefactKind(str(data.get("artefact_kind", "merged")))
        data["notes"] = tuple(data.get("notes", ()))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    def summary(self) -> str:
        rate = f"{self.tokens_per_s_sustained:.1f} tok/s sustained"
        if not self.closes_the_loop:
            return (f"{self.device_id}: {rate}, but the eval subset did not run "
                    "— throughput only")
        return (
            f"{self.device_id}: {rate}, {self.peak_ram_mb} MB peak, "
            f"eval {self.eval_subset_score:.3f} over {self.eval_subset_size} examples "
            f"on the device ({self.artefact_kind.value} artefact)"
        )


# =========================================================================== #
# Verification
# =========================================================================== #
@dataclass(frozen=True)
class VerificationOutcome:
    """Either a certificate, or the reasons one cannot be issued."""

    certificate: DeviceCertificate | None = None
    refusals: tuple[str, ...] = ()

    @property
    def certified(self) -> bool:
        return self.certificate is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "certified": self.certified,
            "certificate": self.certificate.to_dict() if self.certificate else None,
            "refusals": list(self.refusals),
        }


def verify_on_device(
    run: OnDeviceRun,
    profile: DeviceProfile | None = None,
    *,
    soc: str = "",
    ram_ceiling_mb: int | None = None,
    min_subset: int = MIN_EVAL_SUBSET,
    min_parity: float = MIN_OUTPUT_PARITY,
    max_regression: float = MAX_SCORE_REGRESSION,
) -> VerificationOutcome:
    """Turn an on-device run into a certificate, or refuse with reasons.

    Refusal is the common and correct outcome for a run that only measured
    throughput. Issuing a certificate from tokens per second alone would make the
    manifest say "verified on this device" about a property nobody checked.
    """
    refusals: list[str] = []
    notes: list[str] = []

    source = (VerificationSource.of(profile.source) if profile
              else VerificationSource.CUSTOMER_DEVICE)

    if run.n_examples < min_subset:
        refusals.append(
            f"the eval subset ran {run.n_examples} examples, below the floor of "
            f"{min_subset}: compilation and format conversion change numerics, and a "
            "subset this small cannot detect it"
        )

    parity = run.output_parity
    if parity is None:
        refusals.append(
            "no workstation outputs were supplied, so on-device output parity could "
            "not be checked — the one thing this run exists to catch"
        )
    elif parity < min_parity:
        refusals.append(
            f"output parity {parity:.1%} against the workstation is below "
            f"{min_parity:.0%}: this runtime on this SoC is not reproducing the "
            "artefact that was evaluated, so the eval certificate does not transfer"
        )

    score = run.eval_subset_score
    if score is None:
        refusals.append("the eval subset produced no scores")
    elif run.reference_score is not None and score < run.reference_score - max_regression:
        refusals.append(
            f"on-device eval {score:.3f} is {run.reference_score - score:.3f} below the "
            f"workstation's {run.reference_score:.3f}, past the {max_regression:.3f} "
            "tolerance — the artefact degraded in transit, not in training"
        )

    ceiling = ram_ceiling_mb if ram_ceiling_mb is not None else (
        profile.usable_ram_mb if profile else None
    )
    if ceiling is not None and run.peak_ram_mb > ceiling:
        refusals.append(
            f"peak RAM {run.peak_ram_mb} MB exceeded the {ceiling} MB the device makes "
            "available in production"
        )

    if profile is not None and not profile.measured:
        refusals.append(
            f"the device profile is {profile.source.value}, so this run cannot be "
            "attributed to hardware anyone measured"
        )
    if profile is not None and profile.device_id != run.device_id:
        refusals.append(
            f"the run reports device {run.device_id!r} but the profile is for "
            f"{profile.device_id!r} — a certificate must name one device"
        )

    if run.thermal_derate < 0.5:
        # Not a refusal: it is a measurement, and a severe one. The promise uses
        # the sustained figure either way, so recording it is the honest move.
        notes.append(
            f"thermal derate {run.thermal_derate:.2f} — this device loses over half its "
            "throughput under sustained load, and every promise uses the sustained figure"
        )
    if run.artefact_kind is ArtefactKind.SEPARATE:
        notes.append(
            "certifies the base-plus-adapter artefact; the merged model is a different "
            "artefact with different numerics and is not covered"
        )

    if refusals:
        logger.warning(
            "certification: refusing %s on %s — %s",
            run.artefact_kind.value, run.device_id, refusals[0],
        )
        return VerificationOutcome(refusals=tuple(refusals))

    energy = run.energy_per_1k_tokens_j()
    certificate = DeviceCertificate(
        device_id=run.device_id,
        source=source,
        artefact_kind=run.artefact_kind,
        tokens_per_s=round(run.tokens_per_s_burst, 2),
        tokens_per_s_sustained=round(run.tokens_per_s_sustained, 2),
        peak_ram_mb=run.peak_ram_mb,
        verified_at=run.ran_at,
        eval_subset_score=round(score, 4) if score is not None else None,
        eval_subset_size=run.n_examples,
        output_parity=round(parity, 4) if parity is not None else None,
        energy_per_1k_tokens_j=round(energy, 1) if energy is not None else None,
        soc=soc,
        runtime=run.runtime,
        context=run.context,
        probe_version=profile.probe_version if profile else "",
        notes=tuple(notes),
    )
    logger.info("certification: %s", certificate.summary())
    return VerificationOutcome(certificate=certificate)


# =========================================================================== #
# The ledger
# =========================================================================== #
@dataclass
class CertificationLedger:
    """The manifest's ``measured_performance`` array.

    Its whole job is to be honest about absence. A device that has never been
    verified is reported as unverified — never as passing, never as failing, and
    never silently defaulted to the nearest device that happens to be present.
    """

    entries: list[DeviceCertificate] = field(default_factory=list)

    def add(self, certificate: DeviceCertificate) -> DeviceCertificate:
        """Record a certificate, superseding any earlier one for the same pair."""
        self.entries = [
            e for e in self.entries
            if not e.covers(certificate.device_id, certificate.artefact_kind)
        ]
        self.entries.append(certificate)
        return certificate

    def record(self, outcome: VerificationOutcome) -> DeviceCertificate | None:
        return self.add(outcome.certificate) if outcome.certificate else None

    def for_device(
        self, device_id: str, kind: ArtefactKind | None = None
    ) -> DeviceCertificate | None:
        for entry in self.entries:
            if entry.covers(device_id, kind):
                return entry
        return None

    def certified_for(self, device_id: str, kind: ArtefactKind | None = None) -> bool:
        entry = self.for_device(device_id, kind)
        return entry is not None and entry.closes_the_loop

    def may_promise_on(self, device_id: str, kind: ArtefactKind | None = None) -> bool:
        entry = self.for_device(device_id, kind)
        return entry is not None and entry.may_promise

    def status_for(self, device_id: str, kind: ArtefactKind | None = None) -> str:
        """A sentence fit to print on a Build Card. Never optimistic."""
        entry = self.for_device(device_id, kind)
        if entry is None:
            covered = self.for_device(device_id)
            if covered is not None and kind is not None:
                return (
                    f"UNVERIFIED on {device_id} for the {kind.value} artefact — the "
                    f"{covered.artefact_kind.value} artefact was certified there, and "
                    "the two have different numerics"
                )
            return (
                f"UNVERIFIED on {device_id}: this cartridge has never run on this "
                f"device. It is certified for {len(self.certified_devices)} other "
                "device(s)"
            )
        if not entry.closes_the_loop:
            return (
                f"THROUGHPUT ONLY on {device_id}: {entry.tokens_per_s_sustained:.1f} "
                "tok/s sustained, but no eval subset ran on the device"
            )
        return f"CERTIFIED — {entry.summary()}"

    @property
    def devices(self) -> list[str]:
        return sorted({e.device_id for e in self.entries})

    @property
    def certified_devices(self) -> list[str]:
        return sorted({e.device_id for e in self.entries if e.closes_the_loop})

    def slowest(self) -> DeviceCertificate | None:
        """The device a fleet-wide promise has to be made against."""
        candidates = [e for e in self.entries if e.closes_the_loop]
        return min(candidates, key=lambda e: e.tokens_per_s_sustained, default=None)

    def to_list(self) -> list[dict[str, Any]]:
        ordered = sorted(self.entries, key=lambda e: (e.device_id, e.artefact_kind.value))
        return [e.to_dict() for e in ordered]

    @classmethod
    def from_list(cls, rows: Iterable[dict[str, Any]] | None) -> CertificationLedger:
        return cls(entries=[DeviceCertificate.from_dict(r) for r in (rows or [])])

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_list(), indent=2, sort_keys=True), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> CertificationLedger:
        return cls.from_list(json.loads(Path(path).read_text(encoding="utf-8")))

    def build_card(self) -> dict[str, Any]:
        """What the customer is shown about where this cartridge actually runs."""
        slowest = self.slowest()
        return {
            "certified_devices": self.certified_devices,
            "recorded_devices": self.devices,
            "worst_case_sustained_tok_s": (
                slowest.tokens_per_s_sustained if slowest else None
            ),
            "entries": self.to_list(),
            "caveat": (
                "Certification is per device and per artefact kind. A device not "
                "listed here has not been verified, and no claim is made about it."
            ),
        }


def certify(
    ledger: CertificationLedger,
    run: OnDeviceRun,
    profile: DeviceProfile | None = None,
    **kwargs: Any,
) -> VerificationOutcome:
    """Verify a run and record it if it passes. The common entry point."""
    outcome = verify_on_device(run, profile, **kwargs)
    ledger.record(outcome)
    return outcome


def unverified_devices(
    ledger: CertificationLedger, fleet: Sequence[str], kind: ArtefactKind | None = None
) -> list[str]:
    """Which devices in a fleet are still uncovered. The rollout's to-do list."""
    return [d for d in fleet if not ledger.certified_for(d, kind)]


__all__ = [
    "MAX_SCORE_REGRESSION", "MIN_EVAL_SUBSET", "MIN_OUTPUT_PARITY",
    "ArtefactKind", "CertificationError", "CertificationLedger", "DeviceCertificate",
    "OnDeviceRun", "VerificationOutcome", "VerificationSource",
    "certify", "unverified_devices", "verify_on_device",
]
