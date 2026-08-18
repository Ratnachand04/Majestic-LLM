"""The compounding device model (§12).

Every probe returns a ``(SoC, model_size, tok/s, thermal_derate)`` tuple.
Accumulated across customers, these fit a device-class latency predictor **with
no device lab at all**.

Two consequences:

* Interpolated predictions for un-probed devices improve monotonically with
  deployment count.
* A new customer on a previously-seen SoC can skip the probe entirely, using the
  accumulated profile — cutting build latency at no cost to soundness, **provided
  the tier is honestly marked**. An interpolated profile is never
  :class:`~modelrig.probe.Tier.MEASURED`, however many observations back it.

This is the same compounding structure as the build-outcome meta-learner: a
measurement made for one customer improves planning for every subsequent one. It
is the strongest form of moat available here, because it requires deployments
rather than cleverness.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from majestic.logging_utils import get_logger
from modelrig.probe import DeviceProfile, ProfileSource

logger = get_logger(__name__)

#: Observations of one SoC below which an interpolated profile is not offered.
# HYPOTHESIS — three is enough to see spread but not enough to trust the mean.
MIN_OBSERVATIONS = 3


@dataclass(frozen=True)
class ProbeObservation:
    """One accumulated probe result, reduced to what generalises.

    Deliberately narrow: SoC identity, the calibrated bandwidth, the per-token
    overhead and the thermal derate. Nothing customer-specific is retained,
    because the probe measured hardware and only hardware.
    """

    soc: str
    bw_eff_gbps: float
    overhead_ms_per_token: float
    thermal_derate: float
    ram_total_mb: int
    simd: tuple[str, ...] = ()
    accelerator: str = "cpu"
    observed_at: str = ""

    @classmethod
    def of(cls, profile: DeviceProfile, soc: str) -> ProbeObservation:
        return cls(
            soc=soc,
            bw_eff_gbps=profile.bw_eff_gbps,
            overhead_ms_per_token=profile.overhead_ms_per_token,
            thermal_derate=profile.thermal_derate_180s,
            ram_total_mb=profile.ram_total_mb,
            simd=profile.simd,
            accelerator=profile.accelerator,
            observed_at=profile.measured_at or datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "soc": self.soc, "bw_eff_gbps": self.bw_eff_gbps,
            "overhead_ms_per_token": self.overhead_ms_per_token,
            "thermal_derate": self.thermal_derate, "ram_total_mb": self.ram_total_mb,
            "simd": list(self.simd), "accelerator": self.accelerator,
            "observed_at": self.observed_at,
        }


@dataclass
class SocSummary:
    """What the accumulated probes say about one SoC."""

    soc: str
    n: int
    bw_eff_gbps: float
    bw_spread: float
    overhead_ms_per_token: float
    thermal_derate: float
    ram_total_mb: int
    simd: tuple[str, ...]
    accelerator: str

    @property
    def confident(self) -> bool:
        return self.n >= MIN_OBSERVATIONS

    @property
    def dispersion(self) -> float:
        """Relative spread of bandwidth across units of the same SoC.

        High dispersion means the SoC label is not predicting the thing it is
        being asked to predict — different phones, thermal designs and memory
        configurations hide behind one name.
        """
        return self.bw_spread / self.bw_eff_gbps if self.bw_eff_gbps else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "soc": self.soc, "observations": self.n,
            "bw_eff_gbps": round(self.bw_eff_gbps, 3),
            "bw_dispersion": round(self.dispersion, 3),
            "overhead_ms_per_token": round(self.overhead_ms_per_token, 3),
            "thermal_derate": round(self.thermal_derate, 3),
            "confident": self.confident,
        }


class DeviceDB:
    """Accumulated probes, and the interpolated profiles they support."""

    def __init__(self, min_observations: int = MIN_OBSERVATIONS) -> None:
        self.observations: list[ProbeObservation] = []
        self.min_observations = min_observations

    # -- accumulation ----------------------------------------------------- #
    def record(self, observation: ProbeObservation) -> None:
        self.observations.append(observation)

    def record_profile(self, profile: DeviceProfile, soc: str) -> None:
        """Only genuine measurements enter the database.

        Admitting an interpolated profile would let the model train on its own
        output — the same degenerate loop the Data Factory refuses.
        """
        if not profile.measured:
            raise ValueError(
                f"refusing to record a {profile.source.value!r} profile: only measured "
                "probes may enter the device database, or it trains on its own output"
            )
        self.record(ProbeObservation.of(profile, soc))

    def __len__(self) -> int:
        return len(self.observations)

    @property
    def socs(self) -> list[str]:
        return sorted({o.soc for o in self.observations})

    def for_soc(self, soc: str) -> list[ProbeObservation]:
        return [o for o in self.observations if o.soc == soc]

    # -- the regression --------------------------------------------------- #
    def summarise(self, soc: str) -> SocSummary | None:
        """Aggregate one SoC. Median, not mean — one throttled outlier should not
        drag the estimate, and throttled outliers are exactly what phones produce.
        """
        rows = self.for_soc(soc)
        if not rows:
            return None
        bws = [o.bw_eff_gbps for o in rows]
        return SocSummary(
            soc=soc,
            n=len(rows),
            bw_eff_gbps=statistics.median(bws),
            bw_spread=statistics.pstdev(bws) if len(bws) > 1 else 0.0,
            overhead_ms_per_token=statistics.median(o.overhead_ms_per_token for o in rows),
            # The MINIMUM derate, not the median: a thermal promise must hold for
            # the worst unit seen, not the typical one.
            thermal_derate=min(o.thermal_derate for o in rows),
            ram_total_mb=int(statistics.median(o.ram_total_mb for o in rows)),
            simd=rows[-1].simd,
            accelerator=rows[-1].accelerator,
        )

    def interpolate(self, soc: str, *, device_id: str = "", ram_free_mb: int | None = None
                    ) -> DeviceProfile | None:
        """An interpolated profile for a previously-seen SoC.

        Returns ``None`` below the observation floor rather than a shaky guess —
        a caller that gets ``None`` falls back to the ``devices.yaml`` prior and
        marks the build ``assumed``, which is honest. The returned profile is
        always :attr:`ProfileSource.INTERPOLATED`, so it can never license a
        latency promise no matter how many observations back it.
        """
        summary = self.summarise(soc)
        if summary is None or not summary.confident:
            return None
        return DeviceProfile(
            device_id=device_id or f"{soc}-interpolated",
            ram_total_mb=summary.ram_total_mb,
            ram_free_mb=ram_free_mb if ram_free_mb is not None else int(summary.ram_total_mb * 0.6),
            bw_eff_gbps=summary.bw_eff_gbps,
            overhead_ms_per_token=summary.overhead_ms_per_token,
            thermal_derate_180s=summary.thermal_derate,
            simd=summary.simd,
            accelerator=summary.accelerator,
            source=ProfileSource.INTERPOLATED,
            measured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def needs_probe(self, soc: str) -> bool:
        """Whether a new customer on this SoC still has to run the probe."""
        summary = self.summarise(soc)
        return summary is None or not summary.confident

    # -- how the moat is doing -------------------------------------------- #
    def coverage(self) -> dict[str, Any]:
        """Deployment-count-driven progress, which is the point of §12."""
        summaries = [self.summarise(s) for s in self.socs]
        confident = [s for s in summaries if s and s.confident]
        noisy = [s.soc for s in summaries if s and s.dispersion > 0.25]
        return {
            "observations": len(self.observations),
            "socs_seen": len(self.socs),
            "socs_confident": len(confident),
            "probe_skippable": [s.soc for s in confident],
            "high_dispersion": noisy,
            "note": (
                "interpolated profiles never reach the measured tier, however many "
                "observations back them"
            ),
        }

    # -- persistence -------------------------------------------------------- #
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps([o.to_dict() for o in self.observations], indent=2),
            encoding="utf-8",
        )
        return p

    def load(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            return
        self.observations = [
            ProbeObservation(
                soc=r["soc"], bw_eff_gbps=float(r["bw_eff_gbps"]),
                overhead_ms_per_token=float(r["overhead_ms_per_token"]),
                thermal_derate=float(r["thermal_derate"]),
                ram_total_mb=int(r["ram_total_mb"]),
                simd=tuple(r.get("simd", ())), accelerator=r.get("accelerator", "cpu"),
                observed_at=r.get("observed_at", ""),
            )
            for r in json.loads(p.read_text(encoding="utf-8"))
        ]


def resolve_profile(
    soc: str,
    db: DeviceDB,
    *,
    probed: DeviceProfile | None = None,
    prior: DeviceProfile | None = None,
) -> tuple[DeviceProfile | None, ProfileSource]:
    """The §9 fallback ladder: probe, then interpolation, then a prior.

    Returns the profile and the source it came from, so the Build Card can say
    exactly which and never imply more.
    """
    if probed is not None and probed.measured:
        return probed, probed.source
    interpolated = db.interpolate(soc)
    if interpolated is not None:
        return interpolated, ProfileSource.INTERPOLATED
    if prior is not None:
        return prior, ProfileSource.ASSUMED
    return None, ProfileSource.ASSUMED
