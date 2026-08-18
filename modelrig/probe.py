"""The device probe and the three-tier verification ladder (Part 3 §6-§9).

The honest problem: **you must certify an artefact for hardware you do not
possess.** There are hundreds of Android SoCs and you will own two or three, so
every certification you issue is for a device you have never touched.

Four approaches, honestly graded:

===================  ==========  ======  =======  ==================
approach             cost        memory  speed    sound?
===================  ==========  ======  =======  ==================
analytical           free        yes     bound    refusal only
emulated             cheap       yes     no       partially
reference device     one/tier    yes     that one interpolation unsound
probe the device     round trip  yes     yes      **sound**
===================  ==========  ======  =======  ==================

The fourth is the answer, and nobody offers it because it requires the
customer's device to participate in the build. Majestic can require it, because
the device is where the model is going to live anyway.

**The result that makes it practical.** Decode at batch 1 is memory-bandwidth
bound — every generated token reads the whole weight matrix — but per-token
framework overhead is not zero, so the honest model is affine:

    t_token(S) = S / BW_eff + c_overhead

Two unknowns, so two probe points solve it exactly. **Two measurements on a
200 MB and a 400 MB model calibrate latency prediction for every base in the
catalogue on that device.** You never ship the real model to find out how fast
the real model will be.

The one-sentence version of the whole document: *the device certifies the model,
rather than the vendor certifying the device.*
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from majestic.logging_utils import get_logger

logger = get_logger(__name__)

MB = 1_000_000


class Tier(int, Enum):
    """How a device claim was established. Never presented as more than it is."""

    ANALYTICAL = 1   # roofline bound. Sound for REFUSAL only.
    EMULATED = 2     # ran under cgroup pressure. Catches OOM, never slowness.
    MEASURED = 3     # ran on the actual silicon. The only tier that may promise.

    @property
    def may_promise(self) -> bool:
        """Only a measurement licenses a commitment."""
        return self is Tier.MEASURED

    @property
    def may_refuse(self) -> bool:
        """Every tier may refuse: all three only ever subtract from performance."""
        return True


class ProfileSource(str, Enum):
    """Where a :class:`DeviceProfile` came from (§14)."""

    PROBE = "probe"                       # the customer's actual device
    DEVICE_LAB = "device_lab"             # our own reference unit for this tier
    REFERENCE_DEVICE = "reference_device"  # a different unit of the same class
    INTERPOLATED = "interpolated"         # regression over accumulated probes
    ASSUMED = "assumed"                   # a devices.yaml prior. A guess.

    @property
    def tier(self) -> Tier:
        if self in (ProfileSource.PROBE, ProfileSource.DEVICE_LAB):
            return Tier.MEASURED
        return Tier.ANALYTICAL

    @property
    def measured(self) -> bool:
        return self.tier is Tier.MEASURED


class ProbeError(ValueError):
    """The probe returned something that cannot be calibrated."""


# =========================================================================== #
# Two-point calibration (§8)
# =========================================================================== #
@dataclass(frozen=True)
class ProbePoint:
    """One benchmark run: a model of known size, and the rate it achieved."""

    size_bytes: int
    tokens_per_s: float
    phase: str = "burst"          # burst (~30 s) | sustained (~180 s)

    def __post_init__(self) -> None:
        if self.size_bytes <= 0:
            raise ProbeError("probe model size must be positive")
        if self.tokens_per_s <= 0:
            raise ProbeError("probe throughput must be positive")

    @property
    def seconds_per_token(self) -> float:
        return 1.0 / self.tokens_per_s


@dataclass(frozen=True)
class Calibration:
    """The affine decode model fitted from two probe points."""

    bw_eff_bytes_per_s: float
    overhead_s_per_token: float
    lo_bytes: int                  # smallest probe size, for the validity range
    hi_bytes: int                  # largest probe size

    @property
    def bw_eff_gbps(self) -> float:
        return self.bw_eff_bytes_per_s / 1e9

    def seconds_per_token(self, size_bytes: int) -> float:
        """``t = S / BW_eff + c``."""
        return size_bytes / self.bw_eff_bytes_per_s + self.overhead_s_per_token

    def tokens_per_s(self, size_bytes: int) -> float:
        return 1.0 / self.seconds_per_token(size_bytes)

    def extrapolation_factor(self, size_bytes: int) -> float:
        """How far outside the probed range a prediction reaches.

        1.0 means inside the bracket. §13: the affine model is good within
        roughly an order of magnitude, so predicting a 4 GB model from 200/400 MB
        probes is a 10x reach and unreliable.
        """
        if self.lo_bytes <= size_bytes <= self.hi_bytes:
            return 1.0
        if size_bytes > self.hi_bytes:
            return size_bytes / self.hi_bytes
        return self.lo_bytes / size_bytes

    def trustworthy_for(self, size_bytes: int, max_reach: float = 10.0) -> bool:
        return self.extrapolation_factor(size_bytes) <= max_reach


def calibrate(points: Sequence[ProbePoint]) -> Calibration:
    """Solve ``t = S/BW + c`` from two or more probe points.

    With exactly two points the solution is exact:

        BW_eff = (S2 - S1) / (t2 - t1),    c = t1 - S1 / BW_eff

    With three or more, least squares is used instead — §13 recommends a third
    point when the target sits far from the bracket, and refusing to use the
    extra information would be perverse.

    Raises :class:`ProbeError` when the points cannot yield a positive
    bandwidth, which happens when a larger model measured *faster* than a
    smaller one. That is a broken benchmark (thermal state, caching, contention),
    not a device that defies the roofline, and it must not be silently fitted.
    """
    burst = [p for p in points if p.phase == "burst"]
    if len(burst) < 2:
        raise ProbeError("two-point calibration needs at least two burst-phase points")

    distinct = {p.size_bytes for p in burst}
    if len(distinct) < 2:
        raise ProbeError("probe points must use at least two distinct model sizes")

    if len(burst) == 2:
        (p1, p2) = sorted(burst, key=lambda p: p.size_bytes)
        dt = p2.seconds_per_token - p1.seconds_per_token
        if dt <= 0:
            raise ProbeError(
                f"the {p2.size_bytes / MB:.0f} MB probe was not slower than the "
                f"{p1.size_bytes / MB:.0f} MB one ({p2.tokens_per_s:.1f} vs "
                f"{p1.tokens_per_s:.1f} tok/s). Decode is bandwidth-bound, so this is "
                "a broken benchmark — thermal state, page cache or contention — not a "
                "device that beats the roofline."
            )
        bw = (p2.size_bytes - p1.size_bytes) / dt
        overhead = p1.seconds_per_token - p1.size_bytes / bw
    else:
        bw, overhead = _least_squares(burst)

    if bw <= 0:
        raise ProbeError("calibration produced a non-positive bandwidth")
    if overhead < 0:
        # A negative intercept is unphysical; clamp and say so rather than
        # shipping a model that predicts free tokens for a zero-size model.
        logger.warning(
            "probe: fitted overhead was negative (%.4f s); clamping to zero. "
            "The probe sizes are probably too close together.", overhead,
        )
        overhead = 0.0

    sizes = sorted(distinct)
    return Calibration(bw, overhead, sizes[0], sizes[-1])


def _least_squares(points: Sequence[ProbePoint]) -> tuple[float, float]:
    """Fit ``t = m*S + c`` over three or more points; ``BW = 1/m``."""
    n = len(points)
    xs = [float(p.size_bytes) for p in points]
    ys = [p.seconds_per_token for p in points]
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        raise ProbeError("probe points are collinear in size")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    if slope <= 0:
        raise ProbeError("fitted slope is non-positive: larger models must be slower")
    return 1.0 / slope, mean_y - slope * mean_x


# =========================================================================== #
# Thermal derate (§8)
# =========================================================================== #
def thermal_derate(burst_tokens_per_s: float, sustained_tokens_per_s: float) -> float:
    """``rho = tok/s at 180 s / tok/s at 30 s``.

    A 30-second benchmark will not reveal throttling that appears at four
    minutes. On passively cooled phones this is routinely 0.5-0.8, and ignoring
    it is how a demo that works becomes a deployment that does not.

    Values above 1.0 are clamped: a device cannot sustainably exceed its own
    burst rate, so that reading means the burst phase was itself throttled or the
    device was still warming up.
    """
    if burst_tokens_per_s <= 0:
        raise ProbeError("burst throughput must be positive")
    if sustained_tokens_per_s <= 0:
        raise ProbeError("sustained throughput must be positive")
    return min(1.0, sustained_tokens_per_s / burst_tokens_per_s)


# =========================================================================== #
# The profile the probe returns (§8)
# =========================================================================== #
@dataclass
class DeviceProfile:
    """Measured hardware facts. The probe measures hardware, never content."""

    device_id: str
    ram_total_mb: int
    ram_free_mb: int                 # FREE, measured — not total
    bw_eff_gbps: float
    overhead_ms_per_token: float
    thermal_derate_180s: float = 1.0
    cores_perf: int = 0
    cores_eff: int = 0
    simd: tuple[str, ...] = ()       # neon | dotprod | i8mm | sve — selects format
    accelerator: str = "cpu"
    storage_free_mb: int = 0
    source: ProfileSource = ProfileSource.PROBE
    probe_version: str = "1"
    measured_at: str = ""
    #: Probe bracket, so extrapolation reach stays checkable after the fact.
    probe_lo_mb: int = 0
    probe_hi_mb: int = 0
    #: §13: the probe runs on an idle device; production shares RAM with other
    #: apps. Free RAM at probe time overstates free RAM in use.
    headroom_factor: float = 0.85

    @property
    def tier(self) -> Tier:
        return self.source.tier

    @property
    def measured(self) -> bool:
        return self.source.measured

    @property
    def usable_ram_mb(self) -> int:
        """Free RAM after the production-headroom discount (§13)."""
        return int(self.ram_free_mb * self.headroom_factor)

    @property
    def calibration(self) -> Calibration:
        return Calibration(
            bw_eff_bytes_per_s=self.bw_eff_gbps * 1e9,
            overhead_s_per_token=self.overhead_ms_per_token / 1000.0,
            lo_bytes=self.probe_lo_mb * MB or 1,
            hi_bytes=self.probe_hi_mb * MB or 1,
        )

    def tokens_per_s(self, weight_bytes: int, sustained: bool = True) -> float:
        """Predicted decode rate. Sustained applies the thermal derate.

        **All latency promises use the sustained figure.** The burst number is
        what a demo shows; the sustained number is what the customer lives with.
        """
        raw = self.calibration.tokens_per_s(weight_bytes)
        return raw * self.thermal_derate_180s if sustained else raw

    def seconds_per_token(self, weight_bytes: int, sustained: bool = True) -> float:
        return 1.0 / self.tokens_per_s(weight_bytes, sustained)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "ram_total_mb": self.ram_total_mb,
            "ram_free_mb": self.ram_free_mb,
            "bw_eff_gbps": round(self.bw_eff_gbps, 4),
            "overhead_ms_per_token": round(self.overhead_ms_per_token, 4),
            "thermal_derate_180s": round(self.thermal_derate_180s, 4),
            "cores_perf": self.cores_perf,
            "cores_eff": self.cores_eff,
            "simd": list(self.simd),
            "accelerator": self.accelerator,
            "storage_free_mb": self.storage_free_mb,
            "source": self.source.value,
            "probe_version": self.probe_version,
            "measured_at": self.measured_at,
            "probe_lo_mb": self.probe_lo_mb,
            "probe_hi_mb": self.probe_hi_mb,
            "headroom_factor": self.headroom_factor,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeviceProfile:
        data = dict(data)
        if "simd" in data:
            data["simd"] = tuple(data["simd"])
        if "source" in data:
            data["source"] = ProfileSource(str(data["source"]))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> DeviceProfile:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# =========================================================================== #
# Building a profile from a probe run
# =========================================================================== #
@dataclass
class ProbeBundle:
    """What ships to the device. Two tiny reference models plus a harness.

    No customer data is touched — the probe measures hardware, not content, which
    is what makes it acceptable to run on a device holding regulated records.
    """

    #: SOURCE: sizes chosen to bracket the deployable range while staying within
    #: one decade of a 1-2 GB cartridge (§13's extrapolation limit).
    small_mb: int = 200
    large_mb: int = 400
    burst_seconds: int = 30
    sustained_seconds: int = 180
    version: str = "1"

    def manifest(self) -> dict[str, Any]:
        return {
            "probe_version": self.version,
            "models_mb": [self.small_mb, self.large_mb],
            "burst_seconds": self.burst_seconds,
            "sustained_seconds": self.sustained_seconds,
            "touches_customer_data": False,
        }


def build_profile(
    device_id: str,
    points: Sequence[ProbePoint],
    *,
    ram_total_mb: int,
    ram_free_mb: int,
    sustained_tokens_per_s: float | None = None,
    burst_reference_tokens_per_s: float | None = None,
    cores_perf: int = 0,
    cores_eff: int = 0,
    simd: Sequence[str] = (),
    accelerator: str = "cpu",
    storage_free_mb: int = 0,
    source: ProfileSource = ProfileSource.PROBE,
    probe_version: str = "1",
    headroom_factor: float = 0.85,
) -> DeviceProfile:
    """Turn raw probe measurements into a :class:`DeviceProfile`.

    ``sustained_tokens_per_s`` should come from the sustained phase run at the
    SAME model size as ``burst_reference_tokens_per_s``; otherwise the ratio
    conflates thermal throttling with the size effect the calibration already
    models. When omitted the derate defaults to 1.0 and the profile says so.
    """
    cal = calibrate(points)

    derate = 1.0
    if sustained_tokens_per_s is not None:
        reference = burst_reference_tokens_per_s
        if reference is None:
            # Fall back to the largest burst point, the closest analogue to a
            # sustained workload.
            reference = max(
                (p for p in points if p.phase == "burst"), key=lambda p: p.size_bytes
            ).tokens_per_s
        derate = thermal_derate(reference, sustained_tokens_per_s)

    profile = DeviceProfile(
        device_id=device_id,
        ram_total_mb=ram_total_mb,
        ram_free_mb=ram_free_mb,
        bw_eff_gbps=cal.bw_eff_gbps,
        overhead_ms_per_token=cal.overhead_s_per_token * 1000.0,
        thermal_derate_180s=derate,
        cores_perf=cores_perf,
        cores_eff=cores_eff,
        simd=tuple(simd),
        accelerator=accelerator,
        storage_free_mb=storage_free_mb,
        source=source,
        probe_version=probe_version,
        measured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        probe_lo_mb=cal.lo_bytes // MB,
        probe_hi_mb=cal.hi_bytes // MB,
        headroom_factor=headroom_factor,
    )
    logger.info(
        "probe %s: BW=%.2f GB/s, overhead=%.2f ms/tok, thermal=%.2f, free=%d MB",
        device_id, profile.bw_eff_gbps, profile.overhead_ms_per_token,
        profile.thermal_derate_180s, profile.ram_free_mb,
    )
    return profile


# =========================================================================== #
# Tier 2 — emulation (§7)
# =========================================================================== #
@dataclass
class EmulationPlan:
    """A cgroup-constrained run of the real artefact on a development machine.

    Genuinely useful and genuinely limited: **emulation catches OOM, never
    slowness.** Memory ceilings are enforceable with cgroups; memory
    *bandwidth* is not, so a run that passes here says nothing about tokens per
    second on the target.
    """

    artefact: str
    memory_max_mb: int
    cpus: str = "0-3"
    context: int = 2048

    VERIFIES = (
        "does it load at all",
        "does it OOM at the target RAM ceiling",
        "KV cache growth over long context",
        "grammar-constrained output correctness",
        "numerical output parity against the pre-package model",
    )
    CANNOT_VERIFY = (
        "achievable memory bandwidth",
        "thermal throttling",
        "NPU or GPU delegate behaviour",
        "real tokens per second",
        "battery drain",
    )

    def command(self) -> str:
        """The systemd-run invocation. Linux-only; documented rather than hidden."""
        return (
            f"systemd-run --scope -p MemoryMax={self.memory_max_mb}M "
            f"taskset -c {self.cpus} "
            f"./llama-cli -m {self.artefact} --ctx-size {self.context}"
        )


def emulation_for(profile: DeviceProfile, artefact: str, context: int = 2048) -> EmulationPlan:
    """Constrain a local run to the target's usable RAM."""
    return EmulationPlan(
        artefact=artefact, memory_max_mb=profile.usable_ram_mb,
        cpus=f"0-{max(profile.cores_perf, 4) - 1}", context=context,
    )


# =========================================================================== #
# The ladder, as a decision (§7)
# =========================================================================== #
@dataclass
class LatencyVerdict:
    """A latency claim, with the tier that licenses it."""

    tokens_per_s: float
    seconds_total: float
    tier: Tier
    may_promise: bool
    reason: str = ""
    extrapolation_reach: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "tokens_per_s": round(self.tokens_per_s, 3),
            "seconds_total": round(self.seconds_total, 3),
            "tier": self.tier.name.lower(),
            "may_promise": self.may_promise,
            "extrapolation_reach": round(self.extrapolation_reach, 2),
            "reason": self.reason,
        }


def roofline_bound(weight_bytes: int, bw_eff_bytes_per_s: float) -> float:
    """``tok/s <= BW_eff / |Q(W')|``. Real throughput is always at or below this.

    Thermal throttling, framework overhead and contention only subtract, which is
    what makes the bound sound for refusal and unsound for promising.
    """
    if weight_bytes <= 0 or bw_eff_bytes_per_s <= 0:
        return math.inf
    return bw_eff_bytes_per_s / weight_bytes


def assess_latency(
    profile: DeviceProfile, weight_bytes: int, tokens_out: int, prefill_s: float = 0.0
) -> LatencyVerdict:
    """Predict decode latency and state what tier the claim sits at."""
    reach = profile.calibration.extrapolation_factor(weight_bytes)
    rate = profile.tokens_per_s(weight_bytes, sustained=True)
    total = prefill_s + tokens_out / rate

    tier = profile.tier
    if tier is Tier.MEASURED and reach > 10.0:
        # §13: the affine model is good within about a decade. Beyond that a
        # measured profile no longer licenses a promise about THIS model.
        return LatencyVerdict(
            rate, total, Tier.ANALYTICAL, False, reach_reason(reach), reach,
        )
    return LatencyVerdict(
        rate, total, tier, tier.may_promise,
        "" if tier.may_promise else f"profile source is {profile.source.value}",
        reach,
    )


def reach_reason(reach: float) -> str:
    return (
        f"target is {reach:.1f}x outside the probed size bracket; the affine decode "
        "model holds within roughly one decade, so this reverts to a bound"
    )
