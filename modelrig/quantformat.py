"""Selecting the quantisation FORMAT from the device's instruction set (§5, §10).

The bit-width is not the whole choice. On ARM cores with ``dotprod`` or ``i8mm``,
some k-quant variants run substantially faster than others **at identical
bit-width**, because their block structure maps onto the available SIMD
instructions. So the quantiser is selected from the device's instruction-set
profile, not from a global default.

This is a real device-specificity axis that "how many bits" framing hides, and it
is why ``P_ram`` and ``P_lat`` can both pass while the artefact still runs at half
the achievable speed.

.. warning::

   §13 is explicit that **the mapping below is empirical folklore, not a
   published table.** It is encoded here as a falsifiable hypothesis with an
   explicit confidence, not as fact. The probe already runs two models on the
   device, so extending it to sweep formats is nearly free — and
   :func:`measured_ranking` exists so a real measurement overrides the guess the
   moment one exists. Until then every selection carries
   ``evidence="hypothesis"``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from majestic.logging_utils import get_logger

logger = get_logger(__name__)


#: SIMD capabilities a probe can report, roughly in order of usefulness for
#: low-bit matrix multiply.
KNOWN_SIMD = ("neon", "dotprod", "i8mm", "sve", "sve2", "avx2", "avx512", "amx")


@dataclass(frozen=True)
class FormatCandidate:
    """One deployment format, with what it needs and what it is thought to give.

    ``speed_hint`` is a RELATIVE hypothesis, not a measurement: 1.0 is the
    baseline k-quant on plain NEON. It orders candidates; it does not predict
    throughput. Anything that predicts throughput comes from the probe.
    """

    name: str
    bits_effective: float
    requires: tuple[str, ...] = ()
    prefers: tuple[str, ...] = ()
    speed_hint: float = 1.0
    accelerators: tuple[str, ...] = ("cpu",)
    note: str = ""

    def supported_by(self, simd: Sequence[str], accelerator: str) -> bool:
        have = {s.lower() for s in simd}
        if accelerator not in self.accelerators:
            return False
        return all(req in have for req in self.requires)

    def affinity(self, simd: Sequence[str]) -> int:
        """How many preferred instruction sets this device actually has."""
        have = {s.lower() for s in simd}
        return sum(1 for p in self.prefers if p in have)


#: The hypothesis table.
# HYPOTHESIS — every speed_hint below is folklore, not a published benchmark.
# The probe should measure these (§13); until it does, they only ORDER
# candidates and never appear in a latency promise.
_FORMATS: tuple[FormatCandidate, ...] = (
    FormatCandidate(
        "q4_k_m", bits_effective=5.2, prefers=("dotprod",), speed_hint=1.00,
        note="the baseline k-quant; super-block structure suits plain NEON",
    ),
    FormatCandidate(
        "q4_0_4_8", bits_effective=4.6, requires=("i8mm",), prefers=("i8mm",),
        speed_hint=1.60,
        note="weights pre-shuffled for the i8mm matmul path; substantially faster "
             "at the same bit-width, and simply unavailable without i8mm",
    ),
    FormatCandidate(
        "q4_0_4_4", bits_effective=4.6, requires=("dotprod",), prefers=("dotprod",),
        speed_hint=1.30,
        note="pre-shuffled for the sdot path; the dotprod-only fallback",
    ),
    FormatCandidate(
        "awq_int4", bits_effective=4.6, prefers=("avx2", "avx512"), speed_hint=1.10,
        accelerators=("cpu", "gpu"),
        note="activation-aware scaling; strongest where a GPU kernel exists",
    ),
    FormatCandidate(
        "gptq_int4", bits_effective=4.6, prefers=("avx2",), speed_hint=1.05,
        accelerators=("cpu", "gpu"),
        note="Hessian-based rounding",
    ),
    FormatCandidate(
        "spqr_int4", bits_effective=5.6, speed_hint=0.85,
        note="preserves outliers; slower and larger, the escalation rung when "
             "answer-flip fails above ~6.7B",
    ),
    FormatCandidate(
        "int8", bits_effective=8.5, prefers=("dotprod", "avx2"), speed_hint=0.60,
        accelerators=("cpu", "gpu", "npu"),
        note="widest support, but twice the bytes to move per token",
    ),
)

FORMATS: dict[str, FormatCandidate] = {f.name: f for f in _FORMATS}


@dataclass
class FormatChoice:
    """A selected format, with the evidence behind it stated plainly."""

    name: str
    bits_effective: float
    evidence: str = "hypothesis"     # hypothesis | measured
    reason: str = ""
    rejected: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)

    @property
    def measured(self) -> bool:
        return self.evidence == "measured"

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": self.name,
            "bits_effective": self.bits_effective,
            "evidence": self.evidence,
            "reason": self.reason,
            "alternatives": list(self.alternatives),
            "rejected": list(self.rejected),
        }


def supported_formats(
    simd: Sequence[str], accelerator: str = "cpu", allowed: Iterable[str] | None = None
) -> list[FormatCandidate]:
    """Formats this device can actually run, best hypothesised first."""
    pool = [FORMATS[n] for n in allowed] if allowed else list(_FORMATS)
    usable = [f for f in pool if f.supported_by(simd, accelerator)]
    return sorted(usable, key=lambda f: (-f.speed_hint, -f.affinity(simd), f.name))


def select_format(
    simd: Sequence[str],
    accelerator: str = "cpu",
    *,
    allowed: Iterable[str] | None = None,
    measured: dict[str, float] | None = None,
    max_bits: float | None = None,
) -> FormatChoice:
    """Choose the deployment format for this device.

    ``measured`` is a ``{format: tokens_per_s}`` map from a probe format sweep.
    When present it **overrides the hypothesis table entirely** and the choice is
    marked ``evidence="measured"`` — which is the only way a format claim ever
    earns that label.

    ``max_bits`` caps effective width when the memory predicate has already
    established that a wider format will not fit.
    """
    candidates = supported_formats(simd, accelerator, allowed)
    if max_bits is not None:
        candidates = [f for f in candidates if f.bits_effective <= max_bits]
    if not candidates:
        raise ValueError(
            f"no deployment format is supported on accelerator {accelerator!r} "
            f"with SIMD {list(simd)} under a {max_bits} bit ceiling"
        )

    rejected = [
        f"{f.name} needs {'+'.join(f.requires)}"
        for f in _FORMATS
        if not f.supported_by(simd, accelerator) and f.requires
    ]

    if measured:
        ranked = [f for f in candidates if f.name in measured]
        if ranked:
            best = max(ranked, key=lambda f: measured[f.name])
            return FormatChoice(
                name=best.name, bits_effective=best.bits_effective, evidence="measured",
                reason=(
                    f"fastest of {len(ranked)} formats swept on this device "
                    f"({measured[best.name]:.1f} tok/s)"
                ),
                rejected=rejected,
                alternatives=[f.name for f in ranked if f.name != best.name],
            )

    best = candidates[0]
    return FormatChoice(
        name=best.name, bits_effective=best.bits_effective, evidence="hypothesis",
        reason=(
            f"{best.note}. Selected from SIMD {sorted({s.lower() for s in simd})} "
            "by the hypothesis table — NOT measured on this device (§13)."
        ),
        rejected=rejected,
        alternatives=[f.name for f in candidates[1:3]],
    )


def measured_ranking(sweep: dict[str, float]) -> list[tuple[str, float]]:
    """Order a probe's format sweep, fastest first.

    The probe already runs two models to calibrate bandwidth; sweeping formats at
    the smaller size costs almost nothing extra and turns the folklore above into
    a table worth publishing.
    """
    unknown = set(sweep) - set(FORMATS)
    if unknown:
        raise ValueError(f"sweep contains unknown formats: {sorted(unknown)}")
    return sorted(sweep.items(), key=lambda kv: -kv[1])


def sweep_plan(simd: Sequence[str], accelerator: str = "cpu") -> dict[str, Any]:
    """What a probe should measure to replace the hypothesis on this device."""
    candidates = supported_formats(simd, accelerator)
    return {
        "formats": [f.name for f in candidates],
        "why": (
            "the SIMD-to-fastest-format mapping is empirical folklore; sweeping it "
            "here costs one extra short run and produces a publishable table"
        ),
        "current_evidence": "hypothesis",
    }


# =========================================================================== #
# The accelerator decides the container, not the device's name (§10)
# =========================================================================== #
#: Which runtime container each accelerator can actually load. Ordered by
#: maturity on that accelerator, best first.
#
# The distinction that makes this worth a table: the *quantiser* is chosen from
# the SIMD flags (above) and the *container* is chosen from the accelerator.
# They are different questions with different answers, and collapsing them is
# how a plan ends up shipping GGUF to an NPU that cannot load it.
_TARGETS_BY_ACCELERATOR: dict[str, tuple[str, ...]] = {
    "cpu": ("gguf", "onnx", "executorch"),
    "gpu": ("onnx", "gguf", "vllm"),
    "npu": ("executorch", "coreml", "onnx"),
    "ane": ("coreml", "executorch"),
}

#: Containers that need a server process and therefore cannot run offline
#: on-device. Kept explicit so the offline filter is a lookup, not a guess.
_SERVER_ONLY_TARGETS = frozenset({"vllm"})


def target_formats_for(
    accelerator: str = "cpu", *, offline: bool = False,
    allowed: Iterable[str] | None = None,
) -> list[str]:
    """Runtime containers this accelerator can load, best-supported first.

    §10 lists ``accelerator -> target format`` as one of six plan coordinates the
    device decides. Selecting it from the device's *name* — ``android_*`` implies
    GGUF — is the guess the probe exists to replace: two devices in the same
    class can differ in whether an NPU delegate is usable at all.
    """
    pool = _TARGETS_BY_ACCELERATOR.get(accelerator.lower())
    if pool is None:
        # An accelerator nobody has mapped is not a licence to invent one. Fall
        # back to the CPU containers, which every target can load.
        logger.warning(
            "quantformat: accelerator %r is not in the target table; falling back to "
            "CPU containers, which is conservative rather than correct", accelerator,
        )
        pool = _TARGETS_BY_ACCELERATOR["cpu"]
    if offline:
        pool = tuple(t for t in pool if t not in _SERVER_ONLY_TARGETS)
    if allowed is not None:
        keep = set(allowed)
        pool = tuple(t for t in pool if t in keep)
    return list(pool)


def target_for(
    accelerator: str = "cpu", *, offline: bool = False,
    allowed: Iterable[str] | None = None,
) -> str:
    """The single best container for this accelerator."""
    options = target_formats_for(accelerator, offline=offline, allowed=allowed)
    if not options:
        raise ValueError(
            f"no runtime container is loadable on accelerator {accelerator!r}"
            + (" under an offline requirement" if offline else "")
        )
    return options[0]
