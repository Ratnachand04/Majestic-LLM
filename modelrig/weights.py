"""Weight compilation: acquisition, merging, and the shape of the artefact (§1-§4).

The whole transformation, in one line:

    W  --train-->  dW = (alpha/r) B A  --merge-->  W' = W + dW
       --quantise-->  Q(W')  --package-->  cartridge

Stating it that plainly matters, because the mystification of this step is where
the "model that builds models" misconception lives. **No architecture is
generated, no layers are designed, nothing is invented.** Four tensor operations
and a packaging step.

Two things here are load-bearing and easy to get silently wrong:

* **There are two entirely separate quantisations** (§3). The NF4 used to fit the
  frozen base in VRAM during training is not the deployment quantisation, is not
  calibrated on anything, and must not survive to the device.
* **The merge happens in BF16** (§4). Merging into the 4-bit training base
  compounds a data-free quantisation error with the deployment one, and the model
  can lose most of the fine-tune's gains while every intermediate step reports
  success.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from majestic.logging_utils import get_logger

logger = get_logger(__name__)

MB = 1_000_000
GB = 1_000_000_000


class WeightError(RuntimeError):
    """An acquisition or merge invariant was violated."""


# =========================================================================== #
# Stage 0 — acquisition and pinning (§2)
# =========================================================================== #
@dataclass(frozen=True)
class PinnedWeights:
    """A base checkpoint, pinned by **content hash — never by name or tag**.

    This is not paranoia. Checkpoints on public hubs get updated in place, get
    renamed and get removed. If a result says "Qwen3-1.7B" and that checkpoint
    changes, the work becomes unreproducible with no error message anywhere.
    Pinning by digest is the minimum standard for a system whose output carries a
    certificate.
    """

    ref: str
    sha256: str
    size_bytes: int
    precision: str = "bf16"          # bf16 for trainable bases, int4 for teachers
    revision: str = ""               # the hub revision the digest was taken at

    def __post_init__(self) -> None:
        if len(self.sha256) != 64 or not all(c in "0123456789abcdef" for c in self.sha256):
            raise WeightError(
                f"{self.ref}: sha256 must be 64 lowercase hex characters, got {self.sha256!r}"
            )
        if self.size_bytes <= 0:
            raise WeightError(f"{self.ref}: size must be positive")

    @property
    def short(self) -> str:
        return self.sha256[:12]

    def verify(self, path: str | Path, chunk: int = 1 << 20) -> bool:
        """Re-hash a local file and compare. Used on every mirror read."""
        digest = hashlib.sha256()
        with Path(path).open("rb") as fh:
            while block := fh.read(chunk):
                digest.update(block)
        return digest.hexdigest() == self.sha256


@dataclass
class WeightMirror:
    """The local, content-addressed store of base checkpoints.

    Storage is modest and worth stating plainly: six trainable bases at BF16 plus
    one or two 4-bit teachers is roughly 30 GB, and a single device tier needs far
    less — ``android_mid`` only ever touches the 0.6B and 1.7B bases.
    """

    root: Path = field(default_factory=lambda: Path("./mirror"))
    entries: dict[str, PinnedWeights] = field(default_factory=dict)

    def pin(self, weights: PinnedWeights) -> PinnedWeights:
        existing = self.entries.get(weights.ref)
        if existing and existing.sha256 != weights.sha256:
            raise WeightError(
                f"{weights.ref} is already pinned at {existing.short} but was offered "
                f"{weights.short}. A checkpoint changing under a stable name is exactly "
                "what pinning exists to catch — bump the ref, do not overwrite the pin."
            )
        self.entries[weights.ref] = weights
        return weights

    def get(self, ref: str) -> PinnedWeights:
        if ref not in self.entries:
            raise WeightError(f"{ref} is not pinned in the mirror")
        return self.entries[ref]

    def path_for(self, ref: str) -> Path:
        """Content-addressed layout: the digest is the filename."""
        return self.root / self.get(ref).sha256[:2] / self.get(ref).sha256

    @property
    def total_bytes(self) -> int:
        return sum(w.size_bytes for w in self.entries.values())

    def footprint(self) -> dict[str, Any]:
        by_precision: dict[str, int] = {}
        for w in self.entries.values():
            by_precision[w.precision] = by_precision.get(w.precision, 0) + w.size_bytes
        return {
            "checkpoints": len(self.entries),
            "total_gb": round(self.total_bytes / GB, 2),
            "by_precision_gb": {k: round(v / GB, 2) for k, v in by_precision.items()},
        }

    def subset_for(self, refs: Iterable[str]) -> int:
        """Bytes needed to serve only these bases — the per-tier figure."""
        return sum(self.get(r).size_bytes for r in refs)


# =========================================================================== #
# Stage 1 — the two quantisations (§3)
# =========================================================================== #
class QuantRole(str, Enum):
    """Which of the two quantisations a setting refers to.

    Treating these as one step produces the single worst bug available in this
    pipeline, so they are distinct types rather than a shared string.
    """

    TRAINING = "training"        # NF4, data-free, VRAM only, must NOT reach the device
    DEPLOYMENT = "deployment"    # k-quant/AWQ/GPTQ, task-calibrated, ships


@dataclass(frozen=True)
class QuantSetting:
    """One quantisation, tagged with its role."""

    role: QuantRole
    scheme: str
    calibrated: bool = False

    def __post_init__(self) -> None:
        if self.role is QuantRole.TRAINING and self.calibrated:
            raise WeightError(
                "the training quantisation is data-free by construction; marking it "
                "calibrated confuses it with the deployment quantisation (§3)"
            )
        if self.role is QuantRole.DEPLOYMENT and not self.calibrated:
            raise WeightError(
                f"deployment quantisation {self.scheme!r} must be calibrated on the "
                "customer's task distribution — off-distribution calibration is the "
                "commonest way a fine-tune evaporates at 4-bit (§5)"
            )

    @property
    def survives_to_device(self) -> bool:
        return self.role is QuantRole.DEPLOYMENT


#: The training-side setting. Fixed, because there is nothing to choose.
NF4_TRAINING = QuantSetting(QuantRole.TRAINING, "nf4", calibrated=False)


@dataclass(frozen=True)
class AdapterSpec:
    """The low-rank correction: ``dW = (alpha / r) B A``.

    ``B`` is initialised to **zero** so ``dW = 0`` at step 0 and training begins
    exactly at the base model's behaviour. That is why LoRA training is stable and
    why it "forgets less": it starts as the identity perturbation.
    """

    rank: int
    alpha: int
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj")
    b_init: str = "zeros"
    a_init: str = "kaiming_uniform"

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise WeightError("adapter rank must be positive")
        if self.b_init != "zeros":
            raise WeightError(
                "B must be initialised to zeros so that dW = 0 at step 0; any other "
                "initialisation perturbs the base before a single gradient step (§3)"
            )

    @property
    def scaling(self) -> float:
        """``alpha / r`` — the factor the low-rank branch is multiplied by."""
        return self.alpha / self.rank

    def parameter_count(self, d_model: int, n_layers: int) -> int:
        """Trainable parameters: ``r * (d + k)`` per targeted projection."""
        return len(self.targets) * n_layers * self.rank * 2 * d_model

    def size_bytes(self, d_model: int, n_layers: int, bytes_per_param: int = 2) -> int:
        return self.parameter_count(d_model, n_layers) * bytes_per_param


# =========================================================================== #
# Stage 2-3 — the merge (§4)
# =========================================================================== #
class MergeStrategy(str, Enum):
    """Whether the adapter is folded into the base or shipped beside it."""

    MERGED = "merged"       # W' = W + dW, one artefact
    SEPARATE = "separate"   # base + adapter, shared across cartridges


@dataclass(frozen=True)
class MergePlan:
    """How the artefact is assembled, and in what precision.

    ``merge_precision`` is BF16 and cannot be anything else. Merging into the
    4-bit training base compounds a data-free quantisation error with the
    deployment quantisation, and the result can lose most of the fine-tune's
    gains while every intermediate step reports success.
    """

    strategy: MergeStrategy
    merge_precision: str = "bf16"
    reason: str = ""

    def __post_init__(self) -> None:
        if self.strategy is MergeStrategy.MERGED and self.merge_precision != "bf16":
            raise WeightError(
                f"merge precision must be bf16, not {self.merge_precision!r}. "
                "Dequantise the NF4 training base first: merging into it bakes in a "
                "data-free quantisation error that the deployment quantiser then "
                "compounds, silently (§4)."
            )


def choose_merge_strategy(
    *,
    cartridges_on_device: int,
    storage_free_bytes: int,
    base_bytes: int,
    adapter_bytes: int,
    runtime_supports_adapters: bool = True,
) -> MergePlan:
    """§4's decision rule.

    One cartridge on the device -> **merge**, because runtime support is better
    and there is nothing to share with. Multiple cartridges sharing a base ->
    **keep separate**, because that is the entire multi-cartridge memory argument:
    N merged models cost ``N x full``, N adapters cost ``base + N x 30 MB``.
    """
    if cartridges_on_device <= 1:
        return MergePlan(
            MergeStrategy.MERGED,
            reason="a single cartridge has nothing to share a base with, and merged "
                   "runtime support is more mature",
        )
    if not runtime_supports_adapters:
        return MergePlan(
            MergeStrategy.MERGED,
            reason="the target runtime cannot load adapters separately, so sharing is "
                   "unavailable however many cartridges there are",
        )

    separate_bytes = base_bytes + cartridges_on_device * adapter_bytes
    merged_bytes = cartridges_on_device * base_bytes
    if separate_bytes > storage_free_bytes:
        return MergePlan(
            MergeStrategy.MERGED,
            reason=f"even the shared layout needs {separate_bytes / GB:.2f} GB, above "
                   f"the {storage_free_bytes / GB:.2f} GB free — sharing does not help here",
        )
    return MergePlan(
        MergeStrategy.SEPARATE,
        reason=f"{cartridges_on_device} cartridges share one base: "
               f"{separate_bytes / GB:.2f} GB instead of {merged_bytes / GB:.2f} GB, "
               "and registry dedup is retained",
    )


@dataclass
class CompilationPlan:
    """The full weight-compilation recipe for one cartridge."""

    base: PinnedWeights
    adapter: AdapterSpec
    merge: MergePlan
    deployment_quant: QuantSetting
    training_quant: QuantSetting = NF4_TRAINING
    quant_format: str = "q4_k_m"

    def __post_init__(self) -> None:
        if self.deployment_quant.role is not QuantRole.DEPLOYMENT:
            raise WeightError("deployment_quant must carry the DEPLOYMENT role")
        if self.training_quant.role is not QuantRole.TRAINING:
            raise WeightError("training_quant must carry the TRAINING role")

    def stages(self) -> list[dict[str, Any]]:
        """The five arrows of §1, as an auditable ordered recipe."""
        return [
            {"stage": "acquire", "detail": f"pin {self.base.ref} at {self.base.short}",
             "precision": self.base.precision},
            {"stage": "prepare", "detail": "load base under NF4, attach adapters",
             "precision": "nf4", "note": "VRAM only; does not reach the device"},
            {"stage": "train", "detail": f"learn A,B at rank {self.adapter.rank} "
                                         f"(scaling {self.adapter.scaling:g})",
             "precision": "bf16 grads"},
            {"stage": "merge",
             "detail": ("dequantise NF4 -> BF16, then W' = W + dW"
                        if self.merge.strategy is MergeStrategy.MERGED
                        else "keep the adapter separate; quantise the base once"),
             "precision": self.merge.merge_precision},
            {"stage": "quantise",
             "detail": f"{self.quant_format} calibrated on the customer's task distribution",
             "precision": self.deployment_quant.scheme},
            {"stage": "package", "detail": "weights + compiled grammar + template + policy"},
        ]

    def audit(self) -> list[str]:
        """Invariants that must hold. Empty means the recipe is sound."""
        problems: list[str] = []
        if self.training_quant.survives_to_device:
            problems.append("the training quantisation must not reach the device")
        if not self.deployment_quant.calibrated:
            problems.append("the deployment quantisation is not task-calibrated")
        if (self.merge.strategy is MergeStrategy.MERGED
                and self.merge.merge_precision != "bf16"):
            problems.append("merge must happen in bf16, not in the 4-bit training base")
        return problems


def compilation_report(plan: CompilationPlan, d_model: int, n_layers: int) -> dict[str, Any]:
    """What this recipe produces, in numbers a Build Card can carry."""
    adapter_bytes = plan.adapter.size_bytes(d_model, n_layers)
    return {
        "base_ref": plan.base.ref,
        "base_sha256": plan.base.short,
        "adapter_params": plan.adapter.parameter_count(d_model, n_layers),
        "adapter_mb": round(adapter_bytes / MB, 1),
        "strategy": plan.merge.strategy.value,
        "strategy_reason": plan.merge.reason,
        "merge_precision": plan.merge.merge_precision,
        "quant_format": plan.quant_format,
        "training_quant": plan.training_quant.scheme,
        "training_quant_ships": plan.training_quant.survives_to_device,
        "audit": plan.audit(),
    }


def save_manifest(report: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return p
