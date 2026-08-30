"""The Trainer's arithmetic — and its deliberate absence of opinions (Part 9).

    train : Plan x Dataset -> AdapterRef

Every decision — base, rank, method, learning rate, epochs, replay fraction,
preference stage — **arrives pre-made in the Build Plan**. The Trainer selects
nothing. It loads what it is told, trains what it is told, returns an artefact.

That is deliberate, and it is what makes the system verifiable. A choice made
inside a GPU job is unauditable: it happens after money is spent, with no
predicate having validated it. Pushing every decision upstream is what lets a
build be refused in eight seconds instead of failing in ninety minutes.

**A subsystem with no opinions has no bugs of judgement.** It can still have bugs
of execution, which is what :mod:`modelrig.preflight` is for.

**Zero novelty, and that is the correct outcome.** Low-rank adaptation is Hu et
al. 2021, 4-bit training is Dettmers et al. 2023, LoRA+ is Hayou et al. 2024,
the preference methods are Rafailov / Hong / Ethayarajh, and the implementation
is peft and TRL. Writing a custom training loop would add risk, subtract
reliability, and contribute no defensible claim. Saying so plainly is the
cheapest credibility available.

What this module contains is the arithmetic the *Planner* needs to make those
decisions — sizes, memory, FLOPs and cost — none of which was derived anywhere
in the series despite being quoted throughout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from majestic.logging_utils import get_logger

logger = get_logger(__name__)

GB = 1_000_000_000
MB = 1_000_000

#: Bytes per parameter under mixed-precision AdamW: bf16 weights + bf16 grads +
#: fp32 master + Adam m + Adam v.
FULL_FT_BYTES_PER_PARAM = 16

#: §6 — NF4 with block scales is about 4.5 bits, not 4.0. The same correction as
#: the deployment-side effective width.
NF4_BITS_EFFECTIVE = 4.5

#: §7 — gradient checkpointing trades compute for activation memory.
CHECKPOINT_TIME_FACTOR = 1.3

#: §9 — LoRA skips backward-to-weights for the frozen base, so 4PN not 6PN.
LORA_FLOPS_PER_PARAM_TOKEN = 4
FULL_FT_FLOPS_PER_PARAM_TOKEN = 6

#: §9 — model FLOP utilisation for small models on an A100.
# SOURCE: 0.3-0.5 observed range; the low end is the honest planning figure.
MFU = 0.4
A100_PEAK_FLOPS = 312e12
A100_USD_PER_HOUR = 2.0

#: §11 — LoRA+ ratio. B starts at zero and must travel; A starts at scale and
#: must not, so they should not share a learning rate.
# SOURCE: Hayou et al. 2024. A one-line change for a free convergence gain.
LORA_PLUS_LAMBDA = 16.0


# =========================================================================== #
# §2 — the parameterisation, and where "~30 MB" comes from
# =========================================================================== #
@dataclass(frozen=True)
class Geometry:
    """The shape numbers an adapter size depends on."""

    d_model: int
    n_layers: int
    d_ffn: int
    d_kv: int          # n_kv_heads * head_dim under GQA, NOT d_model

    def per_layer_fan(self) -> int:
        """``sum over targets of (d_in + d_out)`` for the standard target set."""
        return (
            (self.d_model + self.d_model)          # q_proj
            + 2 * (self.d_model + self.d_kv)       # k_proj, v_proj
            + (self.d_model + self.d_model)        # o_proj
            + 3 * (self.d_model + self.d_ffn)      # gate, up, down
        )


#: The 1.7B-class geometry the series has been quoting adapter sizes for.
QWEN_1_7B = Geometry(d_model=2048, n_layers=28, d_ffn=5504, d_kv=1024)


def trainable_parameters(rank: int, geometry: Geometry = QWEN_1_7B) -> int:
    """``P_train(r) = r * L * sum(d_in + d_out)`` over the target modules."""
    if rank < 1:
        raise ValueError("rank must be at least 1")
    return rank * geometry.n_layers * geometry.per_layer_fan()


def adapter_bytes(rank: int, geometry: Geometry = QWEN_1_7B,
                  bytes_per_param: int = 2) -> int:
    """Adapter size on disk at fp16.

    **This is where the ~30 MB figure used throughout the series comes from:
    r = 16 on a 1.7B base.** It was quoted for eight documents without
    derivation; this is the derivation.
    """
    return trainable_parameters(rank, geometry) * bytes_per_param


def compression_ratio(rank: int, d_in: int, d_out: int) -> float:
    """``r(d+k) / dk`` — trainable share for one adapted matrix."""
    return rank * (d_in + d_out) / (d_in * d_out)


# =========================================================================== #
# §5-§7 — memory, the binding constraint
# =========================================================================== #
def full_finetune_bytes(params: int) -> int:
    """``~16P``. For a 1.7B base that is 27 GB before activations — it does not
    fit a 24 GB card, which is the entire reason QLoRA exists here."""
    return params * FULL_FT_BYTES_PER_PARAM


def qlora_bytes(params: int, rank: int, geometry: Geometry = QWEN_1_7B,
                activation_bytes: int = 0,
                bits_effective: float = NF4_BITS_EFFECTIVE) -> int:
    """Frozen 4-bit base plus adapter-sized optimiser state.

    The optimiser term collapses because Adam states scale with *trainable*
    parameters, which are about 1% of the total. That is the whole trick, and it
    is the difference between "needs an A100" and "runs on a laptop GPU".
    """
    frozen = int(params * bits_effective / 8)
    trainable = full_finetune_bytes(trainable_parameters(rank, geometry))
    return frozen + trainable + activation_bytes


#: Bytes of activation stored per layer per token, as a multiple of d_model.
# SOURCE: bf16 residual stream + QKV projections + MLP intermediate, i.e.
#   2 * (d + 3d + 2 * d_ffn) / d  ~=  2 * (1 + 3 + 2 * 2.7)  ~=  19
# for the 1.7B geometry. A derived figure rather than a round number, because
# this term is what decides whether a batch fits.
ACTIVATION_CONSTANT = 19.0


def activation_bytes(batch: int, seq_len: int, geometry: Geometry = QWEN_1_7B,
                     per_layer_constant: float = ACTIVATION_CONSTANT,
                     checkpointing: bool = True) -> int:
    """``c * b * s * L * d``, or ``c * b * s * sqrt(L) * d`` with checkpointing.

    Roughly 5x less activation memory for about 30% more compute. Majestic's
    builds are compute-cheap and memory-constrained, so checkpointing should be
    on by default.

    §6's quoted 2-4 GB total assumes a real training batch. At ``batch=1`` the
    activation term is small and the total lands near 1.5 GB; the band the spec
    quotes is reached around ``batch=4`` at 2k context.
    """
    depth = math.sqrt(geometry.n_layers) if checkpointing else geometry.n_layers
    return int(per_layer_constant * batch * seq_len * depth * geometry.d_model)


@dataclass(frozen=True)
class MemoryPlan:
    frozen_base: int
    adapter_and_optimiser: int
    activations: int
    full_finetune_equivalent: int

    @property
    def total(self) -> int:
        return self.frozen_base + self.adapter_and_optimiser + self.activations

    @property
    def reduction(self) -> float:
        return self.full_finetune_equivalent / self.total if self.total else 0.0

    def fits(self, vram_bytes: int) -> bool:
        return self.total <= vram_bytes

    def as_dict(self) -> dict[str, Any]:
        return {
            "frozen_base_gb": round(self.frozen_base / GB, 2),
            "adapter_optimiser_gb": round(self.adapter_and_optimiser / GB, 3),
            "activations_gb": round(self.activations / GB, 2),
            "total_gb": round(self.total / GB, 2),
            "full_finetune_gb": round(self.full_finetune_equivalent / GB, 1),
            "reduction": round(self.reduction, 1),
        }


def memory_plan(params: int, rank: int, *, batch: int = 1, seq_len: int = 2048,
                geometry: Geometry = QWEN_1_7B,
                checkpointing: bool = True) -> MemoryPlan:
    acts = activation_bytes(batch, seq_len, geometry, checkpointing=checkpointing)
    return MemoryPlan(
        frozen_base=int(params * NF4_BITS_EFFECTIVE / 8),
        adapter_and_optimiser=full_finetune_bytes(trainable_parameters(rank, geometry)),
        activations=acts,
        full_finetune_equivalent=full_finetune_bytes(params) + acts,
    )


# =========================================================================== #
# §8 — the batch-size identity
# =========================================================================== #
def effective_batch(micro_batch: int, grad_accum: int, devices: int = 1) -> int:
    """``b_eff = b_micro * accum * devices``.

    Gradient accumulation buys effective batch size at no memory cost — but it
    is not *exactly* a larger batch, and the difference bites on variable-length
    inputs. See :func:`modelrig.preflight.normalise_by_tokens`.
    """
    if min(micro_batch, grad_accum, devices) < 1:
        raise ValueError("batch factors must be at least 1")
    return micro_batch * grad_accum * devices


# =========================================================================== #
# §9-§10 — compute and cost
# =========================================================================== #
def training_flops(params: int, tokens: int, full_finetune: bool = False) -> float:
    """``4PN`` for LoRA, ``6PN`` for full fine-tuning.

    LoRA skips backward-to-weights for the frozen base — those gradients are
    never materialised — which removes one of the three ``2P`` terms.
    """
    per = FULL_FT_FLOPS_PER_PARAM_TOKEN if full_finetune else LORA_FLOPS_PER_PARAM_TOKEN
    return per * params * tokens


@dataclass(frozen=True)
class TrainingCost:
    tokens: int
    flops: float
    seconds: float
    usd: float

    @property
    def hours(self) -> float:
        """Wall clock, which is the binding constraint rather than dollars."""
        return self.seconds / 3600

    def as_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "flops": f"{self.flops:.2e}",
            "seconds": round(self.seconds, 1),
            "hours": round(self.hours, 2),
            "usd": round(self.usd, 2),
        }


def training_cost(params: int, examples: int, mean_tokens: int = 500,
                  epochs: int = 3, *, full_finetune: bool = False,
                  mfu: float = MFU, usd_per_hour: float = A100_USD_PER_HOUR,
                  checkpointing: bool = True) -> TrainingCost:
    """What a training run actually costs.

    .. note::

       **§10's correction.** This series quoted "$5-40 for training" throughout.
       An augmentation-dominated extraction build — 2,100 examples at ~500
       tokens, three epochs on a 1.7B base — is about **$0.10**, wrong at the
       low end by a factor of roughly fifty. A synthesis-heavy 4B build over
       60k examples is about $6.40.

       Training cost spans two orders of magnitude, and which end a build lands
       at is decided by the Data Factory's operation mix rather than by anything
       the Trainer does.
    """
    tokens = examples * mean_tokens * epochs
    flops = training_flops(params, tokens, full_finetune)
    throughput = A100_PEAK_FLOPS * mfu
    seconds = flops / throughput
    if checkpointing:
        seconds *= CHECKPOINT_TIME_FACTOR
    return TrainingCost(tokens, flops, seconds, seconds / 3600 * usd_per_hour)


def build_cost_breakdown(
    *, data_usd: float, train_usd: float, eval_usd: float = 0.20,
    quantise_usd: float = 0.15,
) -> dict[str, Any]:
    """The whole build, so the two corrections can be seen compounding.

    Part 8's teacher-cost correction and §10's training correction move an
    extraction build with labelled seeds from about $40 to **under a dollar** —
    which changes the research budget, the pricing model and the viability of a
    bring-your-own-compute path simultaneously.
    """
    total = data_usd + train_usd + eval_usd + quantise_usd
    return {
        "data": round(data_usd, 2),
        "train": round(train_usd, 2),
        "eval": round(eval_usd, 2),
        "quantise": round(quantise_usd, 2),
        "total": round(total, 2),
        "under_one_dollar": total < 1.0,
    }


# =========================================================================== #
# §12 — hedge and sweep are different things
# =========================================================================== #
class CandidateMode(str, Enum):
    """Why more than one candidate is being built.

    Part 2 §15 proved parallel candidates are strictly worse in expected cost,
    ratio ``2/(2-p) > 1``. **That result governs redundancy — the same plan run
    twice hoping one passes.** It does not govern a hyperparameter sweep, where
    the candidates differ systematically and the Proving Ground performs *model
    selection* rather than hedging.

    Redundancy buys a second draw from one distribution. Search buys the maximum
    over a set of different ones, and ``E[max_i q_i] > E[q]`` strictly whenever
    the ``q_i`` differ. Conflating them is why one flag was governing two
    opposite decisions.
    """

    NONE = "none"
    HEDGE = "hedge"      # identical plans; §15 applies. Off by default.
    SWEEP = "sweep"      # systematically varied; model selection. Cheap enough to default on.

    @property
    def default_enabled(self) -> bool:
        return self is CandidateMode.SWEEP

    @property
    def rationale(self) -> str:
        return {
            "none": "a single plan",
            "hedge": ("redundant candidates buy wall-clock time and variance "
                      "reduction, never expected cost — the customer's trade to make"),
            "sweep": ("systematically varied candidates buy the maximum over "
                      "different distributions, which is model selection"),
        }[self.value]


def sweep_cost(params: int, examples: int, ranks: tuple[int, ...] = (8, 16, 32, 64),
               **kw: Any) -> dict[str, Any]:
    """What a rank sweep costs — which is the argument for doing one.

    At about ten cents a run, guessing the rank is a false economy.
    """
    per_run = training_cost(params, examples, **kw)
    return {
        "ranks": list(ranks),
        "runs": len(ranks),
        "usd_per_run": round(per_run.usd, 2),
        "usd_total": round(per_run.usd * len(ranks), 2),
        "adapter_sizes_mb": [round(adapter_bytes(r) / MB, 1) for r in ranks],
        "why": (
            "a sweep is model selection, not hedging: E[max q_i] > E[q] strictly "
            "when the candidates differ, and Part 2 §15's negative result does "
            "not apply"
        ),
    }


# =========================================================================== #
# §11, §16-§17 — the recipe
# =========================================================================== #
class PreferenceMethod(str, Enum):
    """Which preference stage, and what it costs.

    KTO is named specifically in the invariants rather than "a preference
    method", and §17 is the reason: field feedback is a thumbs-up or a
    thumbs-down on **one** output. It is not a pair. Building DPO pairs from it
    means either generating a counterfactual or discarding unmatched signal, and
    both are lossy. **KTO consumes exactly what the flywheel produces, with no
    transformation.**
    """

    NONE = "none"
    DPO = "dpo"      # paired; needs a frozen reference model
    ORPO = "orpo"    # reference-free, single stage
    KTO = "kto"      # unpaired binary labels — the flywheel's shape

    @property
    def needs_reference_model(self) -> bool:
        """A reference model doubles resident memory, which matters given §6."""
        return self in (PreferenceMethod.DPO, PreferenceMethod.KTO)

    @property
    def accepts_unpaired(self) -> bool:
        return self is PreferenceMethod.KTO


def preference_for(has_pairs: bool, from_field_feedback: bool) -> PreferenceMethod:
    """Choose by the shape of the data, not by preference.

    ORPO is the natural default for cost-sensitive builds — half the memory and
    one stage instead of two — but that is a hypothesis to A/B in the Proving
    Ground rather than an assumption to bake in.
    """
    if from_field_feedback:
        return PreferenceMethod.KTO
    if has_pairs:
        return PreferenceMethod.ORPO
    return PreferenceMethod.NONE


def kto_weights(desirable: int, undesirable: int) -> dict[str, float]:
    """``lambda_y`` set from the observed class ratio, not left at defaults.

    Field feedback skews negative — people report failures, not successes — so
    defaults tuned on balanced data will over-weight the complaints.
    """
    total = desirable + undesirable
    if total == 0:
        return {"desirable": 1.0, "undesirable": 1.0, "balanced": True}
    p_desirable = desirable / total
    return {
        "desirable": round(1.0 / max(p_desirable, 1e-6), 3),
        "undesirable": round(1.0 / max(1.0 - p_desirable, 1e-6), 3),
        "observed_desirable_share": round(p_desirable, 4),
        "balanced": abs(p_desirable - 0.5) < 0.1,
    }


def learning_rates(base_lr: float = 2e-4,
                   lam: float = LORA_PLUS_LAMBDA) -> dict[str, float]:
    """LoRA+: ``eta_B = lambda * eta_A``.

    The asymmetry follows from the initialisation. ``B`` starts at zero and has
    to travel; ``A`` starts at scale and should not. Sharing one rate holds
    ``B`` back for no reason.
    """
    return {"lr_A": base_lr, "lr_B": base_lr * lam, "lambda": lam}


def alpha_over_rank(alpha: float, rank: int) -> float:
    """Only the ratio matters (§4), so the plan should record it rather than
    ``alpha`` — which makes a rank change safe without retuning."""
    if rank < 1:
        raise ValueError("rank must be at least 1")
    return alpha / rank


__all__ = [
    "A100_PEAK_FLOPS", "A100_USD_PER_HOUR", "CHECKPOINT_TIME_FACTOR",
    "FULL_FT_BYTES_PER_PARAM", "LORA_PLUS_LAMBDA", "MFU", "NF4_BITS_EFFECTIVE",
    "QWEN_1_7B",
    "CandidateMode", "Geometry", "MemoryPlan", "PreferenceMethod", "TrainingCost",
    "activation_bytes", "adapter_bytes", "alpha_over_rank", "build_cost_breakdown",
    "compression_ratio", "effective_batch", "full_finetune_bytes", "kto_weights",
    "learning_rates", "memory_plan", "preference_for", "qlora_bytes", "sweep_cost",
    "trainable_parameters", "training_cost", "training_flops",
]
