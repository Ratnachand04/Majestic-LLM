"""Build cost estimation (§3.7).

    C(pi) = kappa_gen * N_syn * mean_len          teacher generation
          + kappa_gpu * N_b * N_tok * zeta(m) / Phi_eff    training
          + kappa_eval * |E|                       evaluation

**Money is integer micro-USD throughout.** Floating-point money in a billing path
is a defect waiting to happen: 0.1 + 0.2 != 0.3, and a planner that compares a
float cost against a float ceiling will eventually admit a build one cent over
budget or refuse one a cent under. Every public function here returns ``int``.

Every constant carries a ``# SOURCE:`` comment. A cost model whose constants have
no provenance is a random number generator with units attached, and the budget
predicate is only as trustworthy as the worst-sourced number in it.
"""
from __future__ import annotations

from dataclasses import dataclass

from modelrig.planner.catalog import ModelSpec

#: One US dollar, in micro-USD. All money is integer multiples of this unit.
USD = 1_000_000


# --- rate constants --------------------------------------------------------- #
# SOURCE: rented A100-80GB spot pricing, mid-2025, averaged across three
# providers (~$1.80/hr). Expressed per GPU-second in micro-USD.
KAPPA_GPU_PER_GPU_SECOND = 500          # 0.0005 USD/s == $1.80/hr

# SOURCE: open-weight teacher inference on the same rented hardware, measured as
# generation throughput for a 32B model at bf16 with vLLM (~900 output tok/s).
# Cost per generated token in micro-USD.
KAPPA_GEN_PER_TOKEN = 1                  # ~$1.00 per million generated tokens

# SOURCE: Proving Ground run cost — seven axes over a held-out set, dominated by
# judge inference. Measured per evaluated example.
KAPPA_EVAL_PER_EXAMPLE = 400             # $0.0004 per example

# SOURCE: A-03 / QLoRA 2305.14314. LoRA backpropagates only through the adapter,
# so gradient and optimiser work scales with adapter parameters, not base
# parameters. The multiplier is FLOPs relative to a full fine-tune.
ZETA: dict[str, float] = {
    "lora": 0.35,
    "dora": 0.40,      # DoRA adds a magnitude term on top of LoRA
    "qlora": 0.30,     # frozen 4-bit base: cheaper still, at some throughput cost
    "full_ft": 1.00,
}

# SOURCE: achieved (not peak) training throughput on A100-80GB, bf16, with
# gradient checkpointing — roughly 40% of the 312 TFLOP/s peak.
PHI_EFF_FLOPS = 1.25e14                  # 125 TFLOP/s achieved

# SOURCE: standard transformer training cost accounting (Kaplan et al.
# 2001.08361): forward+backward is ~6 FLOPs per parameter per token.
FLOPS_PER_PARAM_TOKEN = 6.0

#: Mean generated length per synthetic example, in tokens.
# SOURCE: B-06 data factory output — instruction plus response for an
# extraction-shaped record.
MEAN_SYNTH_TOKENS = 320


@dataclass(frozen=True)
class CostBreakdown:
    """Every term, in micro-USD, so a budget refusal can name the dominant one."""

    generation: int
    training: int
    evaluation: int

    @property
    def total(self) -> int:
        return self.generation + self.training + self.evaluation

    @property
    def dominant(self) -> str:
        return max(
            (("teacher generation", self.generation),
             ("training", self.training),
             ("evaluation", self.evaluation)),
            key=lambda kv: kv[1],
        )[0]

    def as_usd(self) -> float:
        """For display only. Never compare money in floating point."""
        return round(self.total / USD, 2)


def generation_cost(n_synthetic: int, mean_tokens: int = MEAN_SYNTH_TOKENS) -> int:
    """Cost of having the teacher generate the amplified corpus."""
    if n_synthetic < 0 or mean_tokens < 0:
        raise ValueError("generation cost inputs must be non-negative")
    return int(KAPPA_GEN_PER_TOKEN * n_synthetic * mean_tokens)


def training_cost(params: int, n_tokens: int, method: str) -> int:
    """Cost of the fine-tune itself.

    ``zeta(m)`` is the method's FLOP multiplier: roughly 0.35 for LoRA because no
    gradients flow through the frozen weights, 1.0 for a full fine-tune.
    """
    if method not in ZETA:
        raise ValueError(f"unknown PEFT method {method!r}; have {sorted(ZETA)}")
    flops = FLOPS_PER_PARAM_TOKEN * params * n_tokens * ZETA[method]
    gpu_seconds = flops / PHI_EFF_FLOPS
    return int(KAPPA_GPU_PER_GPU_SECOND * gpu_seconds)


def evaluation_cost(n_eval_examples: int) -> int:
    """Cost of running the seven-axis suite, twice (pre and post quantisation)."""
    if n_eval_examples < 0:
        raise ValueError("evaluation example count must be non-negative")
    return int(KAPPA_EVAL_PER_EXAMPLE * n_eval_examples * 2)


def estimate(
    model: ModelSpec,
    *,
    method: str,
    n_synthetic: int,
    n_train_tokens: int,
    n_eval_examples: int,
    uses_teacher: bool = True,
) -> CostBreakdown:
    """Full build cost in micro-USD.

    For MoE models only the ACTIVE parameters participate in each forward pass,
    so training cost scales with those — even though the RAM predicate must still
    account for every resident parameter (A-07).
    """
    trainable_params = model.active_params if (model.is_moe and model.active_params) else model.params
    return CostBreakdown(
        generation=generation_cost(n_synthetic) if uses_teacher else 0,
        training=training_cost(trainable_params, n_train_tokens, method),
        evaluation=evaluation_cost(n_eval_examples),
    )


def usd(micro: int) -> float:
    """Convert micro-USD to USD for display. Never for comparison."""
    return round(micro / USD, 2)
