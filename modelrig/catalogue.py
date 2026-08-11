"""The typed parts catalogue the PLANNER searches (B-05).

Six open-weight bases sized 0.6B to 8B, plus the PEFT methods, distillation
modes, quantisers and backend targets that compose with them. Flexibility is a
property of the matrix, not of any one part (B-02) — these few dozen entries
compose into millions of distinct producible models.

The catalogue is deliberately NARROW. A-04 shows that customers on different
base models cannot share a GPU, so every extra base fragments the multi-adapter
serving economics; B-08 notes the countervailing pressure that base diversity
limits blast radius. That tension is a live strategic trade, not a solved
problem.

Each base carries its KV-cache geometry, because the deployable size is bounded
by weights PLUS KV cache PLUS runtime, never weights alone (A-01).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from modelrig.licence import Licence
from modelrig.primitives import TaskPrimitive


@dataclass(frozen=True)
class BaseModel:
    """One open-weight base, with everything the planner needs to reason about it.

    ``n_layers``/``n_kv_heads``/``head_dim`` give the KV-cache growth rate, which
    is the variable nobody budgets for and the usual cause of on-device OOM.
    ``tokenizer_family`` decides whether logit distillation is even legal:
    Path 1 requires an IDENTICAL tokenizer (A-02), so the planner treats this as
    an automatic rule rather than a preference.

    ``is_moe`` / ``active_params_b`` capture the trap in A-07: sparse activation
    does NOT mean sparse residency. A Mixture-of-Experts model activates only
    ``active_params_b`` per token but must hold all ``params_b`` in memory.
    Confusing active parameters with resident parameters is a common and costly
    mistake, so the planner excludes MoE bases from memory-constrained targets.
    """

    ref: str
    params_b: float
    licence: Licence
    tokenizer_family: str
    n_layers: int
    n_kv_heads: int
    head_dim: int
    max_context: int
    good_for: tuple[TaskPrimitive, ...] = ()
    can_teach: bool = False
    is_moe: bool = False
    active_params_b: float | None = None

    @property
    def resident_params_b(self) -> float:
        """Parameters that must be RESIDENT in memory — always the full count."""
        return self.params_b

    def kv_bytes_per_token(self, bytes_per_element: float = 2.0) -> float:
        """KV-cache bytes consumed per token of context (K and V, all layers)."""
        return 2.0 * self.n_layers * self.n_kv_heads * self.head_dim * bytes_per_element


_ALL = (TaskPrimitive.EXTRACT, TaskPrimitive.CLASSIFY, TaskPrimitive.SUMMARISE,
        TaskPrimitive.REWRITE, TaskPrimitive.GENERATE, TaskPrimitive.ANSWER,
        TaskPrimitive.ROUTE, TaskPrimitive.TOOLCALL)
_SMALL = (TaskPrimitive.CLASSIFY, TaskPrimitive.ROUTE, TaskPrimitive.EXTRACT)

BASES: tuple[BaseModel, ...] = (
    BaseModel("Qwen/Qwen3-0.6B", 0.6, Licence.APACHE_2_0, "qwen", 28, 8, 128, 32768, _SMALL),
    BaseModel("Qwen/Qwen3-1.7B", 1.7, Licence.APACHE_2_0, "qwen", 28, 8, 128, 32768, _ALL),
    BaseModel("Qwen/Qwen3-4B", 4.0, Licence.APACHE_2_0, "qwen", 36, 8, 128, 32768, _ALL),
    BaseModel("Qwen/Qwen3-8B", 8.0, Licence.APACHE_2_0, "qwen", 36, 8, 128, 32768, _ALL,
              can_teach=True),
    BaseModel("HuggingFaceTB/SmolLM2-360M-Instruct", 0.36, Licence.APACHE_2_0, "smollm",
              32, 5, 64, 8192, _SMALL),
    BaseModel("meta-llama/Llama-3.2-1B-Instruct", 1.24, Licence.LLAMA_COMMUNITY, "llama",
              16, 8, 64, 131072, _ALL),
    # Present so the planner's MoE exclusion is exercised, never so it is chosen
    # on a phone: 30B resident for 3B active is exactly the trap in A-07.
    BaseModel("Qwen/Qwen3-30B-A3B", 30.5, Licence.APACHE_2_0, "qwen", 48, 4, 128, 32768,
              _ALL, can_teach=True, is_moe=True, active_params_b=3.3),
)

#: The A-01 size ladder at 4-bit: (params_b, approx GB on disk, RAM tier label).
#: Used to explain a planner decision in terms the customer can check.
SIZE_LADDER: tuple[tuple[float, float, str], ...] = (
    (0.6, 0.4, "3 GB RAM tier"),
    (1.7, 1.1, "4-6 GB RAM tier"),
    (4.0, 2.2, "8 GB tier"),
    (8.0, 4.5, "16 GB laptop only"),
)


def ladder_tier(params_b: float) -> str:
    """Which A-01 size-ladder tier a base falls into."""
    for size, _disk, tier in SIZE_LADDER:
        if params_b <= size * 1.25:
            return tier
    return "beyond the ladder — server only"

# Teachers must be open-weight and permissively licensed (A-02). A closed
# commercial API can never appear here.
TEACHERS: tuple[BaseModel, ...] = (
    BaseModel("Qwen/Qwen3-8B", 8.0, Licence.APACHE_2_0, "qwen", 36, 8, 128, 32768, _ALL, True),
    BaseModel("Qwen/Qwen3-14B", 14.0, Licence.APACHE_2_0, "qwen", 40, 8, 128, 32768, _ALL, True),
    BaseModel("Qwen/Qwen3-32B", 32.0, Licence.APACHE_2_0, "qwen", 64, 8, 128, 32768, _ALL, True),
)

PEFT_METHODS: tuple[str, ...] = ("lora", "dora", "qlora", "full_ft")
DISTIL_MODES: tuple[str, ...] = ("none", "logit_kd", "sequence_kd", "gkd")
QUANTISERS: tuple[str, ...] = ("gptq", "awq", "k_quant", "spqr", "none")
BIT_WIDTHS: tuple[str, ...] = ("int4", "int8", "fp16", "none")
TARGETS: tuple[str, ...] = ("gguf", "executorch", "coreml", "onnx", "vllm", "npz")

# Bytes per parameter by bit width, as DEPLOYED.
#
# A-01 states two int4 figures that do not reconcile: "~0.55 GB per billion
# parameters", and a size ladder whose entries (0.6B->0.4GB, 1.7B->1.1GB,
# 3-4B->2.2GB, 7-8B->4.5GB) all imply ~0.65 GB per billion. Both are correct
# about different things: 0.55 is the pure-4-bit weight cost, while the ladder
# reports real on-disk size, where mixed-precision schemes like Q4_K_M keep
# embeddings and some tensors wider than 4 bits.
#
# Planning uses the LADDER figure. Budgeting a device with the theoretical
# number would under-count every footprint by roughly 18%, which on a 4 GB
# device is the difference between fitting and an OOM in the field.
INT4_THEORETICAL_GB_PER_B = 0.55   # pure 4-bit weights, for reference only
INT4_DEPLOYED_GB_PER_B = 0.65      # observed on-disk, reconciles the size ladder

BYTES_PER_PARAM: dict[str, float] = {
    "int4": INT4_DEPLOYED_GB_PER_B, "int8": 1.0, "fp16": 2.0, "fp32": 4.0, "none": 2.0,
}


@dataclass
class Catalogue:
    """Queryable view over the parts catalogue."""

    bases: tuple[BaseModel, ...] = field(default=BASES)
    teachers: tuple[BaseModel, ...] = field(default=TEACHERS)

    def base(self, ref: str) -> BaseModel | None:
        """Look up a base by reference; ``None`` when it is off-catalogue."""
        for b in self.bases:
            if b.ref == ref:
                return b
        return None

    def teacher(self, ref: str) -> BaseModel | None:
        for t in self.teachers:
            if t.ref == ref:
                return t
        return None

    def bases_for(
        self,
        primitive: TaskPrimitive,
        *,
        largest_first: bool = True,
        allow_moe: bool = False,
        max_params_b: float | None = None,
        min_params_b: float = 0.0,
    ) -> list[BaseModel]:
        """Bases that support a primitive, ordered for the k-bit scaling law.

        **Largest first, by default.** Dettmers and Zettlemoyer (2212.09720) show
        4-bit is the optimal accuracy-per-bit point, so under a fixed RAM budget
        the correct move is the LARGEST base that fits at 4-bit — never a smaller
        base at higher precision (A-01). Ordering smallest-first would
        systematically ship weaker models than the device could carry.

        ``allow_moe`` stays False for memory-constrained targets: sparse
        activation is not sparse residency (A-07).
        """
        pool = [
            b for b in self.bases
            if primitive in b.good_for
            and (allow_moe or not b.is_moe)
            and b.params_b >= min_params_b
            and (max_params_b is None or b.params_b <= max_params_b)
        ]
        return sorted(pool, key=lambda b: b.params_b, reverse=largest_first)

    def teacher_for(self, base: BaseModel, distil_mode: str) -> BaseModel | None:
        """Pick a teacher for a base.

        Logit distillation requires an identical tokenizer, so for ``logit_kd``
        only same-family teachers are eligible (A-02, GAP-07). Sequence-level
        distillation crosses any tokenizer boundary and is the default path.
        """
        candidates = [t for t in self.teachers if t.params_b > base.params_b]
        if distil_mode == "logit_kd":
            candidates = [t for t in candidates if t.tokenizer_family == base.tokenizer_family]
        return max(candidates, key=lambda t: t.params_b) if candidates else None


DEFAULT_CATALOGUE = Catalogue()
