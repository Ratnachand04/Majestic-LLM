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
)

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

# Bytes per parameter by bit width. 4-bit lands at ~0.55 GB per billion
# parameters once packing overhead is counted (A-01).
BYTES_PER_PARAM: dict[str, float] = {
    "int4": 0.55, "int8": 1.0, "fp16": 2.0, "fp32": 4.0, "none": 2.0,
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

    def bases_for(self, primitive: TaskPrimitive) -> list[BaseModel]:
        """Bases that support a primitive, smallest first (cheapest sufficient)."""
        return sorted(
            (b for b in self.bases if primitive in b.good_for),
            key=lambda b: b.params_b,
        )

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
