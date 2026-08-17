"""Typed catalogue loaders for the Planner (§10 step 1, §17 step 1).

**A malformed catalogue is a startup failure, not a runtime surprise.** Every
record is validated on load and :class:`CatalogError` is raised eagerly, because
a planner that silently plans against a half-parsed catalogue produces confident
wrong answers — the worst possible failure mode for a system whose entire claim
is that it refuses correctly.

Units are **bytes** everywhere internally. Display helpers convert with
``MB = 1e6`` (decimal), which is the convention the architecture documents use:
the A-01 size ladder and the B-09 budget table are both decimal.

The model records enrich :mod:`modelrig.catalogue` rather than duplicating it —
one source of truth for what bases exist, with the extra architecture fields the
memory and latency predicates need bolted on here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from majestic.logging_utils import get_logger
from modelrig.catalogue import BASES, TEACHERS, BaseModel
from modelrig.licence import Licence
from modelrig.primitives import TaskPrimitive

logger = get_logger(__name__)

#: Bytes per megabyte. Decimal, matching A-01's ladder and B-09's budget table.
MB = 1_000_000
GB = 1_000_000_000


class CatalogError(RuntimeError):
    """The catalogue is malformed. Raised at load time, never swallowed."""


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QuantiserSpec:
    """One quantisation scheme.

    ``bits_effective`` is the correction from §14: a "4-bit" k-quant is **not**
    4 bits per parameter. Scales, zero-points and block metadata push the real
    figure to 4.5-5.5, and using the nominal value under-counts resident weight
    memory by 15-25% — producing plans that pass ``P_ram`` and then OOM on the
    device.

    ``group_overhead`` (the ε of §3.1) is *additional* allocator and alignment
    overhead beyond the packed tensor. It defaults to 0 because
    ``bits_effective`` already accounts for scales and zero-points; setting both
    to non-zero double-counts. Only raise ε for schemes whose effective width is
    quoted as nominal.
    """

    name: str
    bits_nominal: float
    bits_effective: float
    group_overhead: float = 0.0
    preserves_outliers: bool = False

    def __post_init__(self) -> None:
        if self.bits_nominal <= 0 or self.bits_effective <= 0:
            raise CatalogError(f"quantiser {self.name!r}: bit widths must be positive")
        if self.bits_effective < self.bits_nominal:
            raise CatalogError(
                f"quantiser {self.name!r}: effective bits ({self.bits_effective}) cannot be "
                f"below nominal ({self.bits_nominal}) — metadata only ever adds"
            )
        if not 0.0 <= self.group_overhead < 1.0:
            raise CatalogError(f"quantiser {self.name!r}: group_overhead must be in [0, 1)")


@dataclass(frozen=True)
class ModelSpec:
    """A base or teacher, with the architecture fields the predicates need.

    ``n_kv_heads`` is the grouped-query count, NOT the query-head count. §14's
    second correction: substituting ``d_model`` for ``n_kv_heads * head_dim``
    overestimates the KV cache by the GQA group factor (typically 4x-8x) and
    causes spurious refusals.
    """

    ref: str
    params: int                     # exact parameter count
    n_layers: int
    n_kv_heads: int
    head_dim: int
    tokenizer: str
    licence: Licence
    max_context: int
    good_for: tuple[TaskPrimitive, ...] = ()
    can_teach: bool = False
    is_moe: bool = False
    active_params: int | None = None
    #: Optional per-model override. Small models carry a larger share of
    #: higher-precision tensors (embeddings, norms), so their effective bits per
    #: parameter run above a big model's under the same scheme.
    bits_effective_override: float | None = None

    def __post_init__(self) -> None:
        for name in ("params", "n_layers", "n_kv_heads", "head_dim", "max_context"):
            if getattr(self, name) <= 0:
                raise CatalogError(f"model {self.ref!r}: {name} must be positive")
        if self.is_moe and not self.active_params:
            raise CatalogError(f"model {self.ref!r}: MoE models must declare active_params")
        if self.active_params and self.active_params > self.params:
            raise CatalogError(
                f"model {self.ref!r}: active params exceed total — "
                "sparse activation is not sparse residency (A-07)"
            )

    @property
    def params_b(self) -> float:
        return self.params / 1e9

    @property
    def d_kv(self) -> int:
        """The GQA key/value dimension: ``n_kv_heads * head_dim`` (§14)."""
        return self.n_kv_heads * self.head_dim

    def kv_bytes_per_token(self, bytes_per_element: int = 2) -> int:
        """Exact KV bytes per token of context: ``2 * L * H_kv * d_h * p_kv``.

        The leading 2 is keys and values. This term is why grouped-query
        attention made on-device inference viable at all — with multi-head
        attention ``H_kv == H`` and the cache is four to eight times larger.
        """
        return 2 * self.n_layers * self.d_kv * bytes_per_element


@dataclass(frozen=True)
class DeviceSpec:
    """A deployment target.

    ``latency_source`` is load-bearing. ``P_lat`` **fails** when it is
    ``"unmeasured"``: effective FLOPs and bandwidth run at 30-60% of theoretical
    peak and vary with thermal state, kernel quality and quantisation format.
    A planner that interpolates them is fabricating a promise.
    """

    name: str
    free_ram: int                   # bytes the device makes available in total
    app_reserve: int = 0            # M_app: bytes the application and OS hold
    accelerator: str = "cpu"
    latency_source: str = "unmeasured"   # unmeasured | device_lab | vendor
    prefill_tok_s: float | None = None   # prompt-processing rate at reference_params
    decode_tok_s: float | None = None    # generation rate at reference_params
    #: Model size the two rates above were quoted at. Rates are meaningless
    #: without it, because both scale with model size — see :meth:`rates_for`.
    reference_params: int | None = None
    #: Quantisation the rates were measured under. Decode is bandwidth-bound, so
    #: rescaling it needs the reference WEIGHT SIZE — which depends on this. Using
    #: the candidate's own quantiser instead makes the ratio cancel and
    #: quantisation appears to buy no latency at all, which is exactly backwards.
    reference_quantiser: str = "q4_k_m"
    flops_eff: float | None = None       # achieved FLOP/s, not peak
    bandwidth_eff: float | None = None   # achieved bytes/s, not peak
    battery_wh: float = 0.0

    _SOURCES = ("unmeasured", "device_lab", "vendor")

    def rates_for(self, params: int, weight_bytes: int, ref_weight_bytes: int
                  ) -> tuple[float, float] | None:
        """Scale the reference throughput to another model, by the roofline.

        Both scalings follow from §3.2's physics rather than from curve-fitting:

        * **Prefill is compute-bound**, costing ``~2 N_b C_in`` FLOPs, so the
          achievable token rate scales as ``1 / N_b``.
        * **Decode is memory-bandwidth-bound**, reading every weight once per
          token, so its rate scales as ``1 / M_w``.

        This is why a single measured pair generalises across the catalogue at
        all — and equally why it only generalises within one device and one
        quantisation family. A device lab still has to measure each target.
        """
        if not (self.prefill_tok_s and self.decode_tok_s and self.reference_params):
            return None
        if params <= 0 or weight_bytes <= 0 or ref_weight_bytes <= 0:
            return None
        return (
            self.prefill_tok_s * (self.reference_params / params),
            self.decode_tok_s * (ref_weight_bytes / weight_bytes),
        )

    def __post_init__(self) -> None:
        if self.free_ram <= 0:
            raise CatalogError(f"device {self.name!r}: free_ram must be positive")
        if self.app_reserve < 0:
            raise CatalogError(f"device {self.name!r}: app_reserve cannot be negative")
        if self.app_reserve >= self.free_ram:
            raise CatalogError(
                f"device {self.name!r}: app_reserve ({self.app_reserve}) leaves no room "
                f"inside free_ram ({self.free_ram})"
            )
        if self.latency_source not in self._SOURCES:
            raise CatalogError(
                f"device {self.name!r}: latency_source must be one of {self._SOURCES}"
            )

    @property
    def measured(self) -> bool:
        """True only when a device lab produced the latency numbers."""
        return self.latency_source == "device_lab"

    @property
    def ram_budget(self) -> int:
        """Bytes available to the model once the application is accounted for."""
        return self.free_ram - self.app_reserve


# --------------------------------------------------------------------------- #
#: Quantisers the planner may select.
#: SOURCE: llama.cpp k-quant block layouts; bits_effective measured as
#: file_size_bytes * 8 / parameter_count over published GGUF releases.
#: The 5.2 figure also reproduces the A-01 size ladder (0.65 GB per billion).
_QUANTISERS: tuple[QuantiserSpec, ...] = (
    QuantiserSpec("q4_k_m", bits_nominal=4.0, bits_effective=5.2),
    QuantiserSpec("gptq_int4", bits_nominal=4.0, bits_effective=4.6),
    QuantiserSpec("awq_int4", bits_nominal=4.0, bits_effective=4.6),
    QuantiserSpec("spqr_int4", bits_nominal=4.0, bits_effective=5.6, preserves_outliers=True),
    QuantiserSpec("int8", bits_nominal=8.0, bits_effective=8.5),
    QuantiserSpec("fp16", bits_nominal=16.0, bits_effective=16.0),
)

#: Architecture geometry, keyed by catalogue ref.
#: SOURCE: each model's published config.json (num_hidden_layers,
#: num_key_value_heads, head_dim). Validated by modelrig.conformance.
_GEOMETRY: dict[str, dict[str, int]] = {
    "Qwen/Qwen3-0.6B":                     dict(params=596_000_000,    n_layers=28, n_kv_heads=8, head_dim=128),
    "Qwen/Qwen3-1.7B":                     dict(params=1_720_000_000,  n_layers=28, n_kv_heads=8, head_dim=128),
    "Qwen/Qwen3-4B":                       dict(params=4_022_000_000,  n_layers=36, n_kv_heads=8, head_dim=128),
    "Qwen/Qwen3-8B":                       dict(params=8_190_000_000,  n_layers=36, n_kv_heads=8, head_dim=128),
    "Qwen/Qwen3-14B":                      dict(params=14_800_000_000, n_layers=40, n_kv_heads=8, head_dim=128),
    "Qwen/Qwen3-32B":                      dict(params=32_800_000_000, n_layers=64, n_kv_heads=8, head_dim=128),
    "Qwen/Qwen3-30B-A3B":                  dict(params=30_500_000_000, n_layers=48, n_kv_heads=4, head_dim=128),
    "HuggingFaceTB/SmolLM2-360M-Instruct": dict(params=362_000_000,    n_layers=32, n_kv_heads=5, head_dim=64),
    "meta-llama/Llama-3.2-1B-Instruct":    dict(params=1_240_000_000,  n_layers=16, n_kv_heads=8, head_dim=64),
}

#: Fixed runtime terms, in bytes.
#: SOURCE: B-09 device budget table (embedder 0.09 GB, grammar+runtime 0.06 GB).
EMBEDDER_BYTES = 90 * MB
RUNTIME_BYTES = 60 * MB


@dataclass
class Catalog:
    """The typed, validated parts catalogue the Planner searches."""

    models: dict[str, ModelSpec] = field(default_factory=dict)
    quantisers: dict[str, QuantiserSpec] = field(default_factory=dict)
    devices: dict[str, DeviceSpec] = field(default_factory=dict)

    # -- lookups (raising, never returning None silently) ---------------- #
    def model(self, ref: str) -> ModelSpec:
        if ref not in self.models:
            raise CatalogError(f"unknown model {ref!r}; catalogue has {sorted(self.models)}")
        return self.models[ref]

    def quantiser(self, name: str) -> QuantiserSpec:
        if name not in self.quantisers:
            raise CatalogError(
                f"unknown quantiser {name!r}; catalogue has {sorted(self.quantisers)}"
            )
        return self.quantisers[name]

    def device(self, name: str) -> DeviceSpec:
        if name not in self.devices:
            raise CatalogError(f"unknown device {name!r}; catalogue has {sorted(self.devices)}")
        return self.devices[name]

    # -- ordered views used by the chain argument (§12) ------------------- #
    def bases(
        self, primitive: TaskPrimitive | None = None, allow_moe: bool = False
    ) -> list[ModelSpec]:
        """Deployable bases as a **chain ordered by parameter count**.

        §12 needs a total order to take a maximum over; ties are broken by ref so
        the order is total and deterministic rather than merely a preorder.
        """
        pool = [
            m for m in self.models.values()
            if not m.can_teach and (allow_moe or not m.is_moe)
            and (primitive is None or primitive in m.good_for)
        ]
        return sorted(pool, key=lambda m: (m.params, m.ref))

    def teachers(self, tokenizer: str | None = None) -> list[ModelSpec]:
        """Teachers, largest first. Filtered to one tokenizer family for logit KD."""
        pool = [
            m for m in self.models.values()
            if m.can_teach and (tokenizer is None or m.tokenizer == tokenizer)
        ]
        return sorted(pool, key=lambda m: (-m.params, m.ref))

    def bits_effective(self, model: ModelSpec, quantiser: str) -> float:
        """Effective bits per parameter for this (model, quantiser) pair (§14)."""
        if model.bits_effective_override is not None:
            return model.bits_effective_override
        return self.quantiser(quantiser).bits_effective


# --------------------------------------------------------------------------- #
def _model_from_base(base: BaseModel) -> ModelSpec:
    """Enrich a :mod:`modelrig.catalogue` entry with planner geometry."""
    geom = _GEOMETRY.get(base.ref)
    if geom is None:
        raise CatalogError(
            f"model {base.ref!r} has no published geometry on file. Add it to "
            "_GEOMETRY with a SOURCE comment — the KV-cache predicate cannot "
            "guess n_kv_heads, and guessing it silently corrupts every verdict."
        )
    return ModelSpec(
        ref=base.ref,
        params=geom["params"],
        n_layers=geom["n_layers"],
        n_kv_heads=geom["n_kv_heads"],
        head_dim=geom["head_dim"],
        tokenizer=base.tokenizer_family,
        licence=base.licence,
        max_context=base.max_context,
        good_for=base.good_for,
        can_teach=base.can_teach,
        is_moe=base.is_moe,
        active_params=int(base.active_params_b * 1e9) if base.active_params_b else None,
    )


def load_devices(path: str | Path = "configs/devices.yaml") -> dict[str, DeviceSpec]:
    """Load device profiles. Raises :class:`CatalogError` on malformed records."""
    p = Path(path)
    if not p.exists():
        raise CatalogError(f"device catalogue not found: {path}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise CatalogError(f"device catalogue is not valid YAML: {exc}") from exc

    entries = raw.get("devices")
    if not isinstance(entries, list) or not entries:
        raise CatalogError(f"{path}: expected a non-empty 'devices' list")

    out: dict[str, DeviceSpec] = {}
    for i, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise CatalogError(f"{path}: device #{i} is not a mapping")
        try:
            name = str(entry["name"])
            ram_gb = float(entry["ram_gb"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CatalogError(f"{path}: device #{i} missing name or ram_gb: {exc}") from exc
        if name in out:
            raise CatalogError(f"{path}: duplicate device {name!r}")

        usable = float(entry.get("usable_ram_fraction", 1.0))
        free = int(ram_gb * GB * usable)
        # M_app: B-09 reserves 1.40 GB of a 4 GB tablet for the application and
        # OS. Modelling it as an explicit term rather than folding it into a
        # fraction is what lets the memory breakdown be audited line by line.
        app_reserve = int(float(entry.get("app_reserve_gb", 0.0)) * GB)
        ref_params = entry.get("reference_params_b")
        out[name] = DeviceSpec(
            name=name,
            free_ram=free,
            app_reserve=app_reserve,
            accelerator=str(entry.get("accelerator", "cpu")),
            latency_source=str(
                entry.get("latency_source", "device_lab" if entry.get("measured") else "unmeasured")
            ),
            prefill_tok_s=_opt_float(entry.get("prefill_tok_s"), path, name, "prefill_tok_s"),
            decode_tok_s=_opt_float(entry.get("decode_tok_s"), path, name, "decode_tok_s"),
            reference_params=int(float(ref_params) * GB) if ref_params else None,
            reference_quantiser=str(entry.get("reference_quantiser", "q4_k_m")),
            flops_eff=_opt_float(entry.get("flops_eff"), path, name, "flops_eff"),
            bandwidth_eff=_opt_float(entry.get("bandwidth_eff"), path, name, "bandwidth_eff"),
            battery_wh=float(entry.get("battery_wh", 0.0)),
        )
        if (out[name].prefill_tok_s or out[name].decode_tok_s) and not ref_params:
            raise CatalogError(
                f"{path}: device {name!r} quotes throughput without reference_params_b. "
                "A token rate is meaningless without the model size it was measured at."
            )
    return out


def _opt_float(value: Any, path: Any, device: str, fieldname: str) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise CatalogError(f"{path}: device {device!r} field {fieldname} is not numeric") from exc
    if out <= 0:
        raise CatalogError(f"{path}: device {device!r} field {fieldname} must be positive")
    return out


def load_catalog(
    devices_path: str | Path = "configs/devices.yaml",
    models: Iterable[BaseModel] | None = None,
) -> Catalog:
    """Build and validate the whole catalogue. Fails loudly and early."""
    entries = list(models) if models is not None else [*BASES, *TEACHERS]
    model_specs: dict[str, ModelSpec] = {}
    for base in entries:
        spec = _model_from_base(base)
        model_specs.setdefault(spec.ref, spec)

    if not model_specs:
        raise CatalogError("catalogue contains no models")

    catalog = Catalog(
        models=model_specs,
        quantisers={q.name: q for q in _QUANTISERS},
        devices=load_devices(devices_path),
    )
    logger.info(
        "catalog: %d models, %d quantisers, %d devices",
        len(catalog.models), len(catalog.quantisers), len(catalog.devices),
    )
    return catalog


_DEFAULT: Catalog | None = None


def default_catalog() -> Catalog:
    """The process-wide catalogue, loaded once."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = load_catalog()
    return _DEFAULT
