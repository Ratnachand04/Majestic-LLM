"""The seven resource dimensions, as pure functions (Part 4 §12-§15).

No LLM, no device, no GPU — every equation here is arithmetic over the catalogue
and a probed :class:`~modelrig.probe.DeviceProfile`, which is why §18 says to
build and test this first.

    1  memory    M          model size, context      probed
    2  storage   S          artefact + index         probed
    3  latency   l          size, INPUT LENGTH       probed + elicited budget
    4  compute   F_eff      prefill                  probed
    5  thermal   rho        sustained use            probed
    6  energy    E          battery life             probed + derived
    7  network   B          download, escalation     derived from mode

**Six of seven are probed. One budget is elicited. None is guessed.**

Two dimensions were missing entirely before this module — storage and energy —
and both bind independently of memory. A 4 GB-RAM phone with 8 GB free storage
can hold an artefact it cannot load; a device with ample RAM and a full disk
cannot install one.

.. warning::

   §13.3's correction. Latency is ``prefill + decode``, and for document-heavy
   tasks **prefill dominates**: it scales with input length, and extraction
   inputs are long. Prefill leads when

       n_in / n_out  >  tok/s_prefill / tok/s_decode   (~4-6)

   For a 1000-token lab requisition producing 80 tokens of JSON the ratio is
   12.5, so prefill is the dominant term rather than a correction to it. Costing
   decode alone understates true latency by roughly 3x — which is exactly how a
   build passes Gate 2 and then disappoints in the field.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MB = 1_000_000
GB = 1_000_000_000

#: Phones behave badly near full storage, so only this share is usable.
# SOURCE: filesystem behaviour under low free space; the same rule of thumb
# mobile app stores apply before refusing an install.
STORAGE_SAFETY = 0.80

#: Bytes per element in a float32 retrieval index.
INDEX_ELEM_BYTES_FP32 = 4
INDEX_ELEM_BYTES_INT8 = 1

#: A typical phone battery, in joules. 3500 mAh at 3.85 V nominal.
# SOURCE: 3.5 Ah * 3.85 V = 13.475 Wh = 48,510 J.
BATTERY_JOULES = 3.5 * 3.85 * 3600

#: Sustained package power under inference on a mid-range phone, in watts.
# HYPOTHESIS — the probe should measure this; 3.5 W is a placeholder standing in
# for a per-device measurement.
DEFAULT_POWER_DRAW_W = 3.5


# =========================================================================== #
# 1 — Memory (§13.1)
# =========================================================================== #
@dataclass(frozen=True)
class Memory:
    """``M = weights + KV + embedder + runtime``, in bytes."""

    weights: int
    kv_cache: int
    embedder: int
    runtime: int
    app_reserve: int = 0

    @property
    def model_total(self) -> int:
        return self.weights + self.kv_cache + self.embedder + self.runtime

    @property
    def total(self) -> int:
        return self.model_total + self.app_reserve


def weight_bytes(params: int, bits_effective: float) -> int:
    """``P * beta_eff / 8``.

    ``beta_eff`` is the EFFECTIVE width, 4.5-5.5 for a nominal 4-bit k-quant.
    Taking it as 4.0 under-counts by 15-25%.
    """
    if params <= 0 or bits_effective <= 0:
        raise ValueError("parameter count and bit width must be positive")
    return int(params * bits_effective / 8)


def kv_bytes(n_layers: int, d_kv: int, context: int, bits_kv: float = 16.0,
             batch: int = 1) -> int:
    """``2 * L * d_kv * c * beta_kv / 8``.

    ``d_kv`` is ``n_kv_heads * head_dim`` under grouped-query attention — NOT
    ``d_model``. Substituting the model dimension overestimates by the GQA group
    factor, typically 4x-8x, and causes spurious refusals.
    """
    if min(n_layers, d_kv, context, batch) <= 0:
        raise ValueError("KV geometry and context must be positive")
    return int(2 * n_layers * d_kv * context * batch * bits_kv / 8)


def memory(*, params: int, bits_effective: float, n_layers: int, d_kv: int,
           context: int, embedder: int, runtime: int, app_reserve: int = 0,
           bits_kv: float = 16.0, batch: int = 1) -> Memory:
    return Memory(
        weights=weight_bytes(params, bits_effective),
        kv_cache=kv_bytes(n_layers, d_kv, context, bits_kv, batch),
        embedder=embedder, runtime=runtime, app_reserve=app_reserve,
    )


def max_context(*, free_ram: int, params: int, bits_effective: float,
                n_layers: int, d_kv: int, embedder: int, runtime: int,
                app_reserve: int = 0, bits_kv: float = 16.0, batch: int = 1,
                ceiling: int | None = None) -> int:
    """Invert the memory equation for context — **why it is derived, never asked**.

        c_max = 8(R_free - M_app - P*beta/8 - M_emb - M_rt) / (2 L d_kv beta_kv)

    Fully determined by device and base. And it propagates: ``c_max`` caps the
    number of retrieved chunks, which constrains the retrieval design.
    """
    spare = (free_ram - app_reserve - weight_bytes(params, bits_effective)
             - embedder - runtime)
    if spare <= 0:
        return 0
    per_token = 2 * n_layers * d_kv * batch * bits_kv / 8
    c = int(spare // per_token)
    return min(c, ceiling) if ceiling else c


def memory_fits(m: Memory, free_ram: int) -> bool:
    return m.total <= free_ram


# =========================================================================== #
# 2 — Storage (§13.2)
# =========================================================================== #
@dataclass(frozen=True)
class Storage:
    """``S = artefact + index + grammar + runtime``, in bytes."""

    artefact: int
    index: int
    grammar: int
    runtime: int

    @property
    def total(self) -> int:
        return self.artefact + self.index + self.grammar + self.runtime

    @property
    def dominant(self) -> str:
        return max(
            (("artefact", self.artefact), ("index", self.index),
             ("grammar", self.grammar), ("runtime", self.runtime)),
            key=lambda kv: kv[1],
        )[0]


def index_bytes(n_chunks: int, dim: int = 768, quantised: bool = False) -> int:
    """Retrieval index size.

    **The term most often forgotten.** For a large corpus the index can exceed
    the model: 50k chunks at 768 dimensions in float32 is ~150 MB, and
    quantising it to int8 is often the cheapest storage win available — a 4x
    reduction for negligible retrieval loss.
    """
    if n_chunks < 0 or dim <= 0:
        raise ValueError("chunk count must be non-negative and dim positive")
    elem = INDEX_ELEM_BYTES_INT8 if quantised else INDEX_ELEM_BYTES_FP32
    return n_chunks * dim * elem


def storage(*, artefact: int, n_chunks: int = 0, index_dim: int = 768,
            index_quantised: bool = False, grammar: int = 64_000,
            runtime: int = 25 * MB) -> Storage:
    return Storage(
        artefact=artefact,
        index=index_bytes(n_chunks, index_dim, index_quantised),
        grammar=grammar, runtime=runtime,
    )


def storage_fits(s: Storage, free_storage: int, safety: float = STORAGE_SAFETY) -> bool:
    """``S <= phi * S_free``. Storage and memory bind independently, in both
    directions — neither predicate implies the other."""
    return s.total <= free_storage * safety


# =========================================================================== #
# 3 — Latency (§13.3) and 4 — Compute (§13.4)
# =========================================================================== #
@dataclass(frozen=True)
class Latency:
    """``l = t_prefill + t_decode``, in seconds, before the thermal derate."""

    prefill_s: float
    decode_s: float
    tokens_in: int
    tokens_out: int

    @property
    def total_s(self) -> float:
        return self.prefill_s + self.decode_s

    @property
    def prefill_share(self) -> float:
        return self.prefill_s / self.total_s if self.total_s else 0.0

    @property
    def prefill_dominates(self) -> bool:
        return self.prefill_share > 0.5

    def derated(self, thermal: float = 1.0) -> float:
        """``l / rho`` — the figure every promise must use."""
        if not 0 < thermal <= 1.0:
            raise ValueError("thermal derate must lie in (0, 1]")
        return self.total_s / thermal


def prefill_seconds(params: int, tokens_in: int, flops_eff: float) -> float:
    """``t_prefill = 2 P n_in / F_eff`` — **compute-bound**.

    Two FLOPs per parameter per token, multiply and accumulate.
    """
    if flops_eff <= 0:
        raise ValueError("effective FLOP/s must be positive")
    return 2.0 * params * tokens_in / flops_eff


def prefill_seconds_from_rate(tokens_in: int, prefill_tok_s: float) -> float:
    """Prefill from a measured token rate rather than a FLOP figure."""
    if prefill_tok_s <= 0:
        raise ValueError("prefill rate must be positive")
    return tokens_in / prefill_tok_s


def decode_seconds(weight_bytes_: int, tokens_out: int, bw_eff: float,
                   overhead_s: float = 0.0) -> float:
    """``t_decode = n_out (|Q(W')| / BW_eff + c_ovh)`` — **bandwidth-bound**."""
    if bw_eff <= 0:
        raise ValueError("effective bandwidth must be positive")
    return tokens_out * (weight_bytes_ / bw_eff + overhead_s)


def latency(*, params: int, weight_bytes_: int, tokens_in: int, tokens_out: int,
            flops_eff: float | None = None, prefill_tok_s: float | None = None,
            bw_eff: float, overhead_s: float = 0.0) -> Latency:
    """Both regimes. Supply either ``flops_eff`` or a measured ``prefill_tok_s``."""
    if prefill_tok_s is not None:
        pf = prefill_seconds_from_rate(tokens_in, prefill_tok_s)
    elif flops_eff is not None:
        pf = prefill_seconds(params, tokens_in, flops_eff)
    else:
        raise ValueError("prefill needs either flops_eff or prefill_tok_s")
    return Latency(
        prefill_s=pf,
        decode_s=decode_seconds(weight_bytes_, tokens_out, bw_eff, overhead_s),
        tokens_in=tokens_in, tokens_out=tokens_out,
    )


def prefill_crossover(prefill_tok_s: float, decode_tok_s: float) -> float:
    """The ``n_in / n_out`` ratio above which prefill dominates (~4-6)."""
    if decode_tok_s <= 0:
        raise ValueError("decode rate must be positive")
    return prefill_tok_s / decode_tok_s


def effective_flops(params: int, tokens_in: int, prefill_s: float) -> float:
    """``F_eff = 2 P n_in / t_prefill`` — **measured, never from a spec sheet**.

    Vendor TOPS figures describe NPU int8 peak and are unreachable from a CPU
    runtime, typically by 10-50x. The probe measures this by timing a prefill of
    known length, which is the only number that means anything.
    """
    if prefill_s <= 0:
        raise ValueError("prefill time must be positive")
    return 2.0 * params * tokens_in / prefill_s


def shorten_input_saving(lat: Latency, new_tokens_in: int) -> float:
    """Seconds saved by shortening the input, holding everything else fixed.

    **The derived lever.** When prefill dominates and latency fails, the fix is
    to shorten the input, not to shrink the model: truncation, pre-filtering or
    OCR region-of-interest cropping reduce ``n_in`` linearly and cost no quality
    on the task itself. Usually a far better move than dropping a base tier.
    """
    if lat.tokens_in <= 0:
        return 0.0
    scaled = lat.prefill_s * (new_tokens_in / lat.tokens_in)
    return max(0.0, lat.prefill_s - scaled)


# =========================================================================== #
# 5 — Thermal (§13.5)
# =========================================================================== #
def thermal_ratio(rate_at_t: float, rate_at_0: float) -> float:
    """``rho(t) = tok/s(t) / tok/s(0)``. Typical passive range 0.5-0.8.

    Prefill throttles harder than decode, being compute- rather than
    bandwidth-bound — so document tasks suffer twice.
    """
    if rate_at_0 <= 0:
        raise ValueError("baseline rate must be positive")
    return min(1.0, rate_at_t / rate_at_0)


# =========================================================================== #
# 6 — Energy (§13.6)
# =========================================================================== #
@dataclass(frozen=True)
class Energy:
    """Per-request energy and what it means for a working day."""

    joules_per_request: float
    battery_joules: float
    requests_per_charge: float

    def battery_share(self, daily_volume: int) -> float:
        """Fraction of a full charge a day's work consumes."""
        if self.requests_per_charge <= 0:
            return float("inf")
        return daily_volume / self.requests_per_charge

    def as_dict(self) -> dict[str, Any]:
        return {
            "joules_per_request": round(self.joules_per_request, 1),
            "requests_per_charge": int(self.requests_per_charge),
            "battery_joules": int(self.battery_joules),
        }


def energy(latency_s: float, power_draw_w: float = DEFAULT_POWER_DRAW_W,
           battery_joules: float = BATTERY_JOULES) -> Energy:
    """``E_req = P_draw * l``, and ``N = E_batt / E_req``.

    A real operational fact that belongs on the Build Card: at 3.5 W over a 29 s
    request that is ~100 J, so ~480 requests per charge — and a front-desk tablet
    doing 200 forms a day spends roughly 40% of its battery on inference.
    """
    if latency_s <= 0 or power_draw_w <= 0:
        raise ValueError("latency and power draw must be positive")
    per_request = power_draw_w * latency_s
    return Energy(per_request, battery_joules, battery_joules / per_request)


def energy_fits(e: Energy, daily_volume: int, safety: float = 1.5) -> bool:
    """A day's volume must fit inside one charge with margin."""
    return e.requests_per_charge >= daily_volume * safety


# =========================================================================== #
# 7 — Network (§13.7)
# =========================================================================== #
@dataclass(frozen=True)
class Network:
    """Initial download and recurring escalation traffic, in bytes."""

    initial: int
    recurring_per_day: int
    offline: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "initial_mb": round(self.initial / MB, 1),
            "recurring_mb_per_day": round(self.recurring_per_day / MB, 3),
            "offline": self.offline,
        }


def network(*, artefact_bytes: int, index_bytes_: int = 0, offline: bool = False,
            escalation_rate: float = 0.0, daily_requests: int = 0,
            bytes_per_escalation: int = 8_000) -> Network:
    """Offline mode sets recurring traffic to zero **by construction**.

    That is the static proof from Fabric, not a configuration setting — which is
    why it cannot drift.
    """
    recurring = 0 if offline else int(escalation_rate * daily_requests * bytes_per_escalation)
    return Network(
        initial=artefact_bytes + index_bytes_,
        recurring_per_day=recurring,
        offline=offline,
    )


def delta_update_bytes(adapter_bytes: int, base_unchanged: bool = True,
                       artefact_bytes: int = 0) -> int:
    """What a subsequent generation actually costs to ship.

    Shipping only the ~30 MB adapter when the base is unchanged cuts the download
    by ~97%, and is the main practical argument for the separate-adapter path.
    A 1.1 GB download over an intermittent connection is a genuine deployment
    obstacle in exactly the markets where offline is required.
    """
    return adapter_bytes if base_unchanged else artefact_bytes


# =========================================================================== #
# The joint view (§14)
# =========================================================================== #
@dataclass
class ResourceEnvelope:
    """All seven dimensions for one candidate, with the binding one named."""

    memory: Memory
    storage: Storage
    latency: Latency
    energy: Energy
    network: Network
    thermal: float
    free_ram: int
    free_storage: int
    latency_budget_s: float | None = None
    daily_volume: int = 0
    violations: list[str] = field(default_factory=list)

    @property
    def fits(self) -> bool:
        return not self.violations

    @property
    def binding(self) -> str | None:
        """Which dimension is closest to its ceiling — the one to act on."""
        pressures = {
            "memory": self.memory.total / self.free_ram if self.free_ram else 0.0,
            "storage": (self.storage.total / (self.free_storage * STORAGE_SAFETY)
                        if self.free_storage else 0.0),
        }
        if self.latency_budget_s:
            pressures["latency"] = self.latency.derated(self.thermal) / self.latency_budget_s
        if self.daily_volume:
            pressures["energy"] = self.daily_volume / max(self.energy.requests_per_charge, 1e-9)
        return max(pressures, key=pressures.get) if pressures else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "memory_mb": round(self.memory.total / MB, 1),
            "storage_mb": round(self.storage.total / MB, 1),
            "latency_s": round(self.latency.derated(self.thermal), 2),
            "prefill_share": round(self.latency.prefill_share, 3),
            "energy": self.energy.as_dict(),
            "network": self.network.as_dict(),
            "binding": self.binding,
            "fits": self.fits,
            "violations": list(self.violations),
        }


def envelope(
    *, memory_: Memory, storage_: Storage, latency_: Latency, thermal: float,
    free_ram: int, free_storage: int, latency_budget_s: float | None = None,
    daily_volume: int = 0, power_draw_w: float = DEFAULT_POWER_DRAW_W,
    offline: bool = False, escalation_rate: float = 0.0,
) -> ResourceEnvelope:
    """Evaluate all seven at once, and say which one binds.

    §14: resources are not independent. Base size and context trade against each
    other in the same RAM budget, and context caps retrieved chunks which sets
    ``n_in`` which drives prefill which drives latency and energy. Retrieving
    more evidence makes the model slower, not merely better informed — so these
    must be solved jointly rather than in sequence.
    """
    derated = latency_.derated(thermal)
    e = energy(derated, power_draw_w)
    n = network(artefact_bytes=storage_.artefact, index_bytes_=storage_.index,
                offline=offline, escalation_rate=escalation_rate,
                daily_requests=daily_volume)

    violations: list[str] = []
    if not memory_fits(memory_, free_ram):
        violations.append(
            f"memory {memory_.total / MB:.0f} MB exceeds {free_ram / MB:.0f} MB free"
        )
    if not storage_fits(storage_, free_storage):
        violations.append(
            f"storage {storage_.total / MB:.0f} MB exceeds "
            f"{free_storage * STORAGE_SAFETY / MB:.0f} MB usable "
            f"({storage_.dominant} dominates)"
        )
    if latency_budget_s and derated > latency_budget_s:
        violations.append(
            f"latency {derated:.1f} s exceeds the {latency_budget_s:.1f} s budget "
            f"({'prefill' if latency_.prefill_dominates else 'decode'} dominates)"
        )
    if daily_volume and not energy_fits(e, daily_volume):
        violations.append(
            f"energy: {e.requests_per_charge:.0f} requests per charge against a "
            f"{daily_volume}/day volume"
        )

    return ResourceEnvelope(
        memory=memory_, storage=storage_, latency=latency_, energy=e, network=n,
        thermal=thermal, free_ram=free_ram, free_storage=free_storage,
        latency_budget_s=latency_budget_s, daily_volume=daily_volume,
        violations=violations,
    )
