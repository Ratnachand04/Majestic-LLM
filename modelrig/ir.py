"""The three-level IR stack (B-03) — the contract everything else binds to.

    Level 1  SPEC IR      what the customer wants
      lower  ->  Gate 1 spec admissibility
    Level 2  BUILD PLAN IR  how the system will make it
      lower  ->  Gate 2 plan feasibility
    Level 3  ARTEFACT IR    what actually exists on disk
             ->  Gate 3 artefact certification
    Output   CARTRIDGE — admitted to the registry

GAP-01: no published schema carries task intent, device constraints, offline
requirements, policy and licence terms together in one type-checkable artefact.
Compiler IRs describe computation, not intent; ML platforms use untyped config
files. This module IS the product architecture — everything else is downstream.

Every level is **hash-addressed**: the content hash is computed over a canonical
JSON projection, so specs are versioned, diffable and replayable, and an
identical hash serves the cached artefact at zero marginal cost (B-04, B-08).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from modelrig.licence import DataRights, Licence
from modelrig.primitives import TaskPrimitive


class AbstentionPolicy(str, Enum):
    """What the deployed model does when it is not confident."""

    FLAG = "flag"          # emit a low-confidence marker for human review
    ABSTAIN = "abstain"    # refuse to answer
    ESCALATE = "escalate"  # defer to the cloud teacher (needs offline_required=False)
    GUESS = "guess"        # answer anyway (rarely correct for regulated work)


class ProfileSource(str, Enum):
    """Where the target's device facts came from (Part 3 §14).

    This is the field that decides whether ``P_lat`` may *promise* or only
    *refuse*. Only a real measurement licenses a commitment.
    """

    PROBE = "probe"                        # the customer's actual device
    DEVICE_LAB = "device_lab"              # our own reference unit
    REFERENCE_DEVICE = "reference_device"  # another unit of the same class
    INTERPOLATED = "interpolated"          # regression over accumulated probes
    ASSUMED = "assumed"                    # a devices.yaml prior. A guess.

    @property
    def measured(self) -> bool:
        return self in (ProfileSource.PROBE, ProfileSource.DEVICE_LAB)


def _canonical(value: Any) -> Any:
    """Project a value into a deterministic, JSON-serialisable form."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _canonical(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def content_hash(obj: Any, *, exclude: tuple[str, ...] = ()) -> str:
    """Stable content hash over a canonical JSON projection."""
    data = _canonical(obj)
    if isinstance(data, dict):
        data = {k: v for k, v in data.items() if k not in exclude}
    blob = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


# --------------------------------------------------------------------------- #
@dataclass
class SpecIR:
    """LEVEL 1 — what the customer wants.

    The typed slot schema FORGE fills (B-04). Every slot carries a type and a
    validator; a specification that cannot be type-checked cannot be compiled,
    which is exactly why the interview terminates instead of wandering.
    """

    task_primitive: TaskPrimitive
    io_schema: dict[str, Any] = field(default_factory=dict)
    languages: list[str] = field(default_factory=lambda: ["en"])
    device_target: str = "laptop_cpu"
    #: Measured hardware facts from a device probe (Part 3 §8). Nullable: when
    #: absent the planner falls back to the ``devices.yaml`` prior, which is a
    #: guess and is marked as one.
    device_profile: dict[str, Any] | None = None
    #: Which rung of the verification ladder ``device_profile`` came from. The
    #: single field that decides whether a latency claim may be a promise.
    profile_source: ProfileSource = ProfileSource.ASSUMED
    #: **Derived, never elicited** — the output of the memory equation once the
    #: base is chosen. KV cache is linear in context, so on a tight device the
    #: plan does not merely pick a smaller model, it CAPS the context, which caps
    #: how many retrieved chunks the cartridge may use (§10).
    context_budget: int | None = None
    #: --- Part 4 §17: the schema gap. ``expected_input_tokens`` is a FIRST-ORDER
    #: latency driver, because prefill is compute-bound and scales with it. Its
    #: absence is how a build passes Gate 2 and then disappoints in the field:
    #: costing decode alone understates document-task latency by roughly 3x.
    #:
    #: Derive it wherever possible — tokenise the customer's sample documents and
    #: take the 95th percentile — which turns an elicited slot into a derived one
    #: and saves a question.
    expected_input_tokens: int | None = None
    #: Derivable from the size of ``io_schema``'s output schema.
    expected_output_tokens: int | None = None
    #: Needed for the energy dimension: requests per charge against daily volume.
    expected_daily_volume: int | None = None
    offline_required: bool = False
    latency_budget_ms: int | None = None
    quality_gate: float = 0.9
    seed_data_ref: str | None = None
    seed_data_count: int = 0
    abstention_policy: AbstentionPolicy = AbstentionPolicy.FLAG
    policy_rules: list[str] = field(default_factory=list)
    data_rights: DataRights = DataRights.UNKNOWN
    budget_ceiling_usd: float = 40.0
    jurisdiction: str = "IN"
    max_step_depth: int = 1
    spec_version: int = 1
    notes: str = ""

    @property
    def hash(self) -> str:
        """Content hash — the cache key for the whole build (B-04)."""
        return content_hash(self)

    def to_dict(self) -> dict[str, Any]:
        return _canonical(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpecIR:
        data = dict(data)
        # These enums mix in str, so a member is already an instance of the enum
        # AND of str; coerce only when it is not already the right enum type.
        for key, enum_cls in (
            ("task_primitive", TaskPrimitive),
            ("abstention_policy", AbstentionPolicy),
            ("data_rights", DataRights),
            ("profile_source", ProfileSource),
        ):
            if key in data and not isinstance(data[key], enum_cls):
                data[key] = enum_cls(str(data[key]).lower())
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


# --------------------------------------------------------------------------- #
@dataclass
class BuildPlanIR:
    """LEVEL 2 — how the system will make it.

    Emitted by the PLANNER and admitted only once every Gate 2 predicate holds.
    Only then is a GPU allocated (B-05).
    """

    spec_hash: str
    base_ref: str
    teacher_ref: str | None = None
    distil_mode: str = "none"
    peft_method: str = "lora"
    rank: int = 16
    data_recipe: dict[str, Any] = field(default_factory=dict)
    quantiser: str = "k_quant"
    bit_width: str = "int4"
    grammar_ref: str | None = None
    eval_suite_ref: str | None = None
    target: str = "gguf"
    budget_usd: float = 40.0
    plan_version: int = 1
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def hash(self) -> str:
        return content_hash(self)

    def to_dict(self) -> dict[str, Any]:
        return _canonical(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BuildPlanIR:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in dict(data).items() if k in known})


# --------------------------------------------------------------------------- #
@dataclass
class ArtefactIR:
    """LEVEL 3 — what actually exists on disk.

    Hashes rather than blobs, so the registry can deduplicate: the base is stored
    once and every cartridge is a delta pointing at it (B-08).
    """

    plan_hash: str
    spec_hash: str
    adapter_blob_hash: str | None = None
    quantised_blob_hash: str | None = None
    grammar_blob: str | None = None
    index_ref: str | None = None
    tool_manifest: list[dict[str, Any]] = field(default_factory=list)
    model_card: dict[str, Any] = field(default_factory=dict)
    eval_certificate: dict[str, Any] = field(default_factory=dict)
    licence_chain: dict[str, Any] = field(default_factory=dict)
    artefact_version: int = 1

    @property
    def hash(self) -> str:
        return content_hash(self)

    def to_dict(self) -> dict[str, Any]:
        return _canonical(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtefactIR:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in dict(data).items() if k in known})


# --------------------------------------------------------------------------- #
def load_spec_ir(path: str | Path) -> SpecIR:
    """Load a :class:`SpecIR` from a ``.json``, ``.yaml`` or ``.yml`` file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"spec file not found: {path}")
    text = p.read_text(encoding="utf-8-sig")  # tolerate a UTF-8 BOM
    if p.suffix in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    return SpecIR.from_dict(data)


def save_spec_ir(spec: SpecIR, path: str | Path) -> Path:
    """Write a Spec IR as canonical JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return p


__all__ = [
    "AbstentionPolicy",
    "ArtefactIR",
    "BuildPlanIR",
    "DataRights",
    "Licence",
    "ProfileSource",
    "SpecIR",
    "content_hash",
    "load_spec_ir",
    "save_spec_ir",
]
