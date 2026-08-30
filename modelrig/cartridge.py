"""The cartridge — a certified, licensed, composable unit (B-08).

A cartridge is NOT a model file. It is five slots plus attached certification:

    SLOT 1  base reference   pointer only, never a copy      ~0 bytes stored
    SLOT 2  adapter stack    the trained delta — customer IP  ~30 MB
    SLOT 3  tool bindings    scoped permissions               ~KB
    SLOT 4  I/O contract     prompt template + compiled grammar  ~KB
    SLOT 5  runtime policy   offline/hybrid, thresholds, fallback ~KB

Storing the base once and every customer as a delta is what turns 550 GB of
full fine-tunes into ~16 GB of adapters at 500 customers — a 34x reduction
(A-03, B-08). The certification (model card, eval certificate, licence chain,
provenance) is auto-generated from pipeline telemetry, never hand-authored.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modelrig.certification import ArtefactKind, CertificationLedger
from modelrig.ir import ArtefactIR, BuildPlanIR, SpecIR, content_hash


@dataclass
class ToolBinding:
    """Slot 3 — one scoped tool permission.

    ``requires_network`` is what makes offline closure statically checkable, and
    ``trusts_untrusted_input`` marks a tool that consumes scraped or retrieved
    content — the indirect prompt-injection surface (2302.12173).
    """

    name: str
    scopes: tuple[str, ...] = ()
    requires_network: bool = False
    trusts_untrusted_input: bool = False
    privileged: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scopes": list(self.scopes),
            "requires_network": self.requires_network,
            "trusts_untrusted_input": self.trusts_untrusted_input,
            "privileged": self.privileged,
        }


@dataclass
class IOContract:
    """Slot 4 — prompt template plus a grammar that makes schema violation impossible."""

    prompt_template: str = "{input}"
    grammar: str | None = None
    output_schema: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_template": self.prompt_template,
            "grammar": self.grammar,
            "output_schema": dict(self.output_schema),
        }


@dataclass
class RuntimePolicy:
    """Slot 5 — how the cartridge behaves in the field."""

    offline: bool = True
    confidence_threshold: float = 0.7
    abstention: str = "flag"
    fallback_target: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "offline": self.offline,
            "confidence_threshold": self.confidence_threshold,
            "abstention": self.abstention,
            "fallback_target": self.fallback_target,
        }


@dataclass
class Cartridge:
    """The shippable, versioned artefact."""

    # Slot 1 — pointer to a base stored once for the whole fleet.
    base_ref: str
    # Slot 2 — the trained delta (path or blob hash). The customer's IP.
    adapter_ref: str | None = None
    adapter_bytes: int = 0
    # Slots 3-5
    tool_bindings: list[ToolBinding] = field(default_factory=list)
    io_contract: IOContract = field(default_factory=IOContract)
    runtime_policy: RuntimePolicy = field(default_factory=RuntimePolicy)
    # Certification, auto-generated from pipeline telemetry.
    model_card: dict[str, Any] = field(default_factory=dict)
    eval_certificate: dict[str, Any] = field(default_factory=dict)
    licence_chain: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    #: §11 — where this cartridge has actually been run. Per device AND per
    #: artefact kind, because a merged model and a base-plus-adapter pair have
    #: different numerics. A device absent from the ledger is reported as
    #: unverified; it is never assumed to behave like one that is present.
    measured_performance: CertificationLedger = field(default_factory=CertificationLedger)
    spec_hash: str = ""
    plan_hash: str = ""
    version: int = 1

    @property
    def id(self) -> str:
        """Content-addressed identity: same inputs, same cartridge."""
        return content_hash(
            {
                "base_ref": self.base_ref,
                "adapter_ref": self.adapter_ref,
                "io_contract": self.io_contract.to_dict(),
                "runtime_policy": self.runtime_policy.to_dict(),
                "spec_hash": self.spec_hash,
                "plan_hash": self.plan_hash,
                "version": self.version,
            }
        )

    @property
    def requires_network(self) -> bool:
        """True when any bound tool needs the network (breaks offline closure)."""
        return any(t.requires_network for t in self.tool_bindings)

    @property
    def certified(self) -> bool:
        """A cartridge is admissible only with certification attached.

        This is certification of the *build*: it was evaluated, it is licensed,
        and its provenance is recorded. It says nothing about any particular
        piece of silicon — see :meth:`certified_for`.
        """
        return bool(self.model_card and self.eval_certificate
                    and self.licence_chain.get("permitted", False))

    def certified_for(self, device_id: str, kind: ArtefactKind | None = None) -> bool:
        """Whether this cartridge has been proven to run on **this** device.

        Deliberately separate from :attr:`certified`. A build can be
        impeccably evaluated on a workstation and still answer differently
        through another runtime on another SoC, so the two questions have two
        answers and the manifest keeps them apart.
        """
        return self.certified and self.measured_performance.certified_for(device_id, kind)

    def device_status(self, device_id: str, kind: ArtefactKind | None = None) -> str:
        """A Build Card sentence about this device. Never optimistic."""
        if not self.certified:
            return "UNCERTIFIED: the build itself carries no evaluation or licence chain"
        return self.measured_performance.status_for(device_id, kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "base_ref": self.base_ref,
            "adapter_ref": self.adapter_ref,
            "adapter_bytes": self.adapter_bytes,
            "tool_bindings": [t.to_dict() for t in self.tool_bindings],
            "io_contract": self.io_contract.to_dict(),
            "runtime_policy": self.runtime_policy.to_dict(),
            "model_card": dict(self.model_card),
            "eval_certificate": dict(self.eval_certificate),
            "licence_chain": dict(self.licence_chain),
            "provenance": dict(self.provenance),
            "measured_performance": self.measured_performance.to_list(),
            "spec_hash": self.spec_hash,
            "plan_hash": self.plan_hash,
            "version": self.version,
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> Cartridge:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            base_ref=data["base_ref"],
            adapter_ref=data.get("adapter_ref"),
            adapter_bytes=int(data.get("adapter_bytes", 0)),
            tool_bindings=[ToolBinding(**{k: (tuple(v) if k == "scopes" else v)
                                          for k, v in t.items()})
                           for t in data.get("tool_bindings", [])],
            io_contract=IOContract(**data.get("io_contract", {})),
            runtime_policy=RuntimePolicy(**data.get("runtime_policy", {})),
            model_card=data.get("model_card", {}),
            eval_certificate=data.get("eval_certificate", {}),
            licence_chain=data.get("licence_chain", {}),
            provenance=data.get("provenance", {}),
            measured_performance=CertificationLedger.from_list(
                data.get("measured_performance")
            ),
            spec_hash=data.get("spec_hash", ""),
            plan_hash=data.get("plan_hash", ""),
            version=int(data.get("version", 1)),
        )


def build_model_card(spec: SpecIR, plan: BuildPlanIR, eval_report: dict) -> dict[str, Any]:
    """Auto-generate a model card from pipeline telemetry (never hand-authored)."""
    return {
        "task_primitive": spec.task_primitive.value,
        "intended_use": spec.notes or f"{spec.task_primitive.value} on {spec.device_target}",
        "base_model": plan.base_ref,
        "teacher": plan.teacher_ref,
        "training_method": plan.peft_method,
        "distillation": plan.distil_mode,
        "quantisation": f"{plan.quantiser}/{plan.bit_width}",
        "target_runtime": plan.target,
        "languages": list(spec.languages),
        "device_target": spec.device_target,
        "offline": spec.offline_required,
        "policy_rules": list(spec.policy_rules),
        "abstention_policy": spec.abstention_policy.value,
        "evaluated_on": f"{eval_report.get('n_test', 0)} real held-out examples",
        "known_limitations": [
            "performance figures on mobile are predicted, not measured (GAP-10)",
            "confidence calibration is unvalidated below 2B parameters (GAP-03)",
        ],
    }


def cartridge_from_artefact(
    artefact: ArtefactIR, spec: SpecIR, plan: BuildPlanIR, adapter_bytes: int = 0
) -> Cartridge:
    """Assemble a cartridge from a certified artefact."""
    return Cartridge(
        base_ref=plan.base_ref,
        adapter_ref=artefact.adapter_blob_hash or artefact.quantised_blob_hash,
        adapter_bytes=adapter_bytes,
        tool_bindings=[ToolBinding(**t) for t in artefact.tool_manifest],
        io_contract=IOContract(grammar=artefact.grammar_blob,
                               output_schema=dict(spec.io_schema)),
        runtime_policy=RuntimePolicy(
            offline=spec.offline_required,
            abstention=spec.abstention_policy.value,
        ),
        model_card=artefact.model_card,
        eval_certificate=artefact.eval_certificate,
        licence_chain=artefact.licence_chain,
        spec_hash=spec.hash,
        plan_hash=plan.hash,
    )
