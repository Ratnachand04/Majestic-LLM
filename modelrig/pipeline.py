"""The end-to-end compiler: seven subsystems, three gates (B-01, C-02).

    FORGE -> Spec IR -> [Gate 1] -> PLANNER -> Build Plan IR -> [Gate 2]
          -> DATA FACTORY -> TRAINER -> PROVING GROUND -> [Gate 3]
          -> CARTRIDGE -> REGISTRY

Every gate is checked BEFORE the next stage spends anything. A refusal in eight
seconds is a better product than a failed build in ninety minutes.

What is sold is not a model file: it is a SCORECARD proving a small specialist
does one job as well as a large generalist, plus the installable artefact for the
device the customer owns.

The offline path trains the dependency-free TF-IDF classifier so the whole flow
runs on CPU with no downloads; the heavy LoRA path is reachable through the same
Build Plan IR when the ML extras are installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from majestic.logging_utils import get_logger
from modelrig import classifier
from modelrig.cartridge import Cartridge, build_model_card, cartridge_from_artefact
from modelrig.catalogue import DEFAULT_CATALOGUE, Catalogue
from modelrig.data_factory import DataFactory, DataRefusal
from modelrig.feasibility import YamlDeviceProfiler
from modelrig.gates import GateResult, gate1_spec_admissibility, gate3_artefact_certification
from modelrig.ir import ArtefactIR, BuildPlanIR, SpecIR, content_hash
from modelrig.licence import resolve_licence_chain
from modelrig.planner import Planner, PlanningResult
from modelrig.proving_ground import ProvingGround, Repairer, Scorecard
from modelrig.registry import CartridgeRegistry

logger = get_logger(__name__)

Example = tuple[str, str]


@dataclass
class CompileResult:
    """Everything one compilation produced — including an honest refusal."""

    spec: SpecIR
    admitted: bool = False
    plan: Optional[BuildPlanIR] = None
    artefact: Optional[ArtefactIR] = None
    cartridge: Optional[Cartridge] = None
    cartridge_id: Optional[str] = None
    scorecard: Optional[Scorecard] = None
    gates: list[GateResult] = field(default_factory=list)
    cache_hit: bool = False
    refusal: str = ""
    repair_suggestions: list[str] = field(default_factory=list)
    stage_reached: str = "gate1"

    @property
    def refused(self) -> bool:
        return not self.admitted

    def summary(self) -> dict[str, Any]:
        return {
            "spec_hash": self.spec.hash,
            "primitive": self.spec.task_primitive.value,
            "admitted": self.admitted,
            "stage_reached": self.stage_reached,
            "cache_hit": self.cache_hit,
            "cartridge_id": self.cartridge_id,
            "score": (
                next((a.score for a in self.scorecard.axes if a.name == "task_metric"), None)
                if self.scorecard else None
            ),
            "answer_flip_rate": self.scorecard.answer_flip_rate if self.scorecard else None,
            "refusal": self.refusal,
        }


class MajesticCompiler:
    """Compiles a Spec IR into a certified cartridge, or refuses with reasons."""

    def __init__(
        self,
        registry: Optional[CartridgeRegistry] = None,
        catalogue: Optional[Catalogue] = None,
        profiler: Optional[YamlDeviceProfiler] = None,
        data_factory: Optional[DataFactory] = None,
        base_path: str | Path = "./registry",
    ) -> None:
        self.catalogue = catalogue or DEFAULT_CATALOGUE
        self.profiler = profiler or YamlDeviceProfiler()
        self.registry = registry or CartridgeRegistry(base_path)
        self.planner = Planner(self.catalogue, self.profiler)
        self.data_factory = data_factory or DataFactory()
        self.repairer = Repairer()

    # ------------------------------------------------------------------ #
    def compile(
        self,
        spec: SpecIR,
        corpus: list[Example],
        *,
        use_cache: bool = True,
        allow_repair: bool = True,
    ) -> CompileResult:
        """Run the full flow for one specification."""
        result = CompileResult(spec=spec)

        # --- Gate 1: may we attempt this at all? ------------------------ #
        gate1 = gate1_spec_admissibility(spec, self.profiler, self.catalogue)
        result.gates.append(gate1)
        if not gate1.passed:
            result.refusal = "; ".join(gate1.reasons)
            logger.warning("compile: refused at Gate 1 — %s", result.refusal)
            return result

        # --- cache: an identical spec hash skips the build entirely ------ #
        if use_cache:
            cached = self.registry.by_spec_hash(spec.hash)
            if cached is not None:
                result.admitted = True
                result.cache_hit = True
                result.cartridge = cached
                result.cartridge_id = cached.id
                result.stage_reached = "cache"
                return result

        # --- PLANNER + Gate 2 -------------------------------------------- #
        result.stage_reached = "gate2"
        planning: PlanningResult = self.planner.plan(spec)
        result.gates.append(planning.gate)
        if not planning.admitted or planning.plan is None:
            result.refusal = "; ".join(planning.gate.reasons)
            logger.warning("compile: refused at Gate 2 — %s", result.refusal)
            return result
        plan = planning.plan
        result.plan = plan

        # --- DATA FACTORY ------------------------------------------------- #
        result.stage_reached = "data"
        try:
            bundle = self.data_factory.build(corpus, spec.task_primitive)
        except DataRefusal as exc:
            result.refusal = str(exc)
            logger.warning("compile: refused by the data factory — %s", exc)
            return result

        # --- TRAINER (offline path) --------------------------------------- #
        result.stage_reached = "train"
        model = classifier.fit_centroid(bundle.train, bundle.labels)
        reference_preds = classifier.predict(
            model, [t for t, _ in bundle.held_out]
        )

        # --- COMPRESSION --------------------------------------------------- #
        quantised, compression = classifier.quantize_model(model, plan.bit_width)

        # --- PROVING GROUND (post-quantisation, with answer-flip) ---------- #
        result.stage_reached = "prove"
        ground = ProvingGround(quality_gate=spec.quality_gate)
        card = ground.evaluate(
            lambda texts: classifier.predict(quantised, list(texts)),
            bundle.held_out,
            training_texts=[t for t, _ in bundle.train][:20],
            reference_predictions=reference_preds,
            post_quantisation=True,
        )
        result.scorecard = card

        # --- Gate 3: artefact certification -------------------------------- #
        result.stage_reached = "gate3"
        base = self.catalogue.base(plan.base_ref)
        teacher = self.catalogue.teacher(plan.teacher_ref) if plan.teacher_ref else None
        chain = resolve_licence_chain(
            base_licence=base.licence if base else "unknown",
            teacher_licence=teacher.licence if teacher else None,
            data_rights=spec.data_rights,
            jurisdiction=spec.jurisdiction,
        )
        eval_report = card.to_eval_report()
        artefact = ArtefactIR(
            plan_hash=plan.hash,
            spec_hash=spec.hash,
            adapter_blob_hash=content_hash(
                {"labels": bundle.labels, "train": len(bundle.train), "plan": plan.hash}
            ),
            quantised_blob_hash=content_hash({"q": compression, "plan": plan.hash}),
            grammar_blob=plan.grammar_ref,
            model_card=build_model_card(spec, plan, eval_report),
            eval_certificate=eval_report,
            licence_chain=chain.as_record(),
        )
        result.artefact = artefact

        gate3 = gate3_artefact_certification(artefact, spec, eval_report)
        result.gates.append(gate3)
        if not gate3.passed:
            result.refusal = "; ".join(gate3.reasons)
            if allow_repair and card.failure_report is not None:
                result.repair_suggestions = self.repairer.repair(card.failure_report)
            self.planner.record_outcome(spec, plan, passed=False)
            logger.warning("compile: refused at Gate 3 — %s", result.refusal)
            return result

        # --- CARTRIDGE + REGISTRY ------------------------------------------ #
        result.stage_reached = "registry"
        cartridge = cartridge_from_artefact(
            artefact, spec, plan, adapter_bytes=compression.get("comp_bytes", 0)
        )
        result.cartridge = cartridge
        result.cartridge_id = self.registry.admit(cartridge)
        result.admitted = True
        self.planner.record_outcome(spec, plan, passed=True)
        logger.info("compile: admitted cartridge %s", result.cartridge_id[:8])
        return result
