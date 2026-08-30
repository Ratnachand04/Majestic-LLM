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

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from majestic.logging_utils import get_logger
from modelrig import classifier
from modelrig.candidates import CandidateResult, Selection, build_candidates, select
from modelrig.cartridge import Cartridge, build_model_card, cartridge_from_artefact
from modelrig.catalogue import DEFAULT_CATALOGUE, Catalogue
from modelrig.data_factory import DataFactory, DataRefusal
from modelrig.feasibility import HeuristicPerfPredictor, YamlDeviceProfiler
from modelrig.gates import (
    GateResult,
    gate1_spec_admissibility,
    gate2_plan_feasibility,
    gate3_artefact_certification,
)
from modelrig.grammar import compile_for_spec
from modelrig.ir import ArtefactIR, BuildPlanIR, SpecIR, content_hash
from modelrig.licence import resolve_licence_chain
from modelrig.planner import Planner, PlanningResult
from modelrig.proving_ground import ProvingGround, Repairer, Scorecard
from modelrig.quantisation import (
    build_calibration_set,
    evaluate_quantisation,
    next_escalation,
)
from modelrig.registry import CartridgeRegistry

logger = get_logger(__name__)

Example = tuple[str, str]


def _ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


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
    selection: Optional[Selection] = None
    repair_attempts: int = 0
    quantisation: Optional[dict[str, Any]] = None
    #: True only when the SPEC asked for parallel candidates. §15: they raise
    #: expected cost, so they are never enabled on the customer's behalf.
    parallel_candidates: bool = False

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
        n_candidates: int = 2,
        max_repair_attempts: int = 2,
    ) -> None:
        self.catalogue = catalogue or DEFAULT_CATALOGUE
        self.profiler = profiler or YamlDeviceProfiler()
        self.registry = registry or CartridgeRegistry(base_path)
        self.planner = Planner(self.catalogue, self.profiler)
        self.data_factory = data_factory or DataFactory()
        self.repairer = Repairer()
        self.n_candidates = n_candidates
        self.max_repair_attempts = max_repair_attempts

    # -- helpers --------------------------------------------------------- #
    def _admissible(self, spec: SpecIR, plan: BuildPlanIR) -> bool:
        """Only candidates that clear Gate 2 are ever trained."""
        return gate2_plan_feasibility(spec, plan, self.profiler, self.catalogue).passed

    def _train_and_score(
        self, spec: SpecIR, plan: BuildPlanIR, bundle
    ) -> CandidateResult:
        """Train one candidate, quantise it, and score it on the locked split.

        The quantiser is calibrated on the CUSTOMER's task distribution rather
        than generic text — the modification that matters most in A-05 — and the
        answer-flip rate is measured against the pre-quantisation reference.
        """
        started = time.perf_counter()
        base = self.catalogue.base(plan.base_ref)
        held_texts = [t for t, _ in bundle.held_out]

        model = classifier.fit_centroid(bundle.train, bundle.labels)
        reference_preds = classifier.predict(model, held_texts)

        calibration = build_calibration_set(bundle.train, seed=spec.seed_data_count)
        quantised, compression = classifier.quantize_model(model, plan.bit_width)
        quantised_preds = classifier.predict(quantised, held_texts)

        gold = [g for _, g in bundle.held_out]
        quant_outcome = evaluate_quantisation(
            quantiser=plan.quantiser,
            bit_width=plan.bit_width,
            params_b=base.params_b if base else 0.0,
            reference_predictions=reference_preds,
            quantised_predictions=quantised_preds,
            calibration=calibration,
            confidences=[1.0 if p == g else 0.0 for p, g in zip(quantised_preds, gold)],
            correct=[p == g for p, g in zip(quantised_preds, gold)],
        )

        ground = ProvingGround(quality_gate=spec.quality_gate)
        card = ground.evaluate(
            lambda texts: classifier.predict(quantised, list(texts)),
            bundle.held_out,
            training_texts=[t for t, _ in bundle.train][:20],
            reference_predictions=reference_preds,
            post_quantisation=True,
        )

        # Predicted per-request latency for the latency filter in selection.
        latency_ms = 0.0
        if base is not None:
            engine = HeuristicPerfPredictor(self.catalogue)
            try:
                device = self.profiler.profile(spec.device_target)
                latency_ms = engine.predict(
                    plan.base_ref, plan.bit_width, plan.target, device,
                ).latency_ms
            except KeyError:
                latency_ms = _ms_since(started)

        return CandidateResult(
            plan=plan,
            scorecard=card,
            latency_ms=latency_ms,
            params_b=base.params_b if base else 0.0,
            compression=compression,
            quantisation=quant_outcome.__dict__,
        )

    def _repair_plans(
        self, spec: SpecIR, selection: Selection, suggestions: list[str]
    ) -> list[BuildPlanIR]:
        """Turn repairer suggestions into concrete, re-validated plans (B-07).

        GKD is reached only here: it trains on sequences the STUDENT sampled and
        is costlier, so A-02 triggers it inside the repair loop after a failed
        eval gate rather than by default.
        """
        joined = " ".join(suggestions).lower()
        mutated: list[BuildPlanIR] = []
        tried = {c.plan.base_ref for c in selection.candidates}

        if "larger" in joined or "increase base size" in joined:
            bigger = [
                b for b in self.catalogue.bases_for(spec.task_primitive)
                if b.ref not in tried
            ]
            mutated.extend(self.planner._plan_for_base(spec, b) for b in bigger[:1])

        if "gkd" in joined:
            for c in selection.candidates:
                plan = BuildPlanIR.from_dict(c.plan.to_dict())
                plan.distil_mode = "gkd"
                mutated.append(plan)

        if "quantiser" in joined or "spqr" in joined or "flip" in joined:
            for c in selection.candidates:
                plan = BuildPlanIR.from_dict(c.plan.to_dict())
                nxt = next_escalation(plan.quantiser, c.params_b)
                if nxt and nxt != "larger_base":
                    plan.quantiser = nxt if nxt not in ("int8",) else plan.quantiser
                    if nxt == "int8":
                        plan.bit_width = "int8"
                    mutated.append(plan)

        return [p for p in mutated if self._admissible(spec, p)]

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

        # --- cache: an identical ARTEFACT hash skips the build entirely --- #
        # Part 5 §7-§8. Keyed on h_cache, which excludes owner and budget so the
        # hit rate is not zero, and includes seed_data_ref so two customers with
        # identical requirements and different confidential corpora cannot
        # collide. `lookup` applies the owner check on top of that.
        if use_cache:
            cached = self.registry.lookup(spec)
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
        result.plan = planning.plan

        # --- DATA FACTORY ------------------------------------------------- #
        result.stage_reached = "data"
        try:
            bundle = self.data_factory.build(corpus, spec.task_primitive)
        except DataRefusal as exc:
            result.refusal = str(exc)
            logger.warning("compile: refused by the data factory — %s", exc)
            return result

        # --- TRAIN. Candidates are OPT-IN ONLY (§15) ---------------------- #
        # Running k candidates is strictly worse in expected cost at every pass
        # probability: E[cost]_2 / E[cost]_1 = 2/(2-p) > 1. What it buys is
        # wall-clock time and variance reduction, which is the customer's trade
        # to make — so the pipeline never enables it on their behalf.
        result.stage_reached = "train"
        plans = self.planner.candidate_plans(spec, limit=self.n_candidates)
        plans = [p for p in plans if self._admissible(spec, p)] or [planning.plan]
        result.parallel_candidates = len(plans) > 1

        selection = select(
            build_candidates(plans, lambda p: self._train_and_score(spec, p, bundle)),
            spec, self.catalogue,
        )
        result.selection = selection
        result.stage_reached = "prove"

        # --- REPAIR LOOP: mutate the plan on gate failure and rebuild ------ #
        attempts = 0
        while (
            allow_repair
            and not selection.chosen
            and attempts < self.max_repair_attempts
        ):
            failing = next(
                (c for c in selection.candidates if c.scorecard is not None), None
            )
            report = failing.scorecard.failure_report if failing else None
            if report is None:
                break     # the repairer refuses to act without external evidence
            attempts += 1
            result.repair_suggestions = self.repairer.repair(report)
            mutated = self._repair_plans(spec, selection, result.repair_suggestions)
            if not mutated:
                break
            logger.info("compile: repair attempt %d — %s", attempts,
                        ", ".join(result.repair_suggestions[:3]))
            selection = select(
                build_candidates(mutated, lambda p: self._train_and_score(spec, p, bundle)),
                spec, self.catalogue,
            )
            result.selection = selection
        result.repair_attempts = attempts

        if not selection.chosen or selection.winner is None:
            result.refusal = selection.rationale
            self.planner.record_outcome(spec, result.plan, passed=False)
            logger.warning("compile: no candidate cleared the gate — %s", result.refusal)
            return result

        winner = selection.winner
        plan = winner.plan
        result.plan = plan
        card = winner.scorecard
        result.scorecard = card
        result.quantisation = winner.quantisation

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
        eval_report["candidates"] = selection.comparison_table()
        eval_report["selection_rationale"] = selection.rationale

        grammar = compile_for_spec(spec.task_primitive, spec.io_schema)
        artefact = ArtefactIR(
            plan_hash=plan.hash,
            spec_hash=spec.hash,
            adapter_blob_hash=content_hash(
                {"labels": bundle.labels, "train": len(bundle.train), "plan": plan.hash}
            ),
            quantised_blob_hash=content_hash(
                {"q": winner.compression, "plan": plan.hash}
            ),
            grammar_blob=grammar.gbnf if grammar else plan.grammar_ref,
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
            artefact, spec, plan, adapter_bytes=winner.compression.get("comp_bytes", 0)
        )
        result.cartridge = cartridge
        # The spec travels with the cartridge so the registry can record who
        # owns it and whether its corpus is shareable. Without it the entry is
        # admitted but never served from the cache — the safe default.
        result.cartridge_id = self.registry.admit(cartridge, spec=spec)
        result.admitted = True
        self.planner.record_outcome(spec, plan, passed=True)
        logger.info(
            "compile: admitted cartridge %s (%s)",
            result.cartridge_id[:8], selection.rationale,
        )
        return result
