"""MAJESTIC Planner — the middle end of the compiler.

It receives a validated specification and emits either an executable build plan
or a **refusal**, and it does so before any GPU is allocated. In one sentence: it
decides whether the thing the customer asked for is physically, legally and
statistically possible, and if so, how to build it most cheaply.

Three properties define it:

* **Deterministic.** The common path involves no language model. Same spec, same
  catalogue, same history gives the same plan — which is what makes the system
  auditable, cacheable and reproducible.
* **Refusal is a first-class output.** :class:`~modelrig.planner.refusal.Refusal`
  sits alongside ``BuildPlanIR`` in the return type. Every other platform always
  returns *something*.
* **Constraints span domains that do not normally meet.** RAM arithmetic is
  physics, licence composition is law, seed sufficiency is statistics — and a
  plan must satisfy all three at once.

Entry points
------------
``plan(spec)``         the algorithm of §7, returning :class:`PlanOutcome`
``select_base(...)``   §12's maximum over a chain
``Planner``            the stateful adapter the build pipeline uses
"""
from modelrig.planner.audit import (
    C2_PRECISION_TARGET,
    MIN_SOFT_SHARE,
    PrecisionEstimate,
    RefusalAudit,
    RefusalOutcome,
    enumeration_economy,
    plan_space_size,
    summarise_witnesses,
)
from modelrig.planner.catalog import (
    Catalog,
    CatalogError,
    DeviceSpec,
    ModelSpec,
    QuantiserSpec,
    default_catalog,
    load_catalog,
)
from modelrig.planner.compat import Planner, PlanningResult, PrecedentRecord
from modelrig.planner.core import (
    Enumeration,
    PlanCandidate,
    PlanOutcome,
    enumerate_feasible,
    plan,
    select_base,
    spec_shape,
    verify_antitone,
)
from modelrig.planner.costmodel import CostBreakdown, usd
from modelrig.planner.metalearn import (
    Observation,
    OutcomePredictor,
    PrecedentIndex,
    features_of,
)
from modelrig.planner.objective import (
    DecisionParams,
    Tier,
    refusal_threshold,
    should_build,
    utility,
)
from modelrig.planner.predicates import (
    ALL_PREDICATES,
    HARD,
    SOFT,
    P_cost,
    P_lat,
    P_lic,
    P_off,
    P_ram,
    P_seed,
    P_tok,
    PredicateResult,
    Soundness,
    derive_distil_mode,
    expected_cost,
    ordered_predicates,
    seed_floor,
)
from modelrig.planner.refusal import Refusal, Witness, WitnessSet, minimal_cover

__all__ = [
    # §16 — refusal quality, stratified so it cannot be inflated
    "C2_PRECISION_TARGET", "MIN_SOFT_SHARE", "PrecisionEstimate", "RefusalAudit",
    "RefusalOutcome", "enumeration_economy", "plan_space_size", "summarise_witnesses",
    # catalogue
    "Catalog", "CatalogError", "DeviceSpec", "ModelSpec", "QuantiserSpec",
    "default_catalog", "load_catalog",
    # core
    "Enumeration", "PlanCandidate", "PlanOutcome", "enumerate_feasible",
    "plan", "select_base", "spec_shape", "verify_antitone",
    # predicates
    "ALL_PREDICATES", "HARD", "SOFT", "PredicateResult", "Soundness",
    "P_tok", "P_seed", "P_lic", "P_ram", "P_off", "P_lat", "P_cost",
    "derive_distil_mode", "expected_cost", "ordered_predicates", "seed_floor",
    # refusal
    "Refusal", "Witness", "WitnessSet", "minimal_cover",
    # objective
    "DecisionParams", "Tier", "refusal_threshold", "should_build", "utility",
    # cost + meta-learning
    "CostBreakdown", "usd", "Observation", "OutcomePredictor", "PrecedentIndex",
    "features_of",
    # pipeline adapter
    "Planner", "PlanningResult", "PrecedentRecord",
]
