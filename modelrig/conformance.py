"""Conformance and compatibility self-check.

Two independent questions, answered separately because they fail for different
reasons:

**Model compatibility** — do the catalogue entries match the published model
configurations? A wrong ``n_kv_heads`` silently corrupts every KV-cache estimate,
which corrupts every device feasibility verdict, which is the one number the
product sells. Reference values below come from each model's published config.

**Architecture conformance** — does the implementation actually obey the rules
the architecture states? Rules that live only in documentation drift. Each check
names the diagram it enforces so a violation points at its source.

Run with ``python -m cli.main validate``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from majestic.logging_utils import get_logger
from modelrig.catalogue import BYTES_PER_PARAM, DEFAULT_CATALOGUE, Catalogue
from modelrig.licence import Licence
from modelrig.primitives import TaskPrimitive, all_primitives

logger = get_logger(__name__)

#: Published configuration values, keyed by model reference.
#: (n_layers, n_kv_heads, head_dim, params_b, tokenizer_family)
PUBLISHED_CONFIGS: dict[str, tuple[int, int, int, float, str]] = {
    "Qwen/Qwen3-0.6B": (28, 8, 128, 0.6, "qwen"),
    "Qwen/Qwen3-1.7B": (28, 8, 128, 1.7, "qwen"),
    "Qwen/Qwen3-4B": (36, 8, 128, 4.0, "qwen"),
    "Qwen/Qwen3-8B": (36, 8, 128, 8.2, "qwen"),
    "Qwen/Qwen3-14B": (40, 8, 128, 14.8, "qwen"),
    "Qwen/Qwen3-32B": (64, 8, 128, 32.8, "qwen"),
    "Qwen/Qwen3-30B-A3B": (48, 4, 128, 30.5, "qwen"),
    "HuggingFaceTB/SmolLM2-360M-Instruct": (32, 5, 64, 0.36, "smollm"),
    "meta-llama/Llama-3.2-1B-Instruct": (16, 8, 64, 1.24, "llama"),
}

#: Tolerance on parameter counts — published figures round differently.
_PARAM_TOLERANCE = 0.15


@dataclass
class Finding:
    """One conformance or compatibility problem."""

    check: str
    subject: str
    detail: str
    severity: str = "error"       # error | warning
    source: str = ""              # the diagram the rule comes from


@dataclass
class ConformanceReport:
    findings: list[Finding] = field(default_factory=list)
    checks_run: int = 0

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, **kwargs: Any) -> None:
        self.findings.append(Finding(**kwargs))

    def summary(self) -> dict[str, Any]:
        return {
            "checks_run": self.checks_run,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "ok": self.ok,
        }


# --------------------------------------------------------------------------- #
def check_model_compatibility(catalogue: Catalogue | None = None) -> ConformanceReport:
    """Validate catalogue entries against published model configurations."""
    catalogue = catalogue or DEFAULT_CATALOGUE
    report = ConformanceReport()

    for model in list(catalogue.bases) + list(catalogue.teachers):
        report.checks_run += 1
        published = PUBLISHED_CONFIGS.get(model.ref)
        if published is None:
            report.add(
                check="published_config", subject=model.ref, severity="warning",
                detail="no published reference on file; KV-cache estimates unverified",
                source="A-01",
            )
            continue

        layers, kv_heads, head_dim, params_b, tokenizer = published
        if model.n_layers != layers:
            report.add(check="n_layers", subject=model.ref, source="A-01",
                       detail=f"catalogue {model.n_layers} != published {layers}")
        if model.n_kv_heads != kv_heads:
            report.add(check="n_kv_heads", subject=model.ref, source="A-01",
                       detail=(f"catalogue {model.n_kv_heads} != published {kv_heads}; "
                               "this silently corrupts every KV-cache estimate"))
        if model.head_dim != head_dim:
            report.add(check="head_dim", subject=model.ref, source="A-01",
                       detail=f"catalogue {model.head_dim} != published {head_dim}")
        if abs(model.params_b - params_b) / max(params_b, 1e-6) > _PARAM_TOLERANCE:
            report.add(check="params_b", subject=model.ref, source="A-01",
                       detail=f"catalogue {model.params_b}B != published {params_b}B")
        if model.tokenizer_family != tokenizer:
            report.add(check="tokenizer_family", subject=model.ref, source="A-02",
                       detail=(f"catalogue {model.tokenizer_family!r} != published "
                               f"{tokenizer!r}; logit-KD eligibility depends on this"))

    # Internal consistency the published data cannot cover.
    for model in catalogue.bases:
        report.checks_run += 1
        if model.is_moe and (model.active_params_b or 0) >= model.params_b:
            report.add(check="moe_active_params", subject=model.ref, source="A-07",
                       detail="active parameters must be fewer than resident parameters")
        if not model.is_moe and model.active_params_b is not None:
            report.add(check="moe_flag", subject=model.ref, source="A-07",
                       severity="warning",
                       detail="active_params_b set on a dense model")
        if model.kv_bytes_per_token() <= 0:
            report.add(check="kv_geometry", subject=model.ref, source="A-01",
                       detail="KV bytes per token must be positive")

    # A-02: logit distillation needs a same-family teacher, or it is unreachable.
    report.checks_run += 1
    families = {b.tokenizer_family for b in catalogue.bases if not b.is_moe}
    teacher_families = {t.tokenizer_family for t in catalogue.teachers}
    for family in families - teacher_families:
        report.add(
            check="logit_kd_reachable", subject=family, severity="warning",
            source="A-02",
            detail=(f"no teacher in tokenizer family {family!r}: logit distillation "
                    "is unavailable for these bases, only sequence-level KD"),
        )

    # Teachers must be permissively licensed (A-02 hard legal boundary).
    for teacher in catalogue.teachers:
        report.checks_run += 1
        if teacher.licence not in (Licence.APACHE_2_0, Licence.MIT, Licence.BSD_3):
            report.add(check="teacher_licence", subject=teacher.ref, source="A-02",
                       detail=(f"teacher licence {teacher.licence.value!r} is not "
                               "permissive; its outputs cannot train a competing model"))
        if not teacher.can_teach:
            report.add(check="teacher_flag", subject=teacher.ref, source="A-02",
                       severity="warning", detail="listed as a teacher but can_teach is False")

    return report


# --------------------------------------------------------------------------- #
def check_architecture_conformance(catalogue: Catalogue | None = None) -> ConformanceReport:
    """Assert the implementation obeys the rules the architecture states."""
    catalogue = catalogue or DEFAULT_CATALOGUE
    report = ConformanceReport()

    # A-01: the catalogue must order largest-first, or the planner ships weaker
    # models than the device could carry.
    report.checks_run += 1
    ordered = catalogue.bases_for(TaskPrimitive.EXTRACT)
    if ordered and ordered != sorted(ordered, key=lambda b: -b.params_b):
        report.add(check="largest_first", subject="catalogue.bases_for", source="A-01",
                   detail="bases must be offered largest-first (k-bit scaling law)")

    # A-01: 4-bit must be the cheapest bit-width per parameter.
    report.checks_run += 1
    if BYTES_PER_PARAM["int4"] >= BYTES_PER_PARAM["int8"]:
        report.add(check="bit_width_ordering", subject="BYTES_PER_PARAM", source="A-01",
                   detail="int4 must cost fewer bytes per parameter than int8")

    # A-01: the int4 constant must reproduce the published size ladder. If these
    # drift apart, every device budget is silently wrong.
    from modelrig.catalogue import SIZE_LADDER

    for params_b, disk_gb, _tier in SIZE_LADDER:
        report.checks_run += 1
        predicted = params_b * BYTES_PER_PARAM["int4"]
        if abs(predicted - disk_gb) / disk_gb > 0.20:
            report.add(
                check="size_ladder_matches_constant", subject=f"{params_b}B",
                source="A-01",
                detail=(f"predicted {predicted:.2f} GB vs published {disk_gb} GB on "
                        "disk; the int4 constant no longer reproduces the ladder"),
            )

    # A-07: MoE must never be offered by default.
    report.checks_run += 1
    if any(b.is_moe for b in catalogue.bases_for(TaskPrimitive.EXTRACT)):
        report.add(check="moe_excluded", subject="catalogue.bases_for", source="A-07",
                   detail="MoE bases must be excluded by default: sparse activation "
                          "is not sparse residency")

    # A-04 / B-08: the catalogue is narrow on purpose.
    report.checks_run += 1
    dense = [b for b in catalogue.bases if not b.is_moe]
    if len(dense) > 8:
        report.add(check="narrow_catalogue", subject="catalogue", source="A-04",
                   severity="warning",
                   detail=(f"{len(dense)} dense bases: every extra base fragments "
                           "multi-adapter serving economics"))

    # GAP-06 / B-03: the primitive set is closed at eight.
    report.checks_run += 1
    if len(all_primitives()) != 8:
        report.add(check="eight_primitives", subject="primitives", source="B-03",
                   detail=f"the supported set must be exactly eight, found "
                          f"{len(all_primitives())}")

    # Every primitive must be servable by at least one base.
    for prim in all_primitives():
        report.checks_run += 1
        if not catalogue.bases_for(prim.primitive):
            report.add(check="primitive_servable", subject=prim.primitive.value,
                       source="B-05",
                       detail="no catalogue base declares support for this primitive")

    # A-01: every base must fall on the published size ladder, or the planner
    # cannot explain its choice to the customer.
    from modelrig.catalogue import ladder_tier

    for base in dense:
        report.checks_run += 1
        if "beyond the ladder" in ladder_tier(base.params_b):
            report.add(check="size_ladder", subject=base.ref, source="A-01",
                       severity="warning",
                       detail="base sits beyond the published size ladder")

    return report


# --------------------------------------------------------------------------- #
def check_elicitation_conformance() -> ConformanceReport:
    """Assert FORGE obeys the rules Part 4 states about asking questions.

    These are the ones that fail *silently* when they drift: an interview that
    quietly asks for something it could have probed still returns a spec, and
    nothing downstream notices until the completion rate falls.
    """
    from modelrig.forge import slots as slot_table
    from modelrig.forge.core import (
        MAX_AMBIGUITY,
        ATTRITION_GAMMA,
        completion_probability,
        value_ratio,
    )
    from modelrig.planner.objective import Tier

    report = ConformanceReport()

    # §10: nothing measurable or computable may cost a question.
    for problem in slot_table.validate_table():
        report.checks_run += 1
        report.add(check="slot_table", subject=problem.split(":")[0], source="P4-10",
                   detail=problem)
    report.checks_run += 1

    # §9: the elicited set must stay a minority of the schema, or the economy
    # that makes four questions enough has stopped holding.
    report.checks_run += 1
    asked = len(slot_table.MUST_ASK)
    if asked > len(slot_table.SLOTS) / 2:
        report.add(check="ask_economy", subject="slots.MUST_ASK", source="P4-09",
                   severity="warning",
                   detail=(f"{asked} of {len(slot_table.SLOTS)} slots are must-ask; "
                           "the probe and the derivations are meant to carry most "
                           "of the schema"))

    # §4 pins gamma by three stated points rather than a band: four questions
    # complete 82% of the time, ten 61%, twenty 37%. Asserting the curve through
    # its own published values is stronger than asserting a shape, and it is what
    # catches gamma being changed to something that merely looks reasonable.
    for questions, expected in ((4, 0.82), (10, 0.61), (20, 0.37)):
        report.checks_run += 1
        actual = completion_probability(questions)
        if abs(actual - expected) > 0.01:
            report.add(check="attrition_calibration", subject=f"q={questions}",
                       source="P4-04",
                       detail=(f"gamma={ATTRITION_GAMMA} gives P({questions})="
                               f"{actual:.3f}; §4 states {expected:.2f}"))

    # §4-§6: raising kappa must make the system BOTH refuse more and ask more.
    # If these ever move in opposite directions, one of the two is miscalibrated
    # and the regulated tier becomes the permissive one.
    report.checks_run += 1
    tiers = [Tier.EXPERIMENTAL, Tier.COMMERCIAL, Tier.REGULATED]
    lambdas = [value_ratio(t) for t in tiers]
    thetas = [objective_threshold(t) for t in tiers]
    if lambdas != sorted(lambdas) or thetas != sorted(thetas):
        report.add(check="caution_moves_together", subject="Tier", source="P4-04",
                   detail=(f"lambda={[round(x, 2) for x in lambdas]} and "
                           f"theta*={[round(x, 2) for x in thetas]} must both rise "
                           "with trust damage: more refusals AND more questions"))


    # §5: the ambiguity ceiling must bind. At A_max >= 1 no reading is ever
    # contested and the third ask-trigger silently disappears — the interview
    # would go back to picking one meaning of a sentence that supports two.
    report.checks_run += 1
    if not 0.0 < MAX_AMBIGUITY < 1.0:
        report.add(check="ambiguity_ceiling", subject="MAX_AMBIGUITY", source="P4-05",
                   detail=(f"A_max={MAX_AMBIGUITY} does not bind; ambiguity stops "
                           "being an independent reason to ask"))

    # §5: termination needs all four conditions. The Interview must not report
    # itself complete on the strength of a spec that merely type-checks.
    report.checks_run += 1
    from modelrig.forge.core import Interview

    complete_src = Interview.complete.fget.__doc__ or ""
    if "four conditions" not in complete_src:
        report.add(check="termination_conditions", subject="Interview.complete",
                   source="P4-05", severity="warning",
                   detail="completeness should document all four §5 conditions")

    return report


# --------------------------------------------------------------------------- #
def check_proving_ground_conformance() -> ConformanceReport:
    """Assert Part 7's statistical rules.

    Majestic sells a certificate rather than a model file, so these are the
    checks that keep the product from being a lie with good typography. Both
    corrections they enforce failed silently before: a point-estimate gate looks
    like it works, and seven blocking axes look conservative.
    """
    from modelrig.proving_ground import BLOCKING_AXES
    from modelrig.stats import certifies, clopper_pearson_lcb, required_n, wilson

    report = ConformanceReport()

    # §5: the gate is a hypothesis test. A point-estimate comparison at n=50
    # certifies a claim the data cannot carry, and nothing else would report it.
    report.checks_run += 1
    if certifies(47, 50, 0.93):
        report.add(check="gate_on_lower_bound", subject="certifies", source="P7-05",
                   detail=("47/50 must NOT certify a 0.93 gate: the interval spans "
                           "fourteen points, so the observed score and the gate are "
                           "statistically indistinguishable"))

    # §3: one error in fifty must break a 0.93 claim. If this ever passes, the
    # bound has been replaced by something optimistic.
    report.checks_run += 1
    if certifies(49, 50, 0.93) or not certifies(50, 50, 0.93):
        report.add(check="one_error_breaks_the_claim", subject="clopper_pearson_lcb",
                   source="P7-03",
                   detail=(f"at n=50 a perfect sweep bounds at "
                           f"{clopper_pearson_lcb(50, 50):.3f} and one error at "
                           f"{clopper_pearson_lcb(49, 50):.3f}; only the first may "
                           "certify 0.93"))

    # §3: the interval must not collapse at the boundary. A normal approximation
    # gives zero width at k=n, which would certify anything on five examples.
    report.checks_run += 1
    if wilson(5, 5).low > 0.7:
        report.add(check="interval_honest_at_the_boundary", subject="wilson",
                   source="P7-03",
                   detail=("a clean sweep of five must not produce a narrow "
                           "interval; that is how an approximation certifies noise"))

    # §9: exactly four axes block. Seven would reject four good models in five.
    report.checks_run += 1
    if len(BLOCKING_AXES) != 4:
        report.add(check="four_blocking_axes", subject="BLOCKING_AXES", source="P7-09",
                   detail=(f"{len(BLOCKING_AXES)} blocking axes: at 80% power each, "
                           f"P(ship | good) = 0.8^{len(BLOCKING_AXES)}, and seven "
                           "blocking gates ships only 21% of good models"))

    # §9: the axes that block are chosen by consequence — the job, two unbounded
    # losses, and one deterministic check that costs no power.
    for name in ("task_metric", "safety", "privacy"):
        report.checks_run += 1
        if name not in BLOCKING_AXES:
            report.add(check="blocking_by_consequence", subject=name, source="P7-09",
                       detail=f"{name} must block: it is the job or its loss is unbounded")

    # §4: the power calculator must agree that fifty examples cannot resolve a
    # three-point gate. If it ever does, the sizing formula has drifted.
    report.checks_run += 1
    if required_n(0.93, 0.96) < 200:
        report.add(check="power_calculator", subject="required_n", source="P7-04",
                   detail=(f"detecting three points above a 0.93 gate needs about 380 "
                           f"examples; the calculator says {required_n(0.93, 0.96)}"))

    return report


# --------------------------------------------------------------------------- #
def check_fabric_conformance() -> ConformanceReport:
    """Assert Part 6's analysis rules.

    The first two are the ones that would fail silently. A cartridge quietly
    becoming a sanitiser would make every graph verify, and a threshold standing
    in for graph rewriting would make every offline claim unprovable — while
    both still returned green.
    """
    import math

    from majestic.fabric.capacity import (
        UNBOUNDED_BITS,
        analyse_capacity,
        bits_for_schema,
    )
    from majestic.fabric.graph import FabricGraph, Node, NodeKind, TaintRole
    from majestic.fabric.schedule import belady, lru

    report = ConformanceReport()

    # §10: a cartridge propagates. Treating a model as a sanitiser because it
    # "wrote its own summary" defeats the entire analysis.
    report.checks_run += 1
    plain = Node("c", NodeKind.CARTRIDGE)
    if plain.role is not TaintRole.PROPAGATE or plain.capacity_bits != UNBOUNDED_BITS:
        report.add(check="cartridge_propagates", subject="Node.role", source="P6-10",
                   detail=("an unconstrained cartridge must propagate taint at "
                           "unbounded capacity; a model given injected instructions "
                           "emits attacker-chosen content"))

    # §12: one free-text field opens the channel. This is the asymmetry the
    # whole result rests on.
    report.checks_run += 1
    enum_only = {"priority": ["urgent", "normal", "low"]}
    if bits_for_schema(enum_only) >= UNBOUNDED_BITS or \
            bits_for_schema({**enum_only, "notes": "string"}) != UNBOUNDED_BITS:
        report.add(check="free_text_opens_the_channel", subject="bits_for_schema",
                   source="P6-12",
                   detail=("a constrained schema must have finite capacity and one "
                           "unconstrained field must make it unbounded"))

    # §11: c_max = 0 must reproduce binary taint exactly, or the quantitative
    # analysis is not a generalisation of the strict rule but a weakening of it.
    report.checks_run += 1
    g = FabricGraph(name="conformance")
    g.add(Node("src", NodeKind.TOOL, produces_untrusted=True))
    g.add(Node("mid", NodeKind.CARTRIDGE, output_domain_bits=math.log2(5)))
    g.add(Node("sink", NodeKind.TOOL, privileged=True))
    g.connect("src", "mid")
    g.connect("mid", "sink")
    if analyse_capacity(g, c_max=0.0).safe or not analyse_capacity(g, c_max=3.0).safe:
        report.add(check="c_max_zero_is_binary_taint", subject="analyse_capacity",
                   source="P6-11",
                   detail=("c_max=0 must refuse a 2.3-bit channel and c_max=3 must "
                           "admit it; otherwise the threshold is not a "
                           "generalisation of the strict rule"))

    # §10: a confirm node clears taint, or there is no way to build a graph that
    # touches untrusted content and still acts.
    report.checks_run += 1
    if Node("h", NodeKind.CONFIRM).role is not TaintRole.CLEAR:
        report.add(check="confirm_clears", subject="NodeKind.CONFIRM", source="P6-10",
                   detail="a human approval node must clear taint")

    # §15: Belady is optimal for the offline problem, so it must never lose to
    # an online policy. A regression here means the scheduler stopped being MIN.
    report.checks_run += 1
    sigma = ["a", "b", "c", "a", "d", "a", "b", "c"]
    for capacity in (1, 2, 3):
        if belady(sigma, capacity).swaps > lru(sigma, capacity).swaps:
            report.add(check="belady_optimal", subject=f"capacity={capacity}",
                       source="P6-15",
                       detail=("Belady's MIN is optimal for offline paging and must "
                               "never take more swaps than LRU"))

    return report


# --------------------------------------------------------------------------- #
def check_registry_conformance() -> ConformanceReport:
    """Assert Part 5's storage and privacy rules.

    The first check is the important one. §8 is a data-protection constraint
    that a plausible optimisation would remove — hashing the requirements and
    not the data looks like a free improvement to the hit rate, and would serve
    one customer a model trained on another customer's documents. Nothing else
    in the system would report it.
    """
    from modelrig.cachekey import (
        CACHE_EXCLUDED,
        CACHE_REQUIRED,
        h_cache,
        lookup_allowed,
    )
    from modelrig.corpus import FORBIDDEN_MITIGATIONS, PERMITTED_MITIGATIONS
    from modelrig.economics import break_even_k, dedup_ceiling, dedup_ratio
    from modelrig.ir import SpecIR
    from modelrig.licence import DataRights
    from modelrig.primitives import TaskPrimitive

    report = ConformanceReport()

    # §8 barrier 1: the seed reference is in the cache key. Always.
    report.checks_run += 1
    if "seed_data_ref" not in CACHE_REQUIRED or "seed_data_ref" in CACHE_EXCLUDED:
        report.add(check="seed_ref_in_cache_key", subject="h_cache", source="P5-08",
                   detail=("data.seed_ref must be in the cache key: without it two "
                           "customers with identical requirements and different "
                           "confidential corpora collide, and the second is served a "
                           "model trained on the first customer's documents"))

    # ...and demonstrated, not merely declared.
    report.checks_run += 1

    def _spec(seed: str) -> SpecIR:
        return SpecIR(task_primitive=TaskPrimitive.EXTRACT, seed_data_ref=seed,
                      data_rights=DataRights.CUSTOMER_OWNED)

    if h_cache(_spec("s3://a/forms")) == h_cache(_spec("s3://b/forms")):
        report.add(check="different_data_different_key", subject="h_cache",
                   source="P5-08",
                   detail="two corpora hashed to one cache key: builds would collide")

    # §8 barrier 2: the owner check refuses a cross-owner hit on private data.
    report.checks_run += 1
    if lookup_allowed(requester="beta", owner="acme", hit_is_public=False):
        report.add(check="owner_check", subject="lookup_allowed", source="P5-08",
                   detail=("a cross-owner cache hit on private data must be refused; "
                           "this is the second of two deliberately redundant barriers"))

    # §7: the two hashes must actually differ, or the cache never hits.
    report.checks_run += 1
    if not CACHE_EXCLUDED:
        report.add(check="two_hashes", subject="CACHE_EXCLUDED", source="P5-07",
                   detail=("h_cache excludes nothing, so it equals h_ident and the "
                           "hit rate collapses to zero"))

    # §15: no mitigation may appear in both lists, and the forbidden ones must
    # stay forbidden — each of them lets the corpus shrink, which voids I-03.
    report.checks_run += 1
    overlap = set(PERMITTED_MITIGATIONS) & set(FORBIDDEN_MITIGATIONS)
    if overlap:
        report.add(check="corpus_mitigations", subject=", ".join(sorted(overlap)),
                   source="P5-15",
                   detail=("a mitigation cannot be both permitted and forbidden; "
                           "anything that lets the real corpus shrink voids the "
                           "collapse guarantee"))

    # §2: the dedup ceiling is the size ratio, and D must approach it from below.
    report.checks_run += 1
    ceiling = dedup_ceiling()
    if dedup_ratio(1e6) > ceiling or dedup_ratio(1e6) < 0.99 * ceiling:
        report.add(check="dedup_ceiling", subject="dedup_ratio", source="P5-02",
                   detail=(f"D must approach {ceiling:.1f} from below as k grows; "
                           f"got {dedup_ratio(1e6):.2f}"))

    # §3: break-even sits just above 1, so CAS is worse for the first cartridge.
    # If this ever drifts far from 1 the storage model has changed underneath.
    report.checks_run += 1
    if not 1.0 < break_even_k() < 1.5:
        report.add(check="dedup_break_even", subject="break_even_k", source="P5-03",
                   detail=(f"break-even k = {break_even_k():.2f}; it should sit just "
                           "above 1, and CAS should cost slightly more for the first "
                           "cartridge on a base"))

    return report


# --------------------------------------------------------------------------- #
def check_planner_conformance() -> ConformanceReport:
    """Assert Part 2's structural claims still hold over the live catalogue.

    Both of these are load-bearing arguments rather than implementation details,
    and both can be invalidated by a catalogue change that reports no error
    anywhere else.
    """
    from modelrig.planner.audit import plan_space_size
    from modelrig.planner.predicates import ALL_PREDICATES, HARD, SOFT, ordered_predicates

    report = ConformanceReport()

    # §2.2: the whole algorithm rests on the plan space being enumerable. A
    # catalogue that grew by orders of magnitude would invalidate the argument
    # for rejecting search while every test still passed.
    space = plan_space_size()
    report.checks_run += 1
    if not space["enumerable"]:
        report.add(check="plan_space_enumerable", subject="catalogue", source="P2-02",
                   detail=(f"the plan space is 10^{space['log10']} points; beyond 10^8 "
                           "exhaustive enumeration stops being the correct algorithm "
                           "and the argument for rejecting search fails"))

    # §16: the partition must stay a partition. A predicate in neither set is
    # one whose refusals cannot be attributed, and one in both makes the
    # stratified report incoherent.
    report.checks_run += 1
    names = {p.name for p in ALL_PREDICATES}
    if set(HARD) & set(SOFT) or set(HARD) | set(SOFT) != names:
        report.add(check="soundness_partition", subject="predicates", source="P2-16",
                   detail=(f"HARD and SOFT must partition {sorted(names)}; got "
                           f"hard={sorted(HARD)} soft={sorted(SOFT)}"))

    # §16: the predicates sound by construction must stay marked hard. If one
    # drifts into SOFT its refusals start being counted as evidence about
    # calibration, which they are not.
    for name in ("P_ram", "P_tok", "P_lic", "P_off"):
        report.checks_run += 1
        if name not in HARD:
            report.add(check="sound_by_construction", subject=name, source="P2-16",
                       detail=(f"{name} is sound by construction; marking it soft would "
                               "put its certain refusals into the calibration numbers"))

    # §4: the ordering must actually be sorted by c/(1-rho), or the early exit
    # saves less than it reports.
    report.checks_run += 1
    order = ordered_predicates()
    keys = [p.cost / max(1.0 - p.pass_rate, 1e-9) for p in order]
    if keys != sorted(keys):
        report.add(check="predicate_ordering", subject="ordered_predicates",
                   source="P2-04",
                   detail=("predicates must be evaluated cheapest-and-most-"
                           "discriminating first: ascending in c/(1-rho)"))

    return report


# --------------------------------------------------------------------------- #
def check_device_verification_conformance() -> ConformanceReport:
    """Assert Part 3's rules about what a device claim may say.

    These fail quietly. A ledger that answers "certified" for a device it never
    saw still returns a cartridge, and a container the planner offers but no
    accelerator can load still passes every other gate.
    """
    from modelrig.certification import (
        MIN_EVAL_SUBSET,
        MIN_OUTPUT_PARITY,
        CertificationLedger,
        VerificationSource,
    )
    from modelrig.planner.core import TARGETS
    from modelrig.probe import Tier
    from modelrig.quantformat import _TARGETS_BY_ACCELERATOR, target_formats_for

    report = ConformanceReport()

    # §7: only a measurement licenses a commitment. Every non-measured source
    # must refuse to promise, at every tier.
    for source in VerificationSource:
        report.checks_run += 1
        if source.may_promise != (source.tier is Tier.MEASURED):
            report.add(check="only_measured_may_promise", subject=source.value,
                       source="P3-07",
                       detail=(f"{source.value} is tier {source.tier.name} but "
                               f"may_promise={source.may_promise}; only a run on the "
                               "actual silicon may commit to anything"))

    # §11: absence must read as absence. A ledger holding one device must not
    # answer for another, whatever else it holds.
    report.checks_run += 1
    empty = CertificationLedger()
    if empty.certified_for("any-device") or "UNVERIFIED" not in empty.status_for("x"):
        report.add(check="absence_is_reported", subject="CertificationLedger",
                   source="P3-11",
                   detail="an empty ledger must report every device as unverified")

    # §11: the thresholds must actually constrain. A zero floor certifies runs
    # that measured nothing.
    report.checks_run += 1
    if MIN_EVAL_SUBSET < 1 or not 0.0 < MIN_OUTPUT_PARITY <= 1.0:
        report.add(check="certification_thresholds", subject="certification",
                   source="P3-11",
                   detail=(f"eval subset floor {MIN_EVAL_SUBSET} and parity floor "
                           f"{MIN_OUTPUT_PARITY} must both bind"))

    # §10: every container an accelerator can load must be a target the planner
    # can actually build. Drift here means the planner offers a format nothing
    # downstream produces, and nothing else reports it.
    for accelerator, containers in _TARGETS_BY_ACCELERATOR.items():
        for container in containers:
            report.checks_run += 1
            if container not in TARGETS:
                report.add(check="container_is_buildable", subject=accelerator,
                           source="P3-10",
                           detail=(f"accelerator {accelerator!r} may load {container!r}, "
                                   "which is not in the planner's target set"))

    # §10: an offline build must remain plannable on every accelerator, or the
    # offline requirement silently excludes hardware rather than a format.
    for accelerator in _TARGETS_BY_ACCELERATOR:
        report.checks_run += 1
        if not target_formats_for(accelerator, offline=True):
            report.add(check="offline_container_exists", subject=accelerator,
                       source="P3-10",
                       detail=(f"no offline-capable container on {accelerator!r}: an "
                               "offline spec targeting it cannot be planned at all"))

    return report


def objective_threshold(tier, build_cost_micro: int = 40_000_000) -> float:
    from modelrig.planner.objective import refusal_threshold

    return refusal_threshold(build_cost_micro, tier)


def run_all(catalogue: Catalogue | None = None) -> ConformanceReport:
    """Every check, merged into one report."""
    compat = check_model_compatibility(catalogue)
    arch = check_architecture_conformance(catalogue)
    forge = check_elicitation_conformance()
    device = check_device_verification_conformance()
    planner = check_planner_conformance()
    registry = check_registry_conformance()
    fabric = check_fabric_conformance()
    proving = check_proving_ground_conformance()
    merged = ConformanceReport(
        findings=(compat.findings + arch.findings + forge.findings + device.findings
                  + planner.findings + registry.findings + fabric.findings
                  + proving.findings),
        checks_run=(compat.checks_run + arch.checks_run + forge.checks_run
                    + device.checks_run + planner.checks_run + registry.checks_run
                    + fabric.checks_run + proving.checks_run),
    )
    logger.info(
        "conformance: %d checks, %d errors, %d warnings",
        merged.checks_run, len(merged.errors), len(merged.warnings),
    )
    return merged
