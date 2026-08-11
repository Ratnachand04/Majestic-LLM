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


def run_all(catalogue: Catalogue | None = None) -> ConformanceReport:
    """Both checks, merged into one report."""
    compat = check_model_compatibility(catalogue)
    arch = check_architecture_conformance(catalogue)
    merged = ConformanceReport(
        findings=compat.findings + arch.findings,
        checks_run=compat.checks_run + arch.checks_run,
    )
    logger.info(
        "conformance: %d checks, %d errors, %d warnings",
        merged.checks_run, len(merged.errors), len(merged.warnings),
    )
    return merged
