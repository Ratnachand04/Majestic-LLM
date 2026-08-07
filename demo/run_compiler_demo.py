"""The C-02 worked example: plain language -> certified cartridge, offline.

    python demo/run_compiler_demo.py

Walks the ten acts end to end with the offline components: FORGE interviews,
the PLANNER compiles under hard constraints, the DATA FACTORY amplifies from a
locked split, the PROVING GROUND scores seven axes and measures answer-flip after
quantisation, and the REGISTRY admits a certified cartridge.

It also shows the two things that make this a compiler rather than a script
collection: an infeasible request is REFUSED before any GPU is allocated, and a
Fabric graph that secretly needs the network is caught statically.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from majestic.fabric import FabricGraph, Node, NodeKind, analyse  # noqa: E402
from modelrig.forge import Forge  # noqa: E402
from modelrig.ir import DataRights, SpecIR  # noqa: E402
from modelrig.pipeline import MajesticCompiler  # noqa: E402
from modelrig.primitives import TaskPrimitive  # noqa: E402
from modelrig.registry import CartridgeRegistry  # noqa: E402

_POS = ["good great work", "love this service", "happy with the result",
        "nice and helpful staff", "excellent quality throughout"]
_NEG = ["bad broken service", "hate this experience", "sad and useless outcome",
        "terrible unhelpful staff", "awful quality throughout"]

REQUEST = (
    "Classify incoming support tickets by sentiment on an android phone. "
    "Must work offline. We have 120 real tickets. Flag anything uncertain."
)


def corpus(n: int = 120) -> list[tuple[str, str]]:
    rows = []
    for i in range(n // 2):
        rows.append((f"{_POS[i % len(_POS)]} ticket {i}", "positive"))
        rows.append((f"{_NEG[i % len(_NEG)]} ticket {i}", "negative"))
    return rows


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="majestic_demo_"))

    # --- Acts 1-2: describe the need, four questions ------------------- #
    rule("ACT 1-2  FORGE — plain language to a typed specification")
    forge = Forge()
    state = forge.parse(REQUEST)
    print(f"request   : {REQUEST}")
    print("\nslots FORGE filled by itself:")
    for name, slot in state.slots.items():
        if slot.filled:
            print(f"  {name:<20} = {slot.value}  (confidence {slot.confidence:.2f})")
    print("\nquestions it still needs to ask (four, not forty):")
    for q in state.questions():
        print(f"  - {q}")

    state = forge.answer(
        state, data_rights=DataRights.CUSTOMER_OWNED, quality_gate=0.8,
        seed_data_count=120, offline_required=True,
    )
    spec = forge.to_spec(state)
    print(f"\nSpec IR hash: {spec.hash}  (identical hash -> served from cache)")

    # --- Acts 3-8: compile ---------------------------------------------- #
    rule("ACT 3-8  COMPILE — gates, data, training, proving, certification")
    compiler = MajesticCompiler(
        registry=CartridgeRegistry(workdir), base_path=workdir
    )
    result = compiler.compile(spec, corpus())

    for gate in result.gates:
        print(f"  [{'PASS' if gate.passed else 'FAIL'}] {gate.gate}")
        for reason in gate.reasons:
            print(f"          - {reason}")
    if result.plan:
        print(f"\nplan      : base={result.plan.base_ref} "
              f"peft={result.plan.peft_method} "
              f"quant={result.plan.quantiser}/{result.plan.bit_width} "
              f"target={result.plan.target}  ~${result.plan.budget_usd}")

    # --- Act 9: the scorecard is what is sold ---------------------------- #
    rule("ACT 9  THE SCORECARD — what the customer actually receives")
    if result.scorecard:
        for axis in result.scorecard.axes:
            mark = "ok  " if axis.passed else "FAIL"
            print(f"  [{mark}] {axis.name:<18} {axis.score:.3f}  (gate {axis.threshold:.2f})")
        print(f"\n  answer-flip rate vs FP16: {result.scorecard.answer_flip_rate} "
              "(aggregate parity is an illusion — A-05)")
        print(f"  measured on {result.scorecard.n_held_out} REAL held-out examples "
              "the model never trained on")
        if result.scorecard.honest_failures:
            print("\n  honest failure cases:")
            for f in result.scorecard.honest_failures:
                print(f"    {f['input']!r} -> got {f['got']!r}, expected {f['expected']!r}")

    # --- Act 10: registry + economics ------------------------------------ #
    rule("ACT 10  REGISTRY — certified, content-addressed, deduplicated")
    if result.admitted:
        print(f"  cartridge : {result.cartridge_id}")
        print(f"  certified : {result.cartridge.certified}")
        print(f"  licence   : {result.cartridge.licence_chain.get('resolved_licence')}")
        cached = compiler.compile(spec, corpus())
        print(f"  rebuild   : cache_hit={cached.cache_hit} (zero marginal cost)")
        stats = compiler.registry.stats()
        print(f"  storage   : {stats.cartridges} cartridge(s), "
              f"{stats.distinct_bases} base stored once, dedup {stats.dedup_ratio}x")
    else:
        print(f"  REFUSED   : {result.refusal}")

    # --- the refusals that cost nothing ---------------------------------- #
    rule("GATE-BEFORE-GPU — an infeasible request is refused in milliseconds")
    impossible = SpecIR(
        task_primitive=TaskPrimitive.GENERATE,
        device_target="android_lowend",
        seed_data_count=5,
        data_rights=DataRights.CUSTOMER_OWNED,
        quality_gate=0.95,
    )
    refused = compiler.compile(impossible, corpus())
    print(f"  admitted  : {refused.admitted}")
    print(f"  stage     : {refused.stage_reached}  (no planning, no data, no GPU)")
    for reason in refused.refusal.split("; "):
        print(f"    - {reason}")

    # --- offline closure, proven statically ------------------------------ #
    rule("FABRIC — offline closure proven before the graph ever runs")
    graph = FabricGraph("requisition-flow")
    graph.add(Node("req-extractor", NodeKind.CARTRIDGE, ram_mb=30,
                   metadata={"base_ref": "shared", "base_ram_mb": 1100}))
    graph.add(Node("urgency-classifier", NodeKind.CARTRIDGE, ram_mb=30,
                   metadata={"base_ref": "shared", "base_ram_mb": 1100}))
    graph.add(Node("doctor-notify", NodeKind.TOOL, requires_network=True, privileged=True))
    graph.connect("req-extractor", "urgency-classifier")
    graph.connect("urgency-classifier", "doctor-notify")

    analysis = analyse(graph, offline_required=True, ram_budget_mb=2400)
    print(f"  offline_closed : {analysis.offline_closed}")
    print(f"  peak_ram_mb    : {analysis.peak_ram_mb} "
          "(one shared base + two adapters)")
    for v in analysis.violations:
        print(f"    VIOLATION: {v}")
    print(f"    remedy   : {analysis.suggest_split()}")

    print(f"\nartefacts written to {workdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
