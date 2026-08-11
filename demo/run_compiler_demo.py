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

from majestic.fabric import (  # noqa: E402
    FabricGraph,
    FabricRuntime,
    Node,
    NodeKind,
    analyse,
    split_offline_core,
)
from majestic.serving import (  # noqa: E402
    AdapterHandle,
    ServerTopology,
    device_budget,
    swap_latency_ms,
)
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
        print(f"rule      : {result.plan.provenance.get('rule')}")

    # --- Acts 5-6: candidates trained in parallel, then one is chosen ---- #
    if result.selection and result.selection.candidates:
        rule("ACT 5-6  CANDIDATES — train in parallel, score both, pick one")
        for row in result.selection.comparison_table():
            mark = "<- winner" if row["winner"] else ""
            print(f"  {row['base']:<40} {row['params_b']:>5}B  "
                  f"score {row['score']:.3f}  {row['latency_ms']:>7.1f}ms  {mark}")
        print(f"\n  rationale: {result.selection.rationale}")
    if result.quantisation:
        q = result.quantisation
        print(f"\n  quantisation: {q['quantiser']}/{q['bit_width']} "
              f"flip={q['answer_flip_rate']} "
              f"calibrated on {q['calibration_size']} customer samples "
              f"(on-distribution={q['calibration_on_distribution']})")

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

    # --- the runtime the cartridge lands in ------------------------------ #
    rule("B-09  DEVICE BUDGET — twenty specialists on one 4 GB tablet")
    budget = device_budget(total_gb=4.0, base_params_b=1.7, context_length=2048,
                           n_adapters=20)
    for label, gb in budget.as_table():
        print(f"  {label:<40} {gb:>6.2f} GB")
    print(f"  {'headroom':<40} {budget.headroom_gb:>6.2f} GB")
    print("\n  Twenty specialists cost 0.59 GB because they SHARE the one "
          "resident base.")
    print(f"  Swap latency ~{swap_latency_ms(1)['assumed_ms_per_swap']:.0f}ms "
          "per adapter — UNMEASURED (GAP-10).")

    rule("A-04  SERVING — one GPU, many tenants, and what breaks it")
    topo = ServerTopology(base_ref="Qwen/Qwen3-1.7B")
    print(f"  tenants per A100        : {topo.max_tenants()}")
    mixed = [AdapterHandle(f"t{i}", "Qwen/Qwen3-1.7B") for i in range(3)]
    mixed.append(AdapterHandle("t9", "meta-llama/Llama-3.2-1B-Instruct"))
    out = topo.batch(mixed)
    print(f"  batched in one kernel   : {out['batched']}")
    print(f"  rejected (wrong base)   : {out['rejected_wrong_base']}  "
          "<- customers on different bases cannot share a GPU")
    print(f"  {topo.pool.fragmentation_report()['note']}")

    # --- the graph now RUNS, not just analyses --------------------------- #
    rule("B-10  FABRIC RUNTIME — the offline core actually executes")
    core, tail = split_offline_core(graph)
    print(f"  offline core : {core}")
    print(f"  online tail  : {tail}")
    core_graph = FabricGraph("offline-core")
    for name in core:
        core_graph.add(graph.nodes[name])
    for src, dst in graph.edges:
        if src in core and dst in core:
            core_graph.connect(src, dst)
    runtime = FabricRuntime(
        core_graph,
        {"req-extractor": lambda x, c: f"{x}|fields",
         "urgency-classifier": lambda x, c: f"{x}|urgent"},
        offline=True,
    )
    run = runtime.run("lab form scan")
    print(f"  ran offline  : ok={run.ok} output={run.output!r} "
          f"in {run.total_ms:.1f}ms")

    print(f"\nartefacts written to {workdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
