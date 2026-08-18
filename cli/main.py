"""Command-line entry point for Majestic LLM + ModelRig.

Commands: ``info``, ``validate-spec``, ``build``, ``feasibility``. Runs fully
offline with the default tiny/mock components.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

# Allow ``python cli/main.py`` as well as ``python -m cli.main``.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="majestic", description="Majestic LLM CLI")
    sub = p.add_subparsers(dest="command")

    sub.add_parser("info", help="Print project info, registered artifacts, devices")

    v = sub.add_parser("validate-spec", help="Validate a ModelRig BuildSpec file")
    v.add_argument("path")

    b = sub.add_parser("build", help="Run a ModelRig build")
    b.add_argument("--spec", help="Path to a BuildSpec (.json/.yaml)")
    b.add_argument("--capability", help="Distill a capability (e.g., sentiment)")
    b.add_argument("--device", help="Target device name (see configs/devices.yaml)")
    b.add_argument("--registry", default="./registry", help="Registry base path")

    f = sub.add_parser("feasibility", help="Check device feasibility of a spec")
    f.add_argument("--spec", help="Path to a BuildSpec (.json/.yaml)")
    f.add_argument("--capability", help="Capability name (uses a distilled spec)")
    f.add_argument("--device", required=True, help="Target device name")

    fo = sub.add_parser("forge", help="Turn a plain-language need into a Spec IR")
    fo.add_argument("description", help="What the model should do, in plain language")
    fo.add_argument("--out", help="Write the Spec IR here (.json)")
    fo.add_argument("--offline", action="store_true", help="Answer the offline slot")
    fo.add_argument("--device", help="Answer the device slot")
    fo.add_argument("--seed-count", type=int, help="How many real examples exist")
    fo.add_argument("--tier", default="commercial",
                    choices=("experimental", "commercial", "regulated"),
                    help="Customer tier: sets how many questions are worth asking")
    fo.add_argument("--explain", action="store_true",
                    help="Also show the questions the planner proved were pointless")

    c = sub.add_parser("compile", help="Compile a Spec IR into a certified cartridge")
    c.add_argument("--spec", required=True, help="Path to a Spec IR (.json/.yaml)")
    c.add_argument("--data", help="Training corpus (.jsonl/.csv); builtin if omitted")
    c.add_argument("--registry", default="./registry", help="Registry base path")

    pr = sub.add_parser("primitives", help="List the supported task primitives")

    fb = sub.add_parser("verify-graph", help="Statically verify a Fabric graph (.json)")
    fb.add_argument("path")
    fb.add_argument("--offline", action="store_true", help="Require offline closure")

    va = sub.add_parser(
        "validate", help="Check model compatibility and architecture conformance"
    )
    va.add_argument("--warnings", action="store_true", help="Also print warnings")

    pl = sub.add_parser("plan", help="Run the Planner on a Spec IR: a plan, or a refusal")
    pl.add_argument("--spec", required=True, help="Path to a Spec IR (.json/.yaml)")
    pl.add_argument("--tier", default="commercial",
                    choices=["experimental", "commercial", "regulated"],
                    help="Sets V and kappa, hence the refusal threshold theta*")
    pl.add_argument("--explain", action="store_true",
                    help="Show predicate order, counts and the decision arithmetic")

    pb = sub.add_parser(
        "probe", help="Turn two on-device benchmarks into a measured DeviceProfile"
    )
    pb.add_argument("--device-id", required=True, help="Stable identifier for the unit")
    pb.add_argument("--small-mb", type=int, default=200, help="Smaller probe model size")
    pb.add_argument("--large-mb", type=int, default=400, help="Larger probe model size")
    pb.add_argument("--small-toks", type=float, required=True, help="tok/s at --small-mb")
    pb.add_argument("--large-toks", type=float, required=True, help="tok/s at --large-mb")
    pb.add_argument("--sustained-toks", type=float,
                    help="tok/s at --large-mb after the 180 s sustained phase")
    pb.add_argument("--ram-total-mb", type=int, default=4000)
    pb.add_argument("--ram-free-mb", type=int, required=True,
                    help="FREE RAM, measured — not total")
    pb.add_argument("--simd", default="neon,dotprod",
                    help="Comma-separated SIMD flags; selects the quantisation format")
    pb.add_argument("--accelerator", default="cpu")
    pb.add_argument("--out", help="Write the DeviceProfile here (.json)")

    bg = sub.add_parser("budget", help="Compute an on-device RAM budget")
    bg.add_argument("--device", default="android_tablet_4gb")
    bg.add_argument("--ram-gb", type=float, default=4.0)
    bg.add_argument("--base-params-b", type=float, default=1.7)
    bg.add_argument("--context", type=int, default=2048)
    bg.add_argument("--adapters", type=int, default=20)
    _ = pr
    return p


def _cmd_info() -> int:
    from majestic import __version__
    from modelrig.feasibility import YamlDeviceProfiler
    from modelrig.registry import FileSystemRegistry

    print(f"Majestic LLM v{__version__} — compound AI system + ModelRig factory")
    reg = FileSystemRegistry(os.environ.get("MAJESTIC_MODEL_REGISTRY_PATH", "./registry"))
    artifacts = reg.list()
    print(f"registry: {len(artifacts)} artifact(s)")
    for key in artifacts:
        print(f"  - {key}")
    devices = [d.name for d in YamlDeviceProfiler().all()]
    print(f"devices: {', '.join(devices) if devices else '(none)'}")
    return 0


def _cmd_validate_spec(path: str) -> int:
    from modelrig.buildspec import load_spec, validate_spec

    try:
        spec = load_spec(path)
    except Exception as exc:  # noqa: BLE001 - report cleanly to the user
        print(f"error: could not load spec: {exc}")
        return 2
    errors = validate_spec(spec)
    if errors:
        print("INVALID:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"OK: valid BuildSpec for task={spec.task!r} method={spec.method.value}")
    return 0


def _print_result(result) -> None:
    print(f"build_id : {result.build_id}")
    print(f"success  : {result.success}")
    print(f"eval     : {result.eval_report}")
    if result.artifact_path:
        print(f"artifact : {result.artifact_path}")
    if result.reason:
        print(f"reason   : {result.reason}")


def _cmd_build(args: argparse.Namespace) -> int:
    from modelrig.buildspec import load_spec
    from modelrig.factory import Factory

    factory = Factory(base_path=args.registry)
    if args.spec:
        spec = load_spec(args.spec)
        result = factory.build(spec)
    elif args.capability and args.device:
        from modelrig.teacher_loop import DistillationLoop

        loop = DistillationLoop(
            factory=factory, data_dir=pathlib.Path(args.registry) / "datasets"
        )
        result = loop.distill_and_build(args.capability, args.device)
    else:
        print("error: provide --spec, or both --capability and --device")
        return 2
    _print_result(result)
    return 0 if result.success else 1


def _cmd_feasibility(args: argparse.Namespace) -> int:
    from modelrig.buildspec import BuildSpec, TrainingMethod, load_spec
    from modelrig.feasibility import HeuristicFeasibilityEngine, YamlDeviceProfiler

    engine = HeuristicFeasibilityEngine()
    if args.spec:
        spec = load_spec(args.spec)
        if spec.device is None:
            spec.extras["device"] = args.device
    elif args.capability:
        # A representative distilled spec (no data needed just to check fit).
        spec = BuildSpec(
            task=args.capability, base_model="centroid",
            method=TrainingMethod.CENTROID, quantization="int8", runtime="npz",
            device=YamlDeviceProfiler().profile(args.device),
        )
    else:
        print("error: provide --spec or --capability")
        return 2

    verdict = engine.evaluate(spec)
    print(f"feasible : {verdict.feasible}")
    print(f"estimate : ram={verdict.estimate.ram_mb}MB "
          f"latency={verdict.estimate.latency_ms}ms "
          f"battery={verdict.estimate.battery_pct_per_hr}%/hr")
    print(f"budget   : ram={verdict.ram_budget_mb}MB headroom={verdict.ram_headroom_mb}MB")
    for r in verdict.reasons:
        print(f"  - {r}")
    return 0 if verdict.feasible else 1


def _cmd_primitives() -> int:
    from modelrig.primitives import all_primitives

    print("Supported task primitives (the closed set of eight):")
    for p in all_primitives():
        print(f"  {p.primitive.value:<10} seed floor {p.seed_floor:>4}  "
              f"metric {p.default_metric:<16} min {p.min_params_b}B  — {p.description}")
    print("\nGAP-06: coverage against real requests is not yet validated.")
    return 0


def _cmd_forge(args: argparse.Namespace) -> int:
    from modelrig.forge import Interviewer, unasked_because_irrelevant, value_ratio
    from modelrig.ir import save_spec_ir
    from modelrig.planner.objective import Tier

    answers: dict[str, object] = {}
    if args.offline:
        answers["offline_required"] = True
    if args.device:
        answers["device_target"] = args.device
    if args.seed_count is not None:
        answers["seed_data_count"] = args.seed_count

    tier = Tier(args.tier)
    iv = Interviewer(tier=tier).conduct(args.description, answers=answers)

    if not iv.complete:
        print(f"cannot emit a spec yet: {len(iv.pending)} slot(s) unfilled")
        for q in iv.pending:
            print(f"  - {q.text}")
        return 2

    spec = iv.spec
    print(f"spec_hash : {spec.hash}")
    print(f"primitive : {spec.task_primitive.value}")
    print(f"device    : {spec.device_target}")
    print(f"offline   : {spec.offline_required}")
    print(f"seed data : {spec.seed_data_count}")

    if iv.pending:
        # Ranked by measured information gain — how far each answer moves the
        # plan — not by how uncertain the parser feels about it.
        print(f"\nstill to ask ({tier.value} tier, lambda={value_ratio(tier):.2f}, "
              f"P(complete)={iv.completion_probability:.0%}):")
        for q in iv.pending:
            flag = "must" if q.must_ask else f"{q.value:.2f} > {q.threshold}"
            print(f"  [{flag:>12}] {q.text}")
            print(f"                 {q.rationale}")

    irrelevant = unasked_because_irrelevant(iv)
    if irrelevant and args.explain:
        print("\nnot asked — the planner proved the answer cannot move the plan:")
        for slot, why in irrelevant:
            print(f"  - {slot}: {why}")

    if args.out:
        save_spec_ir(spec, args.out)
        print(f"\nwrote {args.out}")
    return 0


def _cmd_compile(args: argparse.Namespace) -> int:
    from modelrig.datasets import load_dataset
    from modelrig.ir import load_spec_ir
    from modelrig.pipeline import MajesticCompiler

    try:
        spec = load_spec_ir(args.spec)
    except Exception as exc:  # noqa: BLE001 - report cleanly
        print(f"error: could not load spec: {exc}")
        return 2

    corpus = load_dataset(args.data) if args.data else load_dataset("builtin:sentiment")
    result = MajesticCompiler(base_path=args.registry).compile(spec, corpus)

    print(f"spec_hash : {result.spec.hash}")
    print(f"admitted  : {result.admitted}")
    print(f"stage     : {result.stage_reached}")
    print(f"cache hit : {result.cache_hit}")
    for gate in result.gates:
        status = "PASS" if gate.passed else "FAIL"
        print(f"  [{status}] {gate.gate}")
        for reason in gate.reasons:
            print(f"        - {reason}")
    if result.scorecard:
        print("scorecard :")
        for axis in result.scorecard.axes:
            mark = "ok " if axis.passed else "FAIL"
            print(f"  [{mark}] {axis.name:<18} {axis.score:.3f} (>= {axis.threshold:.2f})")
        print(f"  answer-flip rate: {result.scorecard.answer_flip_rate}")
    if result.cartridge_id:
        print(f"cartridge : {result.cartridge_id}")
    if result.refusal:
        print(f"refused   : {result.refusal}")
    for suggestion in result.repair_suggestions:
        print(f"  try: {suggestion}")
    return 0 if result.admitted else 1


def _cmd_verify_graph(args: argparse.Namespace) -> int:
    import json as _json

    from majestic.fabric import FabricGraph, Node, NodeKind, analyse

    data = _json.loads(pathlib.Path(args.path).read_text(encoding="utf-8-sig"))
    graph = FabricGraph(data.get("name", "graph"))
    for node in data.get("nodes", []):
        graph.add(Node(
            name=node["name"],
            kind=NodeKind(node.get("kind", "cartridge")),
            requires_network=bool(node.get("requires_network", False)),
            produces_untrusted=bool(node.get("produces_untrusted", False)),
            privileged=bool(node.get("privileged", False)),
            ram_mb=float(node.get("ram_mb", 0.0)),
            cost_usd=float(node.get("cost_usd", 0.0)),
            metadata=node.get("metadata", {}),
        ))
    for src, dst in data.get("edges", []):
        graph.connect(src, dst)

    result = analyse(graph, offline_required=args.offline)
    print(f"offline_closed : {result.offline_closed}")
    print(f"peak_ram_mb    : {result.peak_ram_mb}")
    print(f"cost_per_req   : ${result.total_cost_usd}")
    for v in result.violations:
        print(f"  VIOLATION: {v}")
    for w in result.warnings:
        print(f"  warning  : {w}")
    split = result.suggest_split()
    if split:
        print(f"  remedy   : {split}")
    return 0 if result.ok else 1


def _cmd_validate(args: argparse.Namespace) -> int:
    from modelrig.conformance import run_all

    report = run_all()
    summary = report.summary()
    print(f"checks run : {summary['checks_run']}")
    print(f"errors     : {summary['errors']}")
    print(f"warnings   : {summary['warnings']}")
    for finding in report.errors:
        print(f"  ERROR [{finding.source or '-'}] {finding.check} "
              f"({finding.subject}): {finding.detail}")
    if args.warnings:
        for finding in report.warnings:
            print(f"  warn  [{finding.source or '-'}] {finding.check} "
                  f"({finding.subject}): {finding.detail}")
    print("RESULT     : " + ("conformant" if report.ok else "NON-CONFORMANT"))
    return 0 if report.ok else 1


def _cmd_plan(args: argparse.Namespace) -> int:
    from modelrig.ir import load_spec_ir
    from modelrig.planner import Tier, default_catalog, ordered_predicates
    from modelrig.planner import plan as run_plan
    from modelrig.planner.costmodel import usd

    try:
        spec = load_spec_ir(args.spec)
    except Exception as exc:  # noqa: BLE001 - report cleanly
        print(f"error: could not load spec: {exc}")
        return 2

    outcome = run_plan(spec, default_catalog(), tier=Tier(args.tier))
    print(f"spec_hash : {spec.hash}")
    print(f"primitive : {spec.task_primitive.value}")
    print(f"device    : {spec.device_target}")

    if args.explain:
        print(f"order     : {' -> '.join(p.name for p in ordered_predicates())}")
        print(f"considered: {outcome.considered} candidate plans")
        print(f"pred-evals: {outcome.predicate_evaluations} "
              f"({outcome.predicate_evaluations / max(outcome.considered, 1):.1f} per candidate "
              "— early exit means well under seven)")

    if outcome.admitted:
        p = outcome.plan
        print(f"\nADMITTED  : {p.base_ref}")
        print(f"  teacher : {p.teacher_ref or '-'}  ({p.distil_mode}, derived from tokenizer)")
        print(f"  method  : {p.peft_method} r{p.rank}")
        print(f"  quant   : {p.quantiser}/{p.bit_width}   target {p.target}")
        print(f"  cost    : ${usd(outcome.cost_micro):.2f}")
        print(f"  quality : {outcome.predicted_quality:.2f} >= theta* {outcome.threshold:.2f}")
        print(f"  rule    : {p.provenance.get('rule')}")
        if outcome.candidates:
            print(f"  parallel: {len(outcome.candidates)} extra candidate(s), opt-in")
        return 0

    print()
    print(outcome.refusal.render())
    return 1


def _cmd_probe(args: argparse.Namespace) -> int:
    from modelrig.probe import MB, ProbePoint, build_profile
    from modelrig.quantformat import select_format, sweep_plan

    simd = [s.strip().lower() for s in args.simd.split(",") if s.strip()]
    points = [
        ProbePoint(args.small_mb * MB, args.small_toks),
        ProbePoint(args.large_mb * MB, args.large_toks),
    ]
    try:
        profile = build_profile(
            args.device_id, points,
            ram_total_mb=args.ram_total_mb, ram_free_mb=args.ram_free_mb,
            sustained_tokens_per_s=args.sustained_toks,
            burst_reference_tokens_per_s=args.large_toks,
            simd=simd, accelerator=args.accelerator,
        )
    except Exception as exc:  # noqa: BLE001 - report cleanly
        print(f"probe failed: {exc}")
        return 2

    print(f"device_id     : {profile.device_id}")
    print(f"BW_eff        : {profile.bw_eff_gbps:.2f} GB/s")
    print(f"overhead      : {profile.overhead_ms_per_token:.2f} ms/token")
    print(f"thermal derate: {profile.thermal_derate_180s:.2f}"
          f"{'  (no sustained phase run)' if args.sustained_toks is None else ''}")
    print(f"free RAM      : {profile.ram_free_mb} MB "
          f"(usable {profile.usable_ram_mb} MB after headroom)")
    print(f"tier          : {profile.tier.name.lower()} — "
          f"{'may promise' if profile.tier.may_promise else 'refusal only'}")

    choice = select_format(simd, args.accelerator)
    print(f"\nquant format  : {choice.name} ({choice.evidence})")
    print(f"  {choice.reason}")
    if choice.rejected:
        print(f"  unavailable : {', '.join(choice.rejected)}")
    print(f"  sweep next  : {', '.join(sweep_plan(simd, args.accelerator)['formats'])}")

    print("\npredicted sustained decode across the catalogue:")
    for mb in (400, 800, 1120, 2600, 5300):
        rate = profile.tokens_per_s(mb * MB)
        reach = profile.calibration.extrapolation_factor(mb * MB)
        flag = "" if reach <= 10 else f"  <- {reach:.1f}x outside the probe bracket"
        print(f"  {mb:>5} MB  {rate:6.2f} tok/s{flag}")

    if args.out:
        profile.save(args.out)
        print(f"\nwrote {args.out}")
    return 0


def _cmd_budget(args: argparse.Namespace) -> int:
    from majestic.serving import plan_device_deployment, swap_latency_ms

    budget, problem = plan_device_deployment(
        base_params_b=args.base_params_b,
        total_gb=args.ram_gb,
        context_length=args.context,
        n_adapters=args.adapters,
    )
    print(f"device: {args.device}  ({args.ram_gb} GB total)")
    for label, gb in budget.as_table():
        print(f"  {label:<40} {gb:>6.2f} GB")
    print(f"  {'headroom':<40} {budget.headroom_gb:>6.2f} GB")
    swap = swap_latency_ms(args.adapters)
    print(f"\nadapter swap : ~{swap['assumed_ms_per_swap']:.0f} ms each "
          f"(measured={swap['measured']})")
    print(f"caveat       : {swap['caveat']}")
    if problem:
        print(f"\nPROBLEM: {problem}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "info":
        return _cmd_info()
    if args.command == "validate-spec":
        return _cmd_validate_spec(args.path)
    if args.command == "build":
        return _cmd_build(args)
    if args.command == "feasibility":
        return _cmd_feasibility(args)
    if args.command == "primitives":
        return _cmd_primitives()
    if args.command == "forge":
        return _cmd_forge(args)
    if args.command == "compile":
        return _cmd_compile(args)
    if args.command == "verify-graph":
        return _cmd_verify_graph(args)
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "plan":
        return _cmd_plan(args)
    if args.command == "probe":
        return _cmd_probe(args)
    if args.command == "budget":
        return _cmd_budget(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
