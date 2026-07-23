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
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
