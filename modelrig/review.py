"""Reproducible local review tools. No document upload, publishing or telemetry."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import statistics
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from modelrig import classifier
from modelrig.datasets import load_dataset, split_dataset
from modelrig.stats import wilson

SMS_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def environment() -> dict:
    packages = {}
    for name in ("numpy", "torch", "transformers", "peft", "bitsandbytes", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python": platform.python_version(), "os": platform.platform(),
              "packages": packages}
    if packages["torch"]:
        import torch

        result["cuda_available"] = torch.cuda.is_available()
        result["cuda_version"] = torch.version.cuda
        result["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return result


def prepare_sms(path: Path) -> dict:
    """Download the public UCI dataset; discard every conflicting group, not just a label."""
    with urllib.request.urlopen(SMS_URL, timeout=60) as response:
        raw = response.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        rows = [line.split("\t", 1) for line in
                archive.read("SMSSpamCollection").decode("utf-8").splitlines() if line]
    groups: dict[str, list] = defaultdict(list)
    for label, text in rows:
        groups[" ".join(text.casefold().split())].append((text, label))
    conflicts = [key for key, values in groups.items() if len({v[1] for v in values}) > 1]
    clean = [values[0] for key, values in groups.items() if key not in conflicts]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps({"text": t, "label": label}) for t, label in clean),
                    encoding="utf-8")
    manifest = {
        "dataset": "UCI SMS Spam Collection", "url": SMS_URL,
        "citation": "https://doi.org/10.24432/C5CC84", "license": "CC BY 4.0 (UCI listing)",
        "download_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_rows": len(rows), "unique_rows": len(clean), "conflicting_groups": len(conflicts),
        "removed_rows": len(rows) - len(clean),
        "labels": dict(Counter(label for _, label in clean)),
        "preprocessing": "casefold and whitespace key; deduplicate; discard conflicting groups",
    }
    write_json(path.with_suffix(".provenance.json"), manifest)
    return manifest


def metrics(preds: list[str], gold: list[str]) -> dict:
    if len(preds) != len(gold) or not gold:
        raise ValueError("one prediction per nonempty test example required")
    correct = sum(p == g for p, g in zip(preds, gold, strict=True))
    interval = wilson(correct, len(gold))
    labels = sorted(set(gold))
    matrix = [[sum(g == a and p == b for p, g in zip(preds, gold, strict=True))
               for b in labels] for a in labels]
    f1 = []
    for i in range(len(labels)):
        tp = matrix[i][i]
        denominator = sum(matrix[i]) + sum(row[i] for row in matrix)
        f1.append(2 * tp / denominator if denominator else 0.0)
    return {"accuracy": correct / len(gold), "macro_f1": statistics.mean(f1),
            "accuracy_ci95": [interval.low, interval.high], "n_test": len(gold),
            "labels": labels, "confusion_matrix": matrix}


def graph_experiment() -> dict:
    """A planted-case regression experiment, not an empirical security guarantee."""
    from majestic.fabric.analyser import analyse
    from majestic.fabric.graph import FabricGraph, Node, NodeKind

    records = []
    for length in (1, 2, 4, 8, 16, 32):
        for untrusted in (False, True):
            for network in (False, True):
                for privileged in (False, True):
                    graph = FabricGraph(name=f"chain-{length}-{untrusted}-{network}-{privileged}")
                    graph.add(Node("input", NodeKind.INPUT, produces_untrusted=untrusted))
                    previous = "input"
                    for i in range(length):
                        name = f"step{i}"
                        graph.add(Node(name, NodeKind.CARTRIDGE,
                                       requires_network=network and i == 0))
                        graph.connect(previous, name)
                        previous = name
                    graph.add(Node("sink", NodeKind.TOOL, privileged=privileged))
                    graph.connect(previous, "sink")
                    start = time.perf_counter()
                    result = analyse(graph, offline_required=True)
                    elapsed = 1000 * (time.perf_counter() - start)
                    expected = not network and not (untrusted and privileged)
                    records.append({"name": graph.name, "nodes": len(graph.nodes),
                                    "expected_ok": expected, "actual_ok": result.ok,
                                    "elapsed_ms": elapsed})
    return {"cases": len(records), "correct": sum(r["expected_ok"] == r["actual_ok"]
                                                  for r in records),
            "scope": "synthetic straight-line graphs with declared effects; no runtime comparison",
            "records": records}


def benchmark(dataset: Path, out: Path) -> dict:
    corpus = load_dataset(str(dataset))
    records = []
    for seed in (0, 1, 2):
        train, test = split_dataset(corpus, 0.2, seed)
        labels = sorted({label for _, label in train})
        texts = [t for t, _ in test]
        gold = [label for _, label in test]
        majority = Counter(label for _, label in train).most_common(1)[0][0]
        records.append(dict(metrics([majority] * len(gold), gold), seed=seed,
                            method="majority", quantization="none", n_train=len(train)))
        for method, fit in (("centroid", classifier.fit_centroid), ("knn", classifier.fit_knn)):
            started = time.perf_counter()
            model = fit(train, labels)
            train_seconds = time.perf_counter() - started
            reference = classifier.predict(model, texts)
            for quantization in ("none", "int8", "int4"):
                compressed, compression = classifier.quantize_model(model, quantization)
                preds = classifier.predict(compressed, texts)
                timings = []
                for text in texts[:32]:
                    start = time.perf_counter()
                    classifier.predict(compressed, [text])
                    timings.append(1000 * (time.perf_counter() - start))
                records.append(dict(
                    metrics(preds, gold), seed=seed, method=method, quantization=quantization,
                    train_seconds=train_seconds, n_train=len(train), compression=compression,
                    flip_rate=sum(p != r for p, r in zip(preds, reference, strict=True))
                    / len(preds),
                    warm_p50_ms=statistics.median(timings), warm_max_ms=max(timings),
                    test_digest=hashlib.sha256(json.dumps(test).encode()).hexdigest(),
                    tensor_digest=classifier.model_digest(compressed),
                ))
                if seed == 0 and method == "centroid" and quantization == "int8":
                    classifier.save_model(compressed, out.parent / "review_model")
    report = {"schema_version": 1, "time_utc": datetime.now(timezone.utc).isoformat(),
              "environment": environment(), "dataset": dataset.name,
              "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
              "protocol": "3 fixed stratified 80/20 splits; no test-driven selection or repair",
              "limits": "single historical dataset; split replicates overlap; no field claims",
              "classification": records, "fabric": graph_experiment()}
    write_json(out, report)
    return report


def predict_artifact(path: Path, texts: list[str]) -> list[str]:
    from modelrig.integrity import verify_manifest

    if (path / "checksums.json").is_file():
        verify_manifest(path)
    if (path / "hf_model.json").is_file():
        from modelrig.training_hf import hf_predict

        record = json.loads((path / "hf_model.json").read_text(encoding="utf-8"))
        if not (path / record["model_dir"]).resolve().is_relative_to(path.resolve()):
            raise ValueError("model path must remain inside the artifact")
        record["model_dir"] = str(path / record["model_dir"])
        return hf_predict(record, texts)
    return classifier.predict(classifier.load_model(path), texts)


def import_artifact(source: Path, registry: Path) -> dict:
    """Verify and copy a portable neural artifact into the local studio registry."""
    import re
    import shutil

    from modelrig.integrity import verify_manifest
    from modelrig.registry import FileSystemRegistry

    verify_manifest(source)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    key = metadata["build_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", key):
        raise ValueError("invalid build identifier")
    if metadata.get("backend") != "hf_seqcls":
        raise ValueError("portable import currently supports HF classifiers")
    target = registry.resolve() / key
    if target.exists():
        verify_manifest(target)
        if (target / "checksums.json").read_bytes() != (source / "checksums.json").read_bytes():
            raise ValueError("a different artifact already uses this build identifier")
    else:
        shutil.copytree(source, target)
    FileSystemRegistry(registry).put(key, str(target), metadata)
    return {"build_id": key, "artifact_path": str(target)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    data = sub.add_parser("prepare-sms")
    data.add_argument("--out", type=Path, default=Path("data/sms.jsonl"))
    bench = sub.add_parser("benchmark")
    bench.add_argument("--dataset", type=Path, default=Path("data/sms.jsonl"))
    bench.add_argument("--out", type=Path, default=Path("docs/evidence/benchmark.json"))
    pred = sub.add_parser("predict")
    pred.add_argument("--path", type=Path, required=True)
    pred.add_argument("--text", action="append", required=True)
    install = sub.add_parser("import-artifact")
    install.add_argument("--path", type=Path, required=True)
    install.add_argument("--registry", type=Path, default=Path("registry"))
    args = parser.parse_args()
    if args.command == "doctor":
        result = environment()
    elif args.command == "prepare-sms":
        result = prepare_sms(args.out)
    elif args.command == "benchmark":
        report = benchmark(args.dataset, args.out)
        result = {"report": str(args.out), "runs": len(report["classification"]),
                  "graph_cases": report["fabric"]["cases"]}
    elif args.command == "import-artifact":
        result = import_artifact(args.path, args.registry)
    else:
        result = {"predictions": predict_artifact(args.path, args.text)}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
