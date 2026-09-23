"""Repeat the fixed LoRA experiment and exercise real QLoRA/quantized export.

Run from the repository root with the local environment. Reuses completed builds
only when their dataset/spec identity matches. Never selects hyperparameters on test data.
"""
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modelrig.buildspec import TrainingMethod, load_spec
from modelrig.factory import Factory, _build_id
from modelrig.integrity import verify_manifest
from modelrig.review import environment, write_json


def main():
    original = load_spec("configs/specs/sms_lora.yaml")
    specs = [replace(original, seed=seed) for seed in (0, 1, 2)]
    specs.append(replace(original, method=TrainingMethod.QLORA, quantization="int4"))
    records = []
    for spec in specs:
        build_id = _build_id(spec)
        path = Path("registry") / build_id
        try:
            if not (path / "eval_report.json").is_file():
                result = Factory().build(spec)
                if not result.success:
                    raise ValueError(result.reason)
            verify_manifest(path)
        except Exception as exc:
            records.append({"build_id": build_id, "success": False, "reason": str(exc)})
            write_json("docs/evidence/neural.json", {"environment": environment(), "runs": records})
            continue
        evaluation = json.loads((path / "eval_report.json").read_text(encoding="utf-8"))
        training = json.loads((path / "training_record.json").read_text(encoding="utf-8"))
        record = {"build_id": build_id, "success": True, "spec": asdict(spec),
                  "training": training, "evaluation": evaluation,
                  "metadata": json.loads((path / "metadata.json").read_text(encoding="utf-8")),
                  "artifact_bytes": sum(p.stat().st_size for p in (path / "deploy").rglob("*")
                                        if p.is_file())}
        records.append(record)
        write_json("docs/evidence/neural.json", {"environment": environment(), "runs": records})
        print(build_id, evaluation["accuracy"], evaluation["macro_f1"], flush=True)
    if any(not record["success"] for record in records):
        raise SystemExit("one or more builds failed; inspect docs/evidence/neural.json")


if __name__ == "__main__":
    main()
