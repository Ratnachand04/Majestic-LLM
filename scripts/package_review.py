"""Create a private local review archive from an explicit allowlist."""
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def main():
    from modelrig.integrity import write_manifest

    reports = json.loads((ROOT / "docs/evidence/neural.json").read_text(encoding="utf-8"))
    # Fixed seed zero float deployment, selected by protocol rather than best test score.
    run = next(r for r in reports["runs"] if r["success"] and
               r["spec"]["seed"] == 0 and r["spec"]["method"] == "lora")
    source = ROOT / "registry" / run["build_id"]
    portable = ROOT / "outputs" / "review_release" / "sms_specialist"
    portable.mkdir(parents=True, exist_ok=True)
    for directory in ("deploy", "adapter"):
        shutil.copytree(source / directory, portable / directory, dirs_exist_ok=True)
    for name in ("hf_model.json", "metadata.json", "eval_report.json", "training_record.json"):
        shutil.copy2(source / name, portable / name)
    (portable / "MODEL_CARD.md").write_text(
        "# Majestic SMS specialist\n\n"
        "DistilBERT base (Apache-2.0), adapted with PEFT LoRA on the public UCI SMS Spam "
        "Collection (CC BY 4.0). Authors of the dataset: T. Almeida and J. Hidalgo, 2011. "
        "Dataset: https://doi.org/10.24432/C5CC84. Base: "
        "https://huggingface.co/distilbert/distilbert-base-uncased.\n\n"
        "Intended use: local English SMS spam classification research and demonstration. "
        "Labels: ham/spam. This is not a general-purpose assistant, medical model, or "
        "production spam service. Historical data and near-duplicates can bias estimates. "
        "Test results do not establish future-domain accuracy, safety, privacy, or mobile "
        "performance. It processes at most 128 tokens; longer inputs are truncated.\n\n"
        f"Evaluation: {run['evaluation']['accuracy']:.6f} accuracy, "
        f"{run['evaluation']['macro_f1']:.6f} macro-F1, n=1031, seed=0. "
        "Read eval_report.json for the interval and per-example label evidence.\n\n"
        "Run from the release root: python -m modelrig.review predict "
        "--path sms_specialist --text 'Please call me after the meeting'. "
        "Install the documented Python ML dependencies first. Model loading is local.\n",
        encoding="utf-8",
    )
    license_path = ROOT / "docs" / "APACHE-2.0.txt"
    if license_path.is_file():
        shutil.copy2(license_path, portable / "BASE_LICENSE.txt")
    write_manifest(portable)
    quantized_run = next(r for r in reports["runs"] if r["success"] and
                         r["spec"]["seed"] == 0 and r["spec"]["method"] == "qlora")
    quantized_source = ROOT / "registry" / quantized_run["build_id"]
    quantized_portable = portable.parent / "sms_specialist_nf4"
    quantized_portable.mkdir(parents=True, exist_ok=True)
    for directory in ("deploy", "adapter"):
        shutil.copytree(quantized_source / directory, quantized_portable / directory,
                        dirs_exist_ok=True)
    for name in ("hf_model.json", "metadata.json", "eval_report.json", "training_record.json"):
        shutil.copy2(quantized_source / name, quantized_portable / name)
    quantized_card = (portable / "MODEL_CARD.md").read_text(encoding="utf-8").replace(
        "adapted with PEFT LoRA", "adapted with PEFT QLoRA and exported with NF4 weights"
    ).replace("--path sms_specialist ", "--path sms_specialist_nf4 ")
    quantized_card += "\nThis NF4 runtime requires CUDA and the `qlora` dependency extra.\n"
    (quantized_portable / "MODEL_CARD.md").write_text(quantized_card, encoding="utf-8")
    shutil.copy2(license_path, quantized_portable / "BASE_LICENSE.txt")
    write_manifest(quantized_portable)
    destination = ROOT / "dist" / "Majestic_Private_Review.zip"
    destination.parent.mkdir(exist_ok=True)
    files = [ROOT / name for name in ("README.md", "LICENSE", "pyproject.toml", "run.py")]
    for directory in ("modelrig", "majestic", "api", "cli", "configs", "frontend",
                      "scripts", "tests", "demo"):
        files.extend(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".yaml", ".html", ".json")
                     and "__pycache__" not in p.parts)
    files.extend(ROOT / "docs" / name for name in
                 ("REVIEW_STATUS.md", "REVIEW_GUIDE.md", "INVESTOR_REVIEW.md", "APACHE-2.0.txt",
                  "LEGACY_DESIGN.md"))
    files.extend((ROOT / "docs/evidence").glob("*.json"))
    files.extend((ROOT / "docs/evidence/review_model").glob("*"))
    files.extend(p for p in (ROOT / "paper").glob("*") if p.suffix in (".tex", ".pdf"))
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.is_file():
                archive.write(path, path.relative_to(ROOT).as_posix())
        for artifact in (portable, quantized_portable):
            for path in artifact.rglob("*"):
                if path.is_file():
                    archive.write(path, artifact.name + "/" + path.relative_to(artifact).as_posix())
    print(destination)
    print(f"{destination.stat().st_size / 1_000_000:.1f} MB; patent/private inputs excluded")


if __name__ == "__main__":
    main()
