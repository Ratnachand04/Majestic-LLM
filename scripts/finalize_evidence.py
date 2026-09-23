"""Collect actual local verification and source identities for the review release."""
import importlib.metadata
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    from modelrig.integrity import digest_file
    from modelrig.review import write_json

    suites = []
    for filename in ("release-tests.xml", "release-integration.xml"):
        path = ROOT / "outputs" / filename
        if path.exists():
            root = ET.parse(path).getroot()
            suites.append({"file": filename, "suites": [s.attrib for s in root]})
    sources = {}
    for directory in ("modelrig", "majestic", "api", "cli", "configs", "frontend",
                      "scripts", "tests", "paper"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and path.suffix in (".py", ".yaml", ".html", ".tex"):
                sources[path.relative_to(ROOT).as_posix()] = digest_file(path)
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    write_json(ROOT / "docs/evidence/software_validation.json", {
        "test_suites": suites, "packages": packages, "source_sha256": sources,
        "scope": "local software verification; no external certification or field evaluation",
    })
    shutil.copy2(ROOT / "data/sms.provenance.json", ROOT / "docs/evidence/data_provenance.json")
    write_json(ROOT / "docs/evidence/development_observations.json", {
        "qlora_initial_failure": {
            "stage": "pretrained sequence-classifier load, before training",
            "error": "AttributeError: weight is not an nn.Module",
            "correction": "exclude uninitialized classifier heads from bitsandbytes conversion",
            "successful_retry": "sms_spam-qlora-a5512e8be785c14da0c986e0",
            "limits": "one completed QLoRA replicate; not a statistical comparison",
        },
    })
    print(json.dumps({"test_reports": len(suites), "source_files": len(sources)}))


if __name__ == "__main__":
    main()
