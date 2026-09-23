"""Verify saved public-data models and local API with Hub access disabled."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def main():
    from fastapi.testclient import TestClient

    from api.app import create_app
    from modelrig.integrity import verify_manifest
    from modelrig.review import environment, predict_artifact, write_json

    evidence = json.loads((ROOT / "docs/evidence/neural.json").read_text(encoding="utf-8"))
    messages = ["Please call me after the meeting", "Claim your FREE cash prize NOW!"]
    results = []
    client = TestClient(create_app())
    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 200
    listed = {r["cartridge_id"] for r in client.get("/api/models").json()["models"]}
    for run in evidence["runs"]:
        assert run["success"]
        key = run["build_id"]
        path = ROOT / "registry" / key
        verify_manifest(path)
        assert key in listed
        predictions = predict_artifact(path, messages)
        assert len(predictions) == len(messages) and set(predictions) <= {"ham", "spam"}
        response = client.post("/api/predict", json={"cartridge_id": key, "texts": messages})
        assert response.status_code == 200
        assert not response.json()["error"], response.json()
        assert [r["label"] for r in response.json()["predictions"]] == predictions
        results.append({"build_id": key, "manifest_verified": True,
                        "offline_reload": True, "api_matches_direct": True,
                        "smoke_predictions": predictions})
    portable = ROOT / "outputs/review_release/sms_specialist"
    if portable.exists():
        verify_manifest(portable)
        assert predict_artifact(portable, messages) == results[0]["smoke_predictions"]
    write_json(ROOT / "docs/evidence/offline_validation.json", {
        "environment": environment(), "hub_offline": True,
        "scope": "local model reload and API parity, not an accuracy benchmark",
        "runs": results, "portable_verified": portable.exists(),
    })
    print(f"Verified {len(results)} models and API inference with Hub access disabled.")


if __name__ == "__main__":
    main()
