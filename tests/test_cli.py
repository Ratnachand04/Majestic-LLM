"""Tests for the CLI entry point."""
from __future__ import annotations

import json
from pathlib import Path

from cli.main import main


def test_info(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("MAJESTIC_MODEL_REGISTRY_PATH", str(tmp_path / "reg"))
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "Majestic LLM" in out
    assert "devices:" in out


def test_validate_spec_ok(tmp_path: Path, capsys):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "task": "sentiment", "base_model": "centroid", "method": "centroid",
    }), encoding="utf-8")
    assert main(["validate-spec", str(spec)]) == 0
    assert "OK" in capsys.readouterr().out


def test_validate_spec_invalid(tmp_path: Path, capsys):
    spec = tmp_path / "bad.json"
    spec.write_text(json.dumps({"task": "", "base_model": "", "quantization": "int7"}),
                    encoding="utf-8")
    assert main(["validate-spec", str(spec)]) == 1
    assert "INVALID" in capsys.readouterr().out


def test_build_via_capability(tmp_path: Path, capsys):
    code = main(["build", "--capability", "sentiment", "--device", "android_midrange",
                 "--registry", str(tmp_path / "reg")])
    assert code == 0
    out = capsys.readouterr().out
    assert "success  : True" in out
    assert "build_id" in out


def test_feasibility_command(capsys):
    assert main(["feasibility", "--capability", "sentiment", "--device", "laptop_gpu"]) == 0
    assert "feasible : True" in capsys.readouterr().out
