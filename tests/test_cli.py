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


def test_primitives_command(capsys):
    assert main(["primitives"]) == 0
    out = capsys.readouterr().out
    assert "extract" in out and "classify" in out
    assert "GAP-06" in out


def test_forge_command(tmp_path: Path, capsys):
    out_path = tmp_path / "spec.json"
    code = main([
        "forge", "Classify support tickets by sentiment on an android phone",
        "--offline", "--seed-count", "150", "--out", str(out_path),
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "primitive : classify" in out
    assert out_path.exists()


def test_forge_refuses_to_guess(capsys):
    assert main(["forge", "please do the needful"]) == 2
    assert "cannot emit a spec yet" in capsys.readouterr().out


def test_compile_command(tmp_path: Path, capsys):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "task_primitive": "classify",
        "device_target": "android_midrange",
        "seed_data_count": 120,
        "data_rights": "customer_owned",
        "quality_gate": 0.8,
    }), encoding="utf-8")
    code = main(["compile", "--spec", str(spec), "--registry", str(tmp_path / "reg")])
    out = capsys.readouterr().out
    assert "gate1_spec_admissibility" in out
    assert code in (0, 1)          # admitted, or an honest refusal with reasons


def test_compile_refuses_below_seed_floor(tmp_path: Path, capsys):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "task_primitive": "classify", "device_target": "android_midrange",
        "seed_data_count": 3, "data_rights": "customer_owned",
    }), encoding="utf-8")
    assert main(["compile", "--spec", str(spec), "--registry", str(tmp_path / "r")]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "seed data" in out


def test_verify_graph_command(tmp_path: Path, capsys):
    graph = tmp_path / "graph.json"
    graph.write_text(json.dumps({
        "name": "flow",
        "nodes": [
            {"name": "extract", "kind": "cartridge", "ram_mb": 30},
            {"name": "notify", "kind": "tool", "requires_network": True},
        ],
        "edges": [["extract", "notify"]],
    }), encoding="utf-8")
    assert main(["verify-graph", str(graph), "--offline"]) == 1
    out = capsys.readouterr().out
    assert "offline_closed : False" in out
    assert "remedy" in out
