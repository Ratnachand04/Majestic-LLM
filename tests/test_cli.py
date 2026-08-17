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


def test_validate_command(capsys):
    """Model compatibility + architecture conformance must both be clean."""
    assert main(["validate"]) == 0
    out = capsys.readouterr().out
    assert "RESULT     : conformant" in out
    assert "errors     : 0" in out


def test_validate_shows_warnings_on_request(capsys):
    assert main(["validate", "--warnings"]) == 0
    assert "logit_kd_reachable" in capsys.readouterr().out


def _apex_spec(tmp_path: Path, **over) -> Path:
    body = {
        "task_primitive": "extract", "device_target": "android_tablet_4gb",
        "offline_required": True, "seed_data_count": 150,
        "data_rights": "customer_owned", "quality_gate": 0.93,
        "latency_budget_ms": 2000,
        "io_schema": {"tokens_in": 500, "tokens_out": 60,
                      "accept_unmeasured_latency": True},
    }
    body.update(over)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_plan_command_refuses_with_a_witness(tmp_path: Path, capsys):
    """§8: eight seconds of computation prevents a 45-minute useless build."""
    assert main(["plan", "--spec", str(_apex_spec(tmp_path))]) == 1
    out = capsys.readouterr().out
    assert "REFUSED" in out
    assert "P_lat" in out and "prefill" in out
    assert "Remedies" in out
    assert "unmeasured" in out
    assert "before any GPU was allocated" in out


def test_plan_command_admits_with_a_realistic_budget(tmp_path: Path, capsys):
    spec = _apex_spec(tmp_path, latency_budget_ms=20_000)
    assert main(["plan", "--spec", str(spec)]) == 0
    out = capsys.readouterr().out
    assert "ADMITTED" in out
    assert "largest base that fits" in out
    assert "theta*" in out


def test_plan_explain_shows_the_predicate_order(tmp_path: Path, capsys):
    main(["plan", "--spec", str(_apex_spec(tmp_path)), "--explain"])
    out = capsys.readouterr().out
    assert "P_tok ->" in out                 # cheapest-and-most-discriminating first
    assert "candidate plans" in out
    assert "early exit" in out


def test_plan_tier_changes_the_threshold(tmp_path: Path, capsys):
    """A regulated tier refuses far more than an experimental one."""
    spec = _apex_spec(tmp_path, latency_budget_ms=20_000, quality_gate=0.5)
    main(["plan", "--spec", str(spec), "--tier", "experimental"])
    experimental = capsys.readouterr().out
    main(["plan", "--spec", str(spec), "--tier", "regulated"])
    regulated = capsys.readouterr().out
    assert experimental != regulated


def test_budget_command_prints_the_b09_table(capsys):
    assert main(["budget"]) == 0
    out = capsys.readouterr().out
    assert "Base model" in out and "KV cache" in out
    assert "Total committed" in out
    assert "measured=False" in out          # GAP-10 is never hidden


def test_budget_command_fails_when_over(capsys):
    assert main(["budget", "--base-params-b", "8", "--ram-gb", "4"]) == 1
    assert "PROBLEM" in capsys.readouterr().out


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
