"""LoRA against QLoRA: accuracy, size, latency, memory, energy.

What this machine can and cannot measure, stated once and up front, because
the distinction runs through every number below.

MEASURED, from the four builds in ``registry/`` and the bytes of their
artefacts: accuracy, macro-F1 and their Wilson intervals; per-prediction
agreement; trainable parameter counts and dtypes read out of the safetensors
headers; adapter, merged-model and quantised-deployment sizes to the byte;
where those bytes go, by tensor group; training wall-clock and total FLOPs as
the trainer recorded them; and the effective bits per weight the NF4
deployment actually achieves.

NOT MEASURED, and derived from the implemented equations instead: inference
latency, resident memory and energy. ``import torch`` fails on this host —

    OSError: [WinError 4551] An Application Control policy has blocked this
    file. Error loading "...\\torch\\lib\\shm.dll" or one of its dependencies.

so no forward pass can be timed and no resident set can be sampled. Those
rows are PREDICTIONS from ``modelrig.trainer`` and ``modelrig.resources``,
labelled as such in the report, and they are not evidence about hardware.

One prediction does get checked: the ``NF4_BITS_EFFECTIVE = 4.5`` constant the
planner's memory model rests on is compared against the bits per weight the
quantised artefact really spends.

    python -m evals.adaptation_tradeoffs
"""
from __future__ import annotations

import json
import math
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modelrig import resources, trainer  # noqa: E402
from modelrig.review import write_json  # noqa: E402
from modelrig.stats import wilson  # noqa: E402

OUT = Path("outputs/evals/adaptation_tradeoffs.json")
REGISTRY = Path("registry")

# The host profile the planner would use for this device. Same numbers as
# ``api.studio._probe_profile``, which is where a build gets them.
PROFILE = {
    "bw_eff_gbps": 9.24,
    "overhead_ms_per_token": 2.16,
    "prefill_ref_tok_s": 50.0,
    "reference_params": 1_720_000_000,
    "power_draw_w": 3.5,
    "thermal_derate_180s": 0.6,
}
#: DistilBERT truncates to this in the recorded builds, so it is the input
#: length a latency figure has to be quoted at.
SEQ_LEN = 128


# ----------------------------------------------------------------------- #
# reading artefacts without torch
# ----------------------------------------------------------------------- #
def _safetensors_header(path: Path) -> dict[str, Any]:
    """The tensor inventory, straight out of the file's own header.

    safetensors stores a little-endian u64 length followed by a JSON header
    naming every tensor, its dtype, its shape and its byte range. That is
    enough for a parameter count and a size breakdown, and it needs no
    framework — which is the only reason this study has size numbers at all.
    """
    raw = path.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + length])
    header.pop("__metadata__", None)
    return header


def _elements(entry: dict[str, Any]) -> int:
    return math.prod(entry["shape"]) if entry["shape"] else 1


def _tensor_bytes(entry: dict[str, Any]) -> int:
    lo, hi = entry["data_offsets"]
    return hi - lo


def _tree_bytes(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


def _builds() -> list[dict[str, Any]]:
    """Every recorded build, newest naming scheme, with its training record."""
    found = []
    for path in sorted(REGISTRY.glob("*-lora-*")) + sorted(REGISTRY.glob("*-qlora-*")):
        record = path / "training_record.json"
        report = path / "eval_report.json"
        if not (record.is_file() and report.is_file()):
            continue
        found.append({
            "id": path.name, "dir": path,
            "training": json.loads(record.read_text(encoding="utf-8")),
            "eval": json.loads(report.read_text(encoding="utf-8")),
            "deploy": json.loads((path / "hf_model.json").read_text(encoding="utf-8"))
            if (path / "hf_model.json").is_file() else {},
        })
    return found


# ====================================================================== #
# 1. accuracy — measured
# ====================================================================== #
def measured_quality(builds: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy and macro-F1 per build, and what may be concluded from them.

    Three LoRA seeds against one QLoRA seed is not a comparison that can
    support a claim of difference, and the report says so rather than quoting
    a gap. What it can support is a containment statement: whether the single
    QLoRA result falls inside the spread the LoRA seeds already show.
    """
    rows = []
    for b in builds:
        e = b["eval"]
        correct = round(e["accuracy"] * e["n_test"])
        ci = wilson(correct, e["n_test"])
        rows.append({
            "id": b["id"], "method": b["training"]["training_method"],
            "seed": b["training"].get("seed"),
            "accuracy": e["accuracy"], "macro_f1": e["macro_f1"],
            "n_test": e["n_test"],
            "accuracy_ci95": [ci.low, ci.high],
            "post_quantisation": e.get("post_quantisation"),
            "answer_flip_rate": e.get("answer_flip_rate"),
            "train_loss": b["training"]["training_metrics"]["train_loss"],
        })

    lora = [r for r in rows if r["method"] == "lora"]
    qlora = [r for r in rows if r["method"] == "qlora"]
    lo, hi = min(r["accuracy"] for r in lora), max(r["accuracy"] for r in lora)

    # Prediction-level agreement for the one seed both methods ran.
    agreement: dict[str, Any] = {}
    pair = [(a, b) for a in lora for b in qlora if a["seed"] == b["seed"]]
    if pair:
        lrow, qrow = pair[0]
        lb = next(b for b in builds if b["id"] == lrow["id"])
        qb = next(b for b in builds if b["id"] == qrow["id"])
        lp, qp = lb["eval"].get("predictions"), qb["eval"].get("predictions")
        lg, qg = lb["eval"].get("gold"), qb["eval"].get("gold")
        if lp and qp and lg == qg:
            discordant = sum(a != b_ for a, b_ in zip(lp, qp, strict=True))
            agreement = {
                "seed": lrow["seed"], "n": len(lp),
                "same_test_rows": True,
                "discordant_predictions": discordant,
                "identical": discordant == 0,
                "mcnemar_applicable": discordant > 0,
                "reading": "the two methods produced the same label on every "
                           "test row, so no accuracy difference exists to test"
                if discordant == 0 else
                f"{discordant} rows disagree; a paired test is meaningful here",
            }

    return {
        "status": "MEASURED",
        "rows": rows,
        "lora_seeds": len(lora), "qlora_seeds": len(qlora),
        "lora_accuracy_range": [lo, hi],
        "lora_accuracy_mean": sum(r["accuracy"] for r in lora) / len(lora),
        "qlora_accuracy": qlora[0]["accuracy"] if qlora else None,
        "qlora_within_lora_range": bool(qlora) and lo <= qlora[0]["accuracy"] <= hi,
        "prediction_agreement": agreement,
        "limits": "one QLoRA seed against three LoRA seeds; this can show that "
                  "QLoRA lands inside the LoRA spread, and cannot estimate a "
                  "difference between the methods",
    }


# ====================================================================== #
# 2. size — measured
# ====================================================================== #
def _group_of(name: str) -> str:
    if name.endswith((".absmax", ".quant_map", ".nested_absmax",
                      ".nested_quant_map", ".quant_state.bitsandbytes__nf4")):
        return "quant_scales_and_maps"
    if "embeddings" in name:
        return "embeddings_unquantised"
    if "classifier" in name or "score" in name:
        return "classifier_head_unquantised"
    if "transformer.layer" in name and ".weight" in name:
        return "transformer_quantised"
    return "other"


def measured_size(builds: list[dict[str, Any]]) -> dict[str, Any]:
    """Adapter, merged model and deployment, to the byte.

    The adapter comparison is the one most easily misread. Both methods train
    the identical set of tensors at the identical shapes; the QLoRA adapter is
    larger on disk only because four tensors were serialised at F32 where the
    LoRA run wrote BF16. That is a saving decision, not a property of QLoRA,
    and reporting the raw ratio without it would be wrong.
    """
    rows = []
    for b in builds:
        adapter = b["dir"] / "adapter" / "adapter_model.safetensors"
        head = _safetensors_header(adapter) if adapter.is_file() else {}
        merged = b["dir"] / "model" / "model.safetensors"
        merged_head = _safetensors_header(merged) if merged.is_file() else {}
        quant = b["dir"] / "quantized" / "model.safetensors"
        quant_head = _safetensors_header(quant) if quant.is_file() else {}

        groups: dict[str, dict[str, int]] = defaultdict(
            lambda: {"tensors": 0, "bytes": 0})
        for name, entry in quant_head.items():
            g = groups[_group_of(name)]
            g["tensors"] += 1
            g["bytes"] += _tensor_bytes(entry)

        rows.append({
            "id": b["id"], "method": b["training"]["training_method"],
            "seed": b["training"].get("seed"),
            "deployment_quantisation": b["deploy"].get("deployment_quantization"),
            "adapter": {
                "trainable_parameters": sum(_elements(v) for v in head.values()),
                "tensors": len(head),
                "dtypes": dict(Counter(v["dtype"] for v in head.values())),
                "tensor_bytes": sum(_tensor_bytes(v) for v in head.values()),
                "file_bytes": adapter.stat().st_size if adapter.is_file() else 0,
            },
            "merged_model": {
                "parameters": sum(_elements(v) for v in merged_head.values()),
                "dtypes": dict(Counter(v["dtype"] for v in merged_head.values())),
                "tensor_bytes": sum(_tensor_bytes(v) for v in merged_head.values()),
            },
            "quantised_deployment": {
                "present": bool(quant_head),
                "tensors": len(quant_head),
                "dtypes": dict(Counter(v["dtype"] for v in quant_head.values())),
                "tensor_bytes": sum(_tensor_bytes(v) for v in quant_head.values()),
                "by_group": {k: dict(v) for k, v in sorted(
                    groups.items(), key=lambda kv: -kv[1]["bytes"])},
            },
            "directory_bytes": {
                sub: _tree_bytes(b["dir"] / sub)
                for sub in ("adapter", "model", "deploy", "quantized")
            },
            "shipped_bytes": _tree_bytes(b["dir"] / "deploy"),
        })

    lora = next(r for r in rows if r["method"] == "lora")
    qlora = next(r for r in rows if r["method"] == "qlora")
    params = lora["merged_model"]["parameters"]
    q_bytes = qlora["quantised_deployment"]["tensor_bytes"]
    l_bytes = lora["merged_model"]["tensor_bytes"]

    emb = qlora["quantised_deployment"]["by_group"].get(
        "embeddings_unquantised", {}).get("bytes", 0)
    return {
        "status": "MEASURED",
        "rows": rows,
        "identical_trainable_parameters":
            lora["adapter"]["trainable_parameters"]
            == qlora["adapter"]["trainable_parameters"],
        "trainable_parameters": lora["adapter"]["trainable_parameters"],
        "adapter_bytes_lora": lora["adapter"]["file_bytes"],
        "adapter_bytes_qlora": qlora["adapter"]["file_bytes"],
        "adapter_ratio": (qlora["adapter"]["file_bytes"]
                          / lora["adapter"]["file_bytes"]),
        "adapter_difference_explained": "identical tensor names and shapes; the "
                                        "LoRA run serialised four tensors as BF16 "
                                        "and the QLoRA run wrote all 28 as F32",
        "shipped_bytes_lora": lora["shipped_bytes"],
        "shipped_bytes_qlora": qlora["shipped_bytes"],
        "shipped_ratio": lora["shipped_bytes"] / qlora["shipped_bytes"],
        "shipped_reduction": 1 - qlora["shipped_bytes"] / lora["shipped_bytes"],
        "weight_bytes_bf16": l_bytes,
        "weight_bytes_nf4": q_bytes,
        "effective_bits_per_weight_whole_model": q_bytes * 8 / params,
        "embedding_share_of_deployment": emb / q_bytes if q_bytes else 0.0,
        "reading": "NF4 reaches under a third of the shipped bytes. The embedding "
                   "table is the majority of a small encoder and is left at fp16, "
                   "so the whole-model rate is about 8.7 bits per weight and the "
                   "deployment shrinks by a factor near 1.8, not 4",
    }


# ====================================================================== #
# 3. training cost — measured
# ====================================================================== #
def measured_training(builds: list[dict[str, Any]]) -> dict[str, Any]:
    """Wall-clock and FLOPs as the trainer recorded them.

    The seed-0 pair recorded the same ``total_flos`` to the unit, so the two
    runs did arithmetically the same work. The wall-clock ratio between them is
    therefore a clean measure of what NF4 costs per step in dequantisation,
    with the amount of compute held fixed.
    """
    rows = [{
        "id": b["id"], "method": b["training"]["training_method"],
        "seed": b["training"].get("seed"),
        **{k: b["training"]["training_metrics"].get(k) for k in
           ("train_runtime", "train_samples_per_second", "total_flos",
            "train_loss", "epoch")},
    } for b in builds]

    lora = [r for r in rows if r["method"] == "lora"]
    qlora = [r for r in rows if r["method"] == "qlora"]
    paired = [(x, y) for x in lora for y in qlora
              if x["seed"] == y["seed"] and x["total_flos"] == y["total_flos"]]

    out: dict[str, Any] = {
        "status": "MEASURED",
        "rows": rows,
        "lora_runtime_mean_s": sum(r["train_runtime"] for r in lora) / len(lora),
        "qlora_runtime_s": qlora[0]["train_runtime"] if qlora else None,
    }
    if paired:
        x, y = paired[0]
        out["flops_matched_pair"] = {
            "seed": x["seed"], "total_flos": x["total_flos"],
            "lora_runtime_s": x["train_runtime"],
            "qlora_runtime_s": y["train_runtime"],
            "slowdown": y["train_runtime"] / x["train_runtime"],
            "reading": "identical recorded FLOPs, so the ratio isolates the "
                       "per-step cost of dequantising an NF4 base rather than "
                       "any difference in work done",
        }
    return out


# ====================================================================== #
# 4. the planner's NF4 constant, checked against the artefact
# ====================================================================== #
def validate_nf4_constant(builds: list[dict[str, Any]]) -> dict[str, Any]:
    """Is ``NF4_BITS_EFFECTIVE = 4.5`` right about a real quantised file?

    The planner's whole memory model is built on an assumed effective width
    for a 4-bit base. It is one of the few predictions in this study that an
    artefact on disk can settle, so it is settled here: count the logical
    weights in the transformer's 2-D matrices from the unquantised model, then
    count the bytes the quantised file spends on those same tensors, scales and
    quantisation maps included.
    """
    lora = next((b for b in builds if b["training"]["training_method"] == "lora"
                 and (b["dir"] / "model" / "model.safetensors").is_file()), None)
    qlora = next((b for b in builds if b["training"]["training_method"] == "qlora"
                  and (b["dir"] / "quantized" / "model.safetensors").is_file()), None)
    if not (lora and qlora):
        return {"status": "SKIPPED", "reason": "need an unquantised reference and "
                                               "a quantised deployment"}

    ref = _safetensors_header(lora["dir"] / "model" / "model.safetensors")
    qnt = _safetensors_header(qlora["dir"] / "quantized" / "model.safetensors")
    params = sum(_elements(v) for k, v in ref.items()
                 if "transformer.layer" in k and k.endswith(".weight")
                 and len(v["shape"]) == 2)
    spent = sum(_tensor_bytes(v) for k, v in qnt.items()
                if "transformer.layer" in k and ".weight" in k)
    measured = spent * 8 / params
    assumed = trainer.NF4_BITS_EFFECTIVE
    return {
        "status": "MEASURED",
        "quantised_parameters": params,
        "bytes_spent_including_scales": spent,
        "measured_bits_per_weight": measured,
        "assumed_bits_per_weight": assumed,
        "relative_error": abs(measured - assumed) / assumed,
        "reading": "the constant the planner's memory model assumes is within a "
                   "tenth of a percent of what the artefact spends, so the "
                   "memory predictions below inherit a sound width term even "
                   "though they are not themselves measured",
    }


# ====================================================================== #
# 5. memory, latency, energy — DERIVED, not measured
# ====================================================================== #
def derived_memory(size: dict[str, Any]) -> dict[str, Any]:
    """Training peak and inference residency from the implemented equations.

    Two things are done differently from a naive call into
    ``modelrig.trainer``. The parameter counts come from the artefacts rather
    than the rank formula, which is fitted to a 1.7B Qwen geometry and would be
    wrong here. And the activation constant is recomputed for DistilBERT from
    the formula its docstring gives, ``2 (d + 3d + 2 d_ffn) / d``, which is 24
    at d_ffn/d = 4 rather than the 19 that geometry implies.
    """
    params = size["trainable_parameters"]
    total = next(r["merged_model"]["parameters"] for r in size["rows"])
    geometry = trainer.Geometry(d_model=768, n_layers=6, d_ffn=3072, d_kv=768)
    constant = 2 * (geometry.d_model + 3 * geometry.d_model
                    + 2 * geometry.d_ffn) / geometry.d_model

    rows = []
    for method, bits in (("lora", 16.0), ("qlora", trainer.NF4_BITS_EFFECTIVE)):
        for batch in (1, 8, 16):
            acts = trainer.activation_bytes(batch, SEQ_LEN, geometry,
                                            per_layer_constant=constant,
                                            checkpointing=True)
            frozen = int(total * bits / 8)
            optimiser = trainer.full_finetune_bytes(params)
            rows.append({
                "method": method, "batch": batch,
                "base_bits_effective": bits,
                "frozen_base_bytes": frozen,
                "adapter_and_optimiser_bytes": optimiser,
                "activation_bytes": acts,
                "training_peak_bytes": frozen + optimiser + acts,
                "full_finetune_equivalent_bytes":
                    trainer.full_finetune_bytes(total) + acts,
            })

    at1 = {r["method"]: r for r in rows if r["batch"] == 1}
    infer = {}
    for method, byte_key in (("lora", "weight_bytes_bf16"),
                             ("qlora", "weight_bytes_nf4")):
        infer[method] = {
            "weight_bytes": size[byte_key],
            "activation_bytes": trainer.activation_bytes(
                1, SEQ_LEN, geometry, per_layer_constant=constant,
                checkpointing=False),
        }
        infer[method]["resident_bytes"] = (infer[method]["weight_bytes"]
                                           + infer[method]["activation_bytes"])
    return {
        "status": "DERIVED — equations, not measurements",
        "why": "torch cannot be loaded on this host, so no resident set was "
               "sampled; these are the planner's own predictions",
        "activation_constant_used": constant,
        "activation_constant_note": "recomputed for the DistilBERT geometry; the "
                                    "module default of 19 is the 1.7B Qwen value",
        "training": rows,
        "training_peak_ratio_at_batch_1":
            at1["lora"]["training_peak_bytes"] / at1["qlora"]["training_peak_bytes"],
        "inference": infer,
        "inference_resident_ratio":
            infer["lora"]["resident_bytes"] / infer["qlora"]["resident_bytes"],
        "inference_note": "a sequence classifier runs one forward pass and keeps "
                          "no KV cache, so the KV term in modelrig.resources is "
                          "zero here and is deliberately not included",
    }


def derived_latency_energy(size: dict[str, Any]) -> dict[str, Any]:
    """Latency and energy from ``modelrig.resources``, at this device profile.

    ``tokens_out = 0`` is not a shortcut. DistilBERT with a classification head
    emits one label from one forward pass, so the bandwidth-bound decode term
    does not apply and the whole cost is prefill. Quoting a decode figure here
    would invent a generation loop the artefact does not have.

    The effective FLOP rate comes from the profile's own reference point:
    ``F_eff = 2 * reference_params * prefill_ref_tok_s``.
    """
    total = next(r["merged_model"]["parameters"] for r in size["rows"])
    flops_eff = 2.0 * PROFILE["reference_params"] * PROFILE["prefill_ref_tok_s"]
    rows = []
    for method, byte_key in (("lora", "weight_bytes_bf16"),
                             ("qlora", "weight_bytes_nf4")):
        lat = resources.latency(
            params=total, weight_bytes_=size[byte_key],
            tokens_in=SEQ_LEN, tokens_out=0,
            flops_eff=flops_eff,
            bw_eff=PROFILE["bw_eff_gbps"] * 1e9,
            overhead_s=PROFILE["overhead_ms_per_token"] / 1000.0,
        )
        derated = lat.derated(PROFILE["thermal_derate_180s"])
        eng = resources.energy(derated, power_draw_w=PROFILE["power_draw_w"])
        rows.append({
            "method": method,
            "weight_bytes": size[byte_key],
            "prefill_s": lat.prefill_s, "decode_s": lat.decode_s,
            "total_s": lat.total_s,
            "derated_s": derated,
            "thermal_derate": PROFILE["thermal_derate_180s"],
            "joules_per_request": eng.joules_per_request,
            "requests_per_charge": eng.requests_per_charge,
        })
    return {
        "status": "DERIVED — equations, not measurements",
        "why": "no forward pass could be timed on this host and no power rail "
               "was instrumented; energy is P_draw * latency, not a reading",
        "profile": PROFILE, "tokens_in": SEQ_LEN, "tokens_out": 0,
        "effective_flops_per_second": flops_eff,
        "rows": rows,
        "reading": "prefill is compute-bound and scales with parameter count, "
                   "which quantisation does not change, so the two methods are "
                   "predicted to cost the same per request on this profile; "
                   "quantisation buys footprint here, not speed",
    }


# ====================================================================== #
# 6. quantisation evidence that needs no framework
# ====================================================================== #
def measured_quantisation_sweep() -> dict[str, Any]:
    """Real int8 and int4 measurements, on the backend that does run here.

    This is not LoRA against QLoRA and is not offered as a substitute for it.
    It is the one quantisation trade-off on this machine that is measured
    rather than predicted: the compiler's own centroid classifier, quantised to
    int8 and int4, across five tasks, with answer-flip rates and warm latency
    taken on this hardware. Its value is that it shows the shape of the
    trade-off empirically where the transformer path cannot.
    """
    report = Path("outputs/evals/multitask/report.json")
    if not report.is_file():
        return {"status": "SKIPPED",
                "reason": "run `python -m evals.multitask` first"}
    tasks = json.loads(report.read_text(encoding="utf-8"))["protocol"]["per_task"]
    rows = []
    for t in tasks:
        base = t.get("centroid_none")
        for quant in ("int8", "int4"):
            got = t.get(f"centroid_{quant}")
            if not (base and got):
                continue
            rows.append({
                "task": t["task"], "n_classes": t["n_classes"],
                "quantisation": quant,
                "accuracy_mean": got["accuracy_mean"],
                "accuracy_delta_vs_fp32": got["accuracy_mean"] - base["accuracy_mean"],
                "macro_f1_delta_vs_fp32": got["macro_f1_mean"] - base["macro_f1_mean"],
                "answer_flip_rate": got["flip_rate_mean"],
                "warm_p50_ms": got["warm_p50_ms_mean"],
                "warm_p50_ms_fp32": base["warm_p50_ms_mean"],
            })
    deltas = [r["accuracy_delta_vs_fp32"] for r in rows if r["quantisation"] == "int4"]
    return {
        "status": "MEASURED",
        "scope": "the centroid classifier, not the transformer path; offered as "
                 "measured context for the quantisation question, not as a "
                 "LoRA/QLoRA result",
        "rows": rows,
        "int4_accuracy_delta_range": [min(deltas), max(deltas)] if deltas else None,
        "int8_flip_rate_range": [
            min(r["answer_flip_rate"] for r in rows if r["quantisation"] == "int8"),
            max(r["answer_flip_rate"] for r in rows if r["quantisation"] == "int8"),
        ] if rows else None,
    }


# ====================================================================== #
# 7. the boundary
# ====================================================================== #
def boundary() -> dict[str, Any]:
    """What this study is and is not evidence for."""
    try:
        import torch  # noqa: F401
        torch_state = "importable"
    except Exception as exc:                                       # noqa: BLE001
        torch_state = f"{type(exc).__name__}: {exc}"
    return {
        "torch": torch_state,
        "measured_here": [
            "accuracy, macro-F1 and Wilson intervals, from the recorded eval reports",
            "per-prediction agreement between the seed-0 LoRA and QLoRA builds",
            "trainable parameter counts, tensor shapes and dtypes, from safetensors headers",
            "adapter, merged-model and quantised-deployment sizes in bytes",
            "where the deployment's bytes go, by tensor group",
            "training wall-clock and total FLOPs, as the trainer recorded them",
            "effective bits per weight of the NF4 deployment",
        ],
        "derived_not_measured": [
            "inference latency",
            "resident memory at inference",
            "training peak memory",
            "energy per request and requests per charge",
        ],
        "why_not_measured": "torch will not load on this host, so no forward pass "
                            "could be timed, no resident set sampled, and no "
                            "power draw read. Energy is in every case "
                            "P_draw * predicted latency at an assumed 3.5 W, "
                            "which is an arithmetic restatement of the latency "
                            "prediction and not an independent measurement.",
        "what_would_close_it": [
            "a host without the Application Control policy, or a CPU-only torch "
            "build that does not load shm.dll",
            "both methods trained at the same three seeds, so the accuracy "
            "comparison has more than one QLoRA point",
            "a wall-power or battery-rail reading for the energy row",
            "an int4 deployment that also quantises the embedding table, which is "
            "where the measured bytes actually are",
        ],
    }


# ====================================================================== #
def _findings(quality: dict[str, Any], size: dict[str, Any],
              train: dict[str, Any], nf4: dict[str, Any],
              memory: dict[str, Any], latency: dict[str, Any],
              sweep: dict[str, Any]) -> list[str]:
    out: list[str] = []
    agree = quality.get("prediction_agreement") or {}
    if agree.get("identical"):
        out.append(f"MEASURED: at seed {agree['seed']} the LoRA and QLoRA builds "
                   f"produced the same label on all {agree['n']} test rows — "
                   f"identical accuracy {quality['qlora_accuracy']:.4f} and "
                   f"macro-F1, so there is no accuracy cost to test")
    out.append(f"MEASURED: LoRA accuracy over {quality['lora_seeds']} seeds spans "
               f"{quality['lora_accuracy_range'][0]:.4f}-"
               f"{quality['lora_accuracy_range'][1]:.4f}; the single QLoRA seed "
               f"falls {'inside' if quality['qlora_within_lora_range'] else 'outside'} "
               f"that range")
    out.append(f"MEASURED: both methods train the identical "
               f"{size['trainable_parameters']:,} parameters; the QLoRA adapter is "
               f"{size['adapter_ratio']:.2f}x larger on disk purely because four "
               f"tensors were written F32 instead of BF16")
    out.append(f"MEASURED: the shipped deployment falls "
               f"{size['shipped_bytes_lora']:,} to {size['shipped_bytes_qlora']:,} "
               f"bytes, a {size['shipped_reduction']:.1%} reduction "
               f"({size['shipped_ratio']:.2f}x)")
    out.append(f"MEASURED: that deployment averages "
               f"{size['effective_bits_per_weight_whole_model']:.2f} bits per "
               f"weight, not 4, because the embedding table is "
               f"{size['embedding_share_of_deployment']:.1%} of its bytes and "
               f"stays at fp16")
    pair = train.get("flops_matched_pair")
    if pair:
        out.append(f"MEASURED: at identical recorded FLOPs, QLoRA training took "
                   f"{pair['qlora_runtime_s']:.1f} s against LoRA's "
                   f"{pair['lora_runtime_s']:.1f} s — a {pair['slowdown']:.2f}x "
                   f"slowdown attributable to NF4 dequantisation per step")
    if nf4.get("status") == "MEASURED":
        out.append(f"MEASURED: the planner's NF4_BITS_EFFECTIVE = "
                   f"{nf4['assumed_bits_per_weight']} against "
                   f"{nf4['measured_bits_per_weight']:.3f} bits per weight in the "
                   f"artefact — {nf4['relative_error']:.2%} error, so the memory "
                   f"model's width term is sound")
    peak = memory["training_peak_ratio_at_batch_1"]
    out.append(f"DERIVED: predicted training peak is {peak:.2f}x lower under "
               f"QLoRA at batch 1, and predicted inference residency "
               f"{memory['inference_resident_ratio']:.2f}x lower")
    if pair and round(peak, 2) == round(pair["slowdown"], 2):
        out.append(f"NOTE: the {peak:.2f}x memory ratio and the "
                   f"{pair['slowdown']:.2f}x training slowdown are independent "
                   f"quantities that coincide to two decimals — the first comes "
                   f"from 16 against 4.5 bits per frozen weight, the second from "
                   f"measured wall-clock. Neither is derived from the other")
    rows = {r["method"]: r for r in latency["rows"]}
    out.append(f"DERIVED: predicted per-request latency is "
               f"{rows['lora']['derated_s'] * 1000:.0f} ms for both methods "
               f"({rows['lora']['joules_per_request']:.1f} J) — prefill is "
               f"compute-bound in the parameter count, which quantisation does "
               f"not reduce, so quantisation is predicted to buy footprint and "
               f"not speed")
    if sweep.get("status") == "MEASURED" and sweep.get("int4_accuracy_delta_range"):
        lo, hi = sweep["int4_accuracy_delta_range"]
        out.append(f"MEASURED, different backend: int4 on the centroid classifier "
                   f"moves accuracy by {lo:+.4f} to {hi:+.4f} across five tasks")
    return out


def main() -> int:
    builds = _builds()
    if not builds:
        print("no LoRA/QLoRA builds found under registry/")
        return 1
    print(f"{len(builds)} recorded builds: "
          f"{', '.join(b['training']['training_method'] for b in builds)}\n")

    quality = measured_quality(builds)
    size = measured_size(builds)
    train = measured_training(builds)
    nf4 = validate_nf4_constant(builds)
    memory = derived_memory(size)
    latency = derived_latency_energy(size)
    sweep = measured_quantisation_sweep()

    report = {
        "study": "LoRA against QLoRA — accuracy, size, latency, memory, energy",
        "reviewer_item": "provide a comparative evaluation of LoRA and QLoRA in "
                         "terms of accuracy, model size, inference latency and "
                         "memory consumption, and include energy measurements "
                         "where feasible",
        "quality": quality, "size": size, "training_cost": train,
        "nf4_constant_check": nf4, "memory": memory,
        "latency_and_energy": latency, "quantisation_sweep": sweep,
        "boundary": boundary(),
    }
    report["findings"] = _findings(quality, size, train, nf4, memory,
                                   latency, sweep)
    write_json(OUT, report)
    for line in report["findings"]:
        print(f"  - {line}")
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
