"""PEFT classification training, full-precision merge, quantization and offline export."""
from __future__ import annotations

import gc
import json
import shutil
from pathlib import Path
from typing import Any

from modelrig.buildspec import BuildSpec, TrainingMethod


def validate_hf_request(spec: BuildSpec) -> None:
    if spec.method not in (TrainingMethod.LORA, TrainingMethod.QLORA):
        raise ValueError("neural backend supports lora/qlora; distillation needs teacher evidence")
    if spec.runtime != "hf":
        raise ValueError("neural backend requires runtime=hf; GGUF/ONNX export is unavailable")
    for key in ("epochs", "learning_rate", "batch_size", "rank", "max_length"):
        if key in spec.extras and float(spec.extras[key]) <= 0:
            raise ValueError(f"{key} must be positive")


def train_hf(spec: BuildSpec, ctx: dict[str, Any]) -> dict[str, Any]:
    validate_hf_request(spec)
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    quant = spec.method is TrainingMethod.QLORA
    if quant and not torch.cuda.is_available():
        raise ValueError("QLoRA requires CUDA in this backend; install CUDA PyTorch or use LoRA")
    set_seed(spec.seed)
    source = Path(spec.base_model)
    revision = spec.extras.get("revision", "main")
    if source.is_dir():
        resolved = str(source.resolve())
        revision = "local"
    else:
        revision = HfApi().model_info(spec.base_model, revision=revision).sha
        resolved = snapshot_download(
            spec.base_model, revision=revision,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        )
    labels = ctx["labels"]
    if len(labels) < 2:
        raise ValueError("classification needs at least two labels")
    label2id = {label: i for i, label in enumerate(labels)}
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(resolved, local_files_only=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("base tokenizer has neither padding nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    max_length = int(spec.extras.get("max_length", 256))
    kwargs = dict(
        num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=label2id,
        local_files_only=True, torch_dtype=dtype,
    )
    if quant:
        kwargs.update(quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
            llm_int8_skip_modules=["classifier", "pre_classifier", "score"],
        ), device_map={"": 0})
    net = AutoModelForSequenceClassification.from_pretrained(resolved, **kwargs)
    net.config.pad_token_id = tokenizer.pad_token_id
    net.config.use_cache = False
    checkpointing = bool(spec.extras.get("gradient_checkpointing", quant))
    if quant:
        net = prepare_model_for_kbit_training(net, use_gradient_checkpointing=checkpointing)
    net = get_peft_model(net, LoraConfig(
        task_type="SEQ_CLS", r=int(spec.extras.get("rank", 8)),
        lora_alpha=int(spec.extras.get("alpha", 16)), lora_dropout=0.05,
        target_modules=spec.extras.get("target_modules", "all-linear"),
    ))

    class Rows(torch.utils.data.Dataset):
        def __len__(self):
            return len(ctx["train"])

        def __getitem__(self, index):
            text, label = ctx["train"][index]
            encoded = tokenizer(text, truncation=True, max_length=max_length)
            encoded["labels"] = label2id[label]
            return encoded

    out = Path(ctx["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=float(spec.extras.get("epochs", 3)),
        per_device_train_batch_size=int(spec.extras.get("batch_size", 4)),
        gradient_accumulation_steps=int(spec.extras.get("gradient_accumulation", 1)),
        learning_rate=float(spec.extras.get("learning_rate", 2e-4)),
        max_steps=int(spec.extras.get("max_steps", -1)),
        logging_steps=20, report_to=[], save_strategy="no", seed=spec.seed,
        data_seed=spec.seed, bf16=torch.cuda.is_available(),
        gradient_checkpointing=checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if checkpointing else None,
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=net, args=args, train_dataset=Rows(),
                      data_collator=DataCollatorWithPadding(tokenizer))
    trained = trainer.train()
    adapter = out / "adapter"
    net.save_pretrained(adapter, safe_serialization=True)
    if quant:
        # Reload the original unquantized base. Never merge into the NF4 training base.
        del trainer, net
        gc.collect()
        torch.cuda.empty_cache()
        kwargs.pop("quantization_config")
        kwargs.pop("device_map")
        base = AutoModelForSequenceClassification.from_pretrained(resolved, **kwargs)
        base.config.pad_token_id = tokenizer.pad_token_id
        net = PeftModel.from_pretrained(base, adapter, local_files_only=True)
    merged = net.merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    merged.save_pretrained(out / "model", safe_serialization=True)
    tokenizer.save_pretrained(out / "model")
    record = {
        "kind": "hf_seqcls", "model_dir": str(out / "model"),
        "adapter_dir": str(adapter), "base_model": spec.base_model,
        "base_revision": revision, "labels": labels, "max_length": max_length,
        "training_method": spec.method.value, "training_metrics": trained.metrics,
        "deployment_quantization": "none", "seed": spec.seed,
    }
    (out / "training_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    del merged, net
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def hf_predict(model: dict[str, Any], texts: list[str]) -> list[str]:
    if not texts:
        return []
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = model["model_dir"]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    kwargs: dict[str, Any] = {"local_files_only": True}
    if model.get("deployment_quantization", "none") != "none":
        kwargs["device_map"] = {"": 0}
    net = AutoModelForSequenceClassification.from_pretrained(path, **kwargs).eval()
    if not getattr(net, "is_quantized", False):
        net.to("cuda" if torch.cuda.is_available() else "cpu")
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(texts), 16):
            encoded = tokenizer(texts[start:start + 16], padding=True, return_tensors="pt",
                                truncation=True, max_length=model.get("max_length", 256))
            logits = net(**encoded.to(net.device)).logits
            predictions.extend(model["labels"][i] for i in logits.argmax(-1).tolist())
    del net
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions


def compress_hf(spec: BuildSpec, model: dict[str, Any]) -> dict[str, Any]:
    """Quantize the merged checkpoint using bitsandbytes, then save the real result."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise ValueError("HF deployment quantization requires CUDA in this backend")
    source = Path(model["model_dir"])
    target = source.parent / "quantized"
    skip = ["classifier", "pre_classifier", "score"]
    config = (BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
              if spec.quantization == "int8" else
              BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 llm_int8_skip_modules=skip,
                                 bnb_4bit_compute_dtype=torch.bfloat16))
    net = AutoModelForSequenceClassification.from_pretrained(
        source, local_files_only=True, quantization_config=config, device_map={"": 0},
    )
    net.save_pretrained(target, safe_serialization=True)
    AutoTokenizer.from_pretrained(source, local_files_only=True).save_pretrained(target)
    original = sum(p.stat().st_size for p in source.glob("*.safetensors"))
    compressed = sum(p.stat().st_size for p in target.glob("*.safetensors"))
    result = dict(model, model_dir=str(target), deployment_quantization=spec.quantization)
    del net
    gc.collect()
    torch.cuda.empty_cache()
    return {"model": result, "compression": {
        "method": f"bitsandbytes_{spec.quantization}", "orig_bytes": original,
        "comp_bytes": compressed, "ratio": original / max(compressed, 1),
    }}


def export_hf(spec: BuildSpec, model: dict[str, Any], out_dir: Path) -> str:
    out_dir = Path(out_dir)
    target = out_dir / "deploy"
    shutil.copytree(Path(model["model_dir"]), target, dirs_exist_ok=True)
    record = dict(model, model_dir="deploy", adapter_dir="adapter")
    (out_dir / "hf_model.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return "hf"
