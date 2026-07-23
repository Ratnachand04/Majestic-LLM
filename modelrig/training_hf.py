"""Heavy training/inference path: PEFT LoRA fine-tuning over a small HF model.

Imported lazily by the planes only when ``spec.method`` is a heavy method. This
requires the ``ml`` extras (torch/transformers/peft) and is exercised by
integration tests, never by the default offline suite. Kept intentionally small
so it can run on modest hardware with a tiny base model.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from majestic.logging_utils import get_logger
from modelrig.buildspec import BuildSpec, TrainingMethod

logger = get_logger(__name__)


def train_hf(spec: BuildSpec, ctx: dict[str, Any]) -> dict[str, Any]:
    """LoRA fine-tune a small sequence-classification model on the split data."""
    import tempfile

    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from transformers.data.data_collator import DataCollatorWithPadding

    labels: list[str] = ctx["labels"]
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(spec.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.base_model, num_labels=len(labels), id2label=id2label, label2id=label2id
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    quant = spec.method == TrainingMethod.QLORA
    lora = LoraConfig(task_type="SEQ_CLS", r=8, lora_alpha=16, lora_dropout=0.05)
    model = get_peft_model(model, lora)

    class _DS(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            text, label = self.rows[i]
            enc = tokenizer(text, truncation=True, max_length=64)
            enc["labels"] = label2id[label]
            return enc

    def _metrics(pred):
        preds = np.argmax(pred.predictions, axis=1)
        return {"accuracy": float((preds == pred.label_ids).mean())}

    with tempfile.TemporaryDirectory() as tmp:
        args = TrainingArguments(
            output_dir=tmp,
            num_train_epochs=float(spec.extras.get("epochs", 3)),
            per_device_train_batch_size=4,
            learning_rate=2e-4,
            logging_steps=5,
            report_to=[],
            seed=spec.seed,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=_DS(ctx["train"]),
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=_metrics,
        )
        trainer.train()
        adapter_dir = Path(tempfile.mkdtemp(prefix="mjc_adapter_"))
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

    logger.info("train_hf: saved LoRA adapter to %s (qlora=%s)", adapter_dir, quant)
    return {"kind": "hf_seqcls", "adapter_dir": str(adapter_dir),
            "base_model": spec.base_model, "labels": labels}


def hf_predict(model: dict[str, Any], texts: list[str]) -> list[str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    labels = model["labels"]
    tokenizer = AutoTokenizer.from_pretrained(model["adapter_dir"])
    base = AutoModelForSequenceClassification.from_pretrained(
        model["base_model"], num_labels=len(labels)
    )
    net = PeftModel.from_pretrained(base, model["adapter_dir"]).eval()
    preds = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            logits = net(**enc).logits
            preds.append(labels[int(logits.argmax(dim=-1))])
    return preds


def compress_hf(spec: BuildSpec, model: dict[str, Any]) -> dict[str, Any]:
    """Passthrough for now; true GGUF/GPTQ export is an optional exporter TODO."""
    logger.info("compress_hf: quantization %s deferred to export", spec.quantization)
    return {"model": model, "compression": {"method": spec.quantization, "ratio": None,
                                             "note": "GGUF/GPTQ export not implemented"}}


def export_hf(spec: BuildSpec, model: dict[str, Any], out_dir: Path) -> str:
    """Copy the trained adapter into the build directory. Returns the runtime tag."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = model.get("adapter_dir")
    if adapter_dir:
        shutil.copytree(adapter_dir, out_dir / "adapter", dirs_exist_ok=True)
    # GGUF/ONNX conversion would happen here (optional exporters).
    return "hf"
