"""Real neural save/reload smoke test with local random weights; no Hub dependency."""
import json

import pytest


@pytest.mark.integration
def test_local_lora_training_export_and_reload(tmp_path):
    pytest.importorskip("torch")
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    from modelrig.buildspec import BuildSpec, TrainingMethod
    from modelrig.factory import Factory
    from modelrig.review import predict_artifact

    base = tmp_path / "base"
    base.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "good", "bad", "case"]
    (base / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    BertTokenizerFast(vocab_file=str(base / "vocab.txt")).save_pretrained(base)
    BertForSequenceClassification(BertConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, num_labels=2,
    )).save_pretrained(base)
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps({"text": f"{word} case {i}", "label": label})
                              for i in range(8) for word, label in
                              (("good", "positive"), ("bad", "negative"))), encoding="utf-8")
    spec = BuildSpec(task="local_smoke", base_model=str(base), method=TrainingMethod.LORA,
                     quantization="none", runtime="hf", dataset=str(data), target_score=0,
                     extras={"max_steps": 2, "batch_size": 2, "max_length": 16})
    result = Factory(base_path=tmp_path / "registry").build(spec)
    assert result.success, result.reason
    artifact = __import__("pathlib").Path(result.artifact_path)
    assert (artifact / "deploy" / "model.safetensors").is_file()
    assert (artifact / "adapter" / "adapter_model.safetensors").is_file()
    predictions = predict_artifact(artifact, ["good case", "bad case"])
    assert len(predictions) == 2
    assert set(predictions) <= {"negative", "positive"}
