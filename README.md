# Majestic

Majestic is a research implementation of a compiler for task-specific machine
learning: typed specifications, feasibility planning, data preparation, training,
evaluation, artifact storage, and a typed workflow runtime.

**Release scope: local technical review and classification experiments.** This is
not a validated universal model factory or independently certified product.
Read the [release assessment](docs/REVIEW_STATUS.md) and
[review guide](docs/REVIEW_GUIDE.md). The [research paper](paper/majestic_research.tex)
is one self-contained LaTeX file with an embedded bibliography.

## Run the studio

```powershell
python -m venv .venv
python -m pip install uv
uv pip install --python .venv/Scripts/python.exe -e '.[api,dev]'
.\.venv\Scripts\python.exe run.py
```

Open http://127.0.0.1:8000. Load the specimen corpus, build, inspect the scorecard,
and classify text. This path trains a **TF-IDF centroid classifier**. Its neural
plan and Android figures are analytical illustrations. The specimen corpus is
synthetic and demonstrates software behavior only. This is a single-user local
application; public deployment needs authentication and storage hardening.

The local workspace already contains four trained neural artifacts. Select one
with **Use** in the model list to run inference. The review ZIP includes the
fixed-seed LoRA model in `sms_specialist/`; after extracting, install `.[api,ml]`
and run `python -m modelrig.review import-artifact --path sms_specialist` to add it
to that copy's studio. Direct offline inference also works with
`python -m modelrig.review predict --path sms_specialist --text "Call me tomorrow"`.
The ZIP also includes `sms_specialist_nf4`, the seed-zero quantized QLoRA model.
That model requires CUDA and the `qlora` extra. The float model also supports CPU.

## Reproduce the public-data experiments

```powershell
.\.venv\Scripts\python.exe -m modelrig.review prepare-sms
.\.venv\Scripts\python.exe -m modelrig.review benchmark
.\.venv\Scripts\python.exe -m modelrig.review predict --path docs/evidence/review_model --text 'Please call me after the meeting'
```

The download retrieves only public UCI data. Cleaning provenance and SHA-256
digests are recorded. Three fixed stratified splits compare majority, centroid
and kNN baselines, with int8 and packed int4 storage. Full
[measurements](docs/evidence/benchmark.json) include macro-F1, intervals, flips,
timings and planted graph checks.

## Train a neural specialist

```powershell
uv pip install --python .venv/Scripts/python.exe -e '.[ml]'
.\.venv\Scripts\python.exe -m modelrig.review doctor
.\.venv\Scripts\python.exe -m cli.main build --spec configs/specs/sms_lora.yaml
.\.venv\Scripts\python.exe -m modelrig.review predict --path registry/BUILD_ID --text 'Please call me after the meeting'
```

Use CUDA-enabled PyTorch on NVIDIA hardware. This backend trains a pretrained
transformer with PEFT LoRA. QLoRA loads an NF4 base and merges into a freshly
loaded unquantized base. Deployment quantization is a separate bitsandbytes
operation, evaluated before export. Model and tokenizer are bundled for offline
inference, and the Hub revision is recorded. The current neural backend performs
sequence classification. Unsupported export and distillation requests are rejected.

Install `.[qlora]` as well to reproduce QLoRA. `python scripts/train_review.py`
runs the three LoRA seeds and the seed-zero QLoRA experiment. The recorded base
revision is `12040accade4e8a0f71eabdb258fecc2e7e948be`; set `extras.revision` to
that value when reproducing these results. Default Hub branches can change.

For your task, supply UTF-8 CSV (`text,label`) or JSONL (`text`, `label`) and edit
the dataset and training settings in a copy of the spec. Keep an independent final
test set; do not repeatedly tune against the factory's reported holdout.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m build --wheel --outdir outputs/wheels
```

Private inputs are excluded from packages. The patent remains local; nothing
publishes it or submits the paper. Historical design claims are preserved in
[the previous README](docs/LEGACY_DESIGN.md), clearly marked as superseded.
