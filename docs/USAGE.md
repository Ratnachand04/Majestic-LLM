# Usage

Everything below runs **offline on CPU** with the default tiny/mock components.
Optional heavy features activate only when you install their extras.

## 1. Environment
```bash
# with uv (preferred)
uv venv --python 3.12 .venv
uv pip install -e '.[api,dev]'

# or with pip
python -m venv .venv && . .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -e '.[api,dev]'
```
Optional extras: `.[ml]` (torch/transformers/peft/trl), `.[retrieval]`
(sentence-transformers/faiss), `.[compress]` (bitsandbytes/gptq/onnx).

Configuration is read from `configs/default.yaml`, then overridden by a local
`.env` and by `MAJESTIC_*` environment variables (copy `.env.example` → `.env`).

## 2. Run the compound system (demo)
```bash
python demo/run_demo.py "What is Majestic LLM?"
```
Output shows the answer, the pipeline `trace`
(`encode -> retrieve -> plan -> route -> run -> ...`) and `verified`.

Programmatically:
```python
from majestic.factory import build_orchestrator
from majestic.types import Request

orch = build_orchestrator(knowledge=["Paris is the capital of France."])
resp = orch.handle(Request(content="What is the capital of France?"))
print(resp.content, resp.trace, resp.verified)
```

### Use a real reasoning core
Set `core.model` in `configs/default.yaml` (or `MAJESTIC_CORE_MODEL`) to a small
open instruct model id and install `.[ml]`. If loading fails the system falls
back to the mock core, so it always stays runnable.

## 3. The API
```bash
uvicorn api.app:app --reload
curl -s localhost:8000/generate -H 'content-type: application/json' \
     -d '{"prompt":"hello world"}'
```
`GET /health` returns `{"status":"ok"}`.

## 4. Build a device specialist (ModelRig)
Distill a capability into a tiny, quantized, device-checked model:
```bash
python -m cli.main build --capability sentiment --device android_midrange
```
This curates data → fits a classifier → **evaluates on a held-out split (the
gate)** → quantizes (int8/packed-int4) → exports an artifact, and registers it.

From a spec file:
```bash
python -m cli.main validate-spec my_spec.json
python -m cli.main build --spec my_spec.json
```
`my_spec.json`:
```json
{"task": "sentiment", "base_model": "centroid", "method": "centroid",
 "quantization": "int8", "target_score": 0.7, "dataset": "builtin:sentiment"}
```
Methods: `centroid`/`none_rag` run offline; `lora`/`qlora`/`distill` need `.[ml]`.

Programmatically:
```python
from modelrig.factory import Factory
from modelrig.buildspec import BuildSpec, TrainingMethod

result = Factory(base_path="./registry").build(
    BuildSpec(task="sentiment", base_model="centroid",
              method=TrainingMethod.CENTROID, quantization="int8")
)
print(result.success, result.eval_report, result.artifact_path)
```

## 5. Feasibility check
Will a build fit a device *before* you run it?
```bash
python -m cli.main feasibility --spec my_spec.json --device android_midrange
```
Returns predicted RAM / latency / battery, the device budget, headroom, and — on
failure — concrete reasons.

## 6. Testing
```bash
pytest                      # offline suite (green, no network/GPU)
pytest -m integration       # heavy paths (needs the optional extras)
```
