# Majestic LLM

**A model compiler, not a model.** You do not build one model flexible enough to
build any small model — you build a compiler. There is a specification IR, a
planner that runs verified passes over it, and a backend matrix that targets
devices. Flexibility is a property of the architecture, not of any one component.

<p align="center">
  <img src="docs/diagrams/pipeline.svg" alt="The Majestic compile pipeline: a plain-language request enters FORGE, becomes a hash-addressed Spec IR, and passes Gate 1 spec admissibility, Gate 2 plan feasibility and Gate 3 artefact certification before being admitted as a certified cartridge." width="100%">
</p>

Majestic turns a plain-language description into a **certified, device-sized
specialist**: it interviews the customer, type-checks the request against
physics, law and its own catalogue, refuses what is impossible *before spending
anything*, then builds, proves and packages what is possible. What ships is not a
model file — it is a **scorecard** proving a small specialist does one job as
well as a large generalist, plus the installable artefact for the device the
customer owns.

> **Status: implemented and runnable, offline on CPU.** The seven subsystems, the
> three-level IR with its three gates, the device-aware feasibility engine
> (including KV-cache accounting), the typed-DAG runtime, multi-adapter serving,
> constrained decoding, the learned deferral router and the flywheel all run with
> tiny/mock defaults and no downloads. Heavy paths (real HF models, PEFT LoRA,
> sentence-transformers, FAISS, GGUF/ONNX export) sit behind lazy imports.
> Frontier pre-training, JEPA/world-models and image generation remain deliberate
> stubs. See [docs/research-gaps.md](docs/research-gaps.md) for what is genuinely
> unsolved versus merely unimplemented.

## Quickstart
```bash
uv venv --python 3.12 .venv && uv pip install -e '.[api,dev]'
python demo/run_compiler_demo.py     # the ten-act worked example, end to end
python -m cli.main validate          # model compatibility + architecture conformance
python -m cli.main budget            # the B-09 device RAM budget, computed
pytest                               # green offline: no network, no GPU
```
Full walkthrough: [docs/USAGE.md](docs/USAGE.md). Conformance evidence:
[docs/conformance.md](docs/conformance.md).

## The compiler
```
FORGE -> Spec IR -> [GATE 1] -> PLANNER -> Build Plan IR -> [GATE 2]
      -> DATA FACTORY -> TRAINER -> PROVING GROUND -> [GATE 3]
      -> CARTRIDGE -> REGISTRY -> FABRIC
```

| # | Subsystem | What it does |
|---|---|---|
| 1 | **FORGE** | slot-filling interview that asks four questions, not forty — and picks them by pushing candidate answers through the **Planner** and keeping the ones that move the plan ([detail](docs/forge.md)) |
| 2 | **PLANNER** | seven hard predicates over an enumerable plan space; returns a plan **or a refusal with a witness**. Deterministic — no LLM on the common path ([detail](docs/planner.md)) |
| 3 | **DATA FACTORY** | seed-anchored amplification behind blocking QA gates; refuses below the real-seed floor |
| 4 | **TRAINER** | LoRA/QLoRA by default; parallel candidates where the plan is uncertain |
| 5 | **PROVING GROUND** | seven axes, all must pass; re-run in full after quantisation |
| 6 | **REGISTRY** | content-addressed cartridges; base stored once, adapters are deltas |
| 7 | **FABRIC** | typed DAG, statically verified for offline closure, RAM and cost — and it runs |

**Every gate is checked before any GPU is spent.** A refusal in eight seconds is
a better product than a failed build in ninety minutes:

```
admitted : False
stage    : gate1  (no planning, no data, no GPU)
  - seed data 5 is below the floor of 200 real examples for primitive 'generate'
  - quality bar unachievable: 'generate' needs at least 1.7B but
    android_lowend caps the base at 1.182B at 4-bit
```

## What is actually enforced
- **The largest base that fits at 4-bit** — never a smaller base at higher
  precision. Under a fixed RAM budget that is the k-bit scaling law's verdict.
- **Memory is weights + KV cache + runtime**, never weights alone. Free RAM must
  be ≥ 2× the model file, which is why a 4 GB phone tops out near 1.7B at 4-bit.
- **MoE is excluded from constrained devices.** Sparse activation is not sparse
  residency: 30B must be held to activate 3B.
- **Candidates are trained in parallel and one is chosen.** The largest base that
  fits RAM may still break the latency budget, and the budget is a contract.
- **Answer-flip rate**, not just accuracy. A quantised model can hold identical
  aggregate accuracy while changing a large share of individual answers. The
  quantiser is calibrated on the customer's own distribution and confidence is
  recalibrated afterwards.
- **A compiled grammar, not a hopeful prompt.** Schema violation is impossible
  rather than unlikely.
- **Untrusted content cannot reach a privileged tool** — proven statically, then
  enforced again at runtime.
- **A latency promise is never made from unmeasured hardware.** Effective
  throughput runs at 30–60% of peak; a planner that interpolates it is
  fabricating a commitment, so `P_lat` refuses unless the estimate is explicitly
  accepted — or until a **device probe** measures the target
  ([detail](docs/device-verification.md)).
- **The device certifies the model, not the vendor.** Two short benchmarks on the
  customer's own hardware calibrate `t = S/BW_eff + c` exactly, which predicts
  latency for the whole catalogue on that device. All promises use the
  *sustained* rate, after the thermal derate.
- **A cartridge is certified per device, and per artefact kind.** The same
  cartridge may be certified for one SoC and unverified for another; a device
  absent from the manifest is reported as unverified rather than assumed fine.
  Throughput alone certifies nothing — a run that did not execute the eval subset
  **on the device** is refused a certificate, because compilation and format
  conversion can change the outputs while every aggregate number holds steady.
- **Merging happens in BF16, never into the 4-bit training base.** There are two
  separate quantisations in the pipeline and conflating them loses most of the
  fine-tune while every step reports success.
- **Refusal is a return value, not an exception** — with a minimal witness and
  ordered remedies. Even a *free* build is refused below θ\* = (C+κ)/(V+κ),
  because the damage of shipping a bad model is not the compute you burned.
- **Latency is prefill + decode, and for document work prefill dominates.** A
  1000-token form producing 80 tokens of JSON puts ~67% of the time in prefill,
  so costing decode alone understates the truth by ~3×. `expected_input_tokens`
  is therefore a first-class Spec IR field, and a decode-only probe is not
  allowed to promise a prefill-bound workload.
- **Seven resource dimensions, six of them probed.** Storage and energy bind
  independently of memory: a phone can hold an artefact it cannot load, and a
  tablet doing 200 forms a day spends ~40% of its battery on inference
  ([detail](docs/forge.md)).
- **A question is asked only when its answer changes the build.** Every question
  costs attrition — `P(complete) = e^(−γq)` — so FORGE asks iff
  `IG·stake·Λ > γ`, and the information gain is *measured* by running the
  Planner over each candidate answer rather than guessed from parser entropy. A
  slot can be maximally ambiguous and still worth zero questions.

---

# Architecture diagrams

The complete set. **Part A** is prior art the design rests on, **Part B** is the
proposed architecture, **Part C** is honest comparison. Colours are consistent
throughout:

| | |
|---|---|
| 🟧 **Majestic component or rule** | what this project contributes |
| 🟥 **Binding constraint / verification gate** | what refuses a build |
| ⬛ **Contract artefact** | the IR everything binds to |
| 🟩 **Certified output** | what the customer receives |
| 🟦 **Prior art** | the published result being used |
| ⬜ **Commodity step / external actor** | available off the shelf |

---

## Part A — prior art

<details>
<summary><b>A-01 · Decoder-only transformer and the on-device memory budget</b> — why deployable size is bounded by weights <i>plus</i> KV cache <i>plus</i> runtime</summary>

<p align="center">
  <img src="docs/diagrams/memory-budget.svg" alt="Animated on-device memory budget showing the KV cache growing with context length until the total crosses device free RAM." width="100%">
</p>

```mermaid
flowchart LR
  subgraph TOKEN["TOKEN PATH"]
    direction TB
    T1["Tokenizer<br/><small>text to token IDs</small>"]
    T2["Embedding matrix<br/><small>vocab x d_model</small>"]
    T3["RMSNorm"]
    T4["Grouped-query attention<br/><small>writes to the KV cache</small>"]
    T5["Feed-forward / SwiGLU<br/><small>bulk of the parameters</small>"]
    T6["LM head<br/><small>logits over vocab</small>"]
    T7["Sample / constrain<br/><small>next token</small>"]
    T1 --> T2 --> T3 --> T4 --> T5 --> T6 --> T7
    T7 -. "autoregressive feedback" .-> T4
  end

  subgraph MEM["WHERE THE MEMORY GOES"]
    direction TB
    M1["<b>RESIDENT</b><br/>Model weights, quantised<br/><small>fixed cost, known before deployment</small>"]
    M2["<b>GROWS AT RUNTIME</b><br/>KV cache<br/><small>linear in context length and batch —<br/>the variable nobody budgets for</small>"]
    M3["<b>OVERHEAD</b><br/>Runtime and activations<br/><small>framework, embedder, grammar state</small>"]
    M4["TOTAL must fit inside device free RAM"]
    M1 --> M4
    M2 --> M4
    M3 --> M4
  end

  subgraph RULE["MAJESTIC RULES"]
    direction TB
    R1["<b>PLANNER CONSTRAINT</b><br/>Require free RAM of at least<br/><b>2x the model file size</b><br/><small>a 4 GB phone tops out near 1.7B at 4-bit,<br/>not the 3.5B file size alone suggests</small>"]
    R2["<b>K-BIT SCALING LAW</b><br/>4-bit is the optimal accuracy-per-bit point.<br/>Take the <b>largest base that fits at 4-bit</b> —<br/>never a smaller base at higher precision."]
  end

  T4 --> M2
  T5 --> M1
  M4 --> R1 --> R2

  classDef pri fill:#eaf0fb,stroke:#2a5caa,stroke-width:2px,color:#0f1d33
  classDef com fill:#f1f3f6,stroke:#8a94a6,stroke-width:1.5px,color:#1b2430
  classDef gate fill:#fdecec,stroke:#c62026,stroke-width:2px,color:#2a1113
  classDef maj fill:#fdf0ea,stroke:#e8562a,stroke-width:2px,color:#20160f
  class T3,T4,T5,T6 pri
  class T1,T2,T7,M1,M3 com
  class M2,M4 gate
  class R1,R2 maj
```

**Why it matters to Majestic:** the deployable model size is bounded by weights
PLUS KV cache PLUS runtime, not weights alone.
*Prior art: Vaswani et al. 1706.03762; GQA; Dettmers & Zettlemoyer 2212.09720.*

</details>

<details>
<summary><b>A-02 · Knowledge distillation — the two transfer paths</b> — path selection is decided by tokenizer identity, not preference</summary>

```mermaid
flowchart LR
  SRC["<b>SOURCE OF CAPABILITY</b><br/>Teacher model<br/><small>open-weight, 14B-70B class.<br/>Apache-2.0 or MIT only.</small>"]
  BOUND["<b>HARD LEGAL BOUNDARY</b><br/>Distilling from a closed commercial API<br/>breaches its terms. This single constraint<br/>eliminates the strongest teachers and<br/>shapes the entire base catalogue."]

  subgraph PATH["TRANSFER PATH — CHOSEN BY TOKENIZER MATCH"]
    direction TB
    P1["<b>HIGHEST FIDELITY</b><br/>Path 1 — Logit / KL distillation<br/><small>transfers the full probability mass,<br/>including 'dark knowledge' in the tail.<br/><b>REQUIRES an identical tokenizer</b> —<br/>only valid within one model family.</small>"]
    P2["<b>DEFAULT PATH</b><br/>Path 2 — Sequence-level distillation<br/><small>teacher generates a corpus; student trains<br/>on it with ordinary SFT. No logit access.<br/><b>Works across ANY tokenizer boundary.</b></small>"]
  end

  OUT["<b>OUTPUT</b><br/>Student model<br/><small>0.6B to 4B, task-specialised.<br/>Deployed as a cartridge.</small>"]
  OBJ["<b>OBJECTIVE CHOICE</b><br/>Reverse-KL is mode-seeking and less<br/>hallucinatory — used for extraction and<br/>classification. Forward-KL for open-ended<br/>drafting, where coverage matters."]
  GKD["<b>ON-POLICY VARIANT</b><br/>GKD trains on sequences the STUDENT<br/>sampled. Costlier, so it is triggered only<br/>inside the repair loop after a failed gate."]
  FAIL["<b>THE FAILURE MODE THAT DEFINES THE EVALUATION</b><br/>Naive distillation teaches the teacher's STYLE without its REASONING.<br/>The student sounds correct and is wrong — invisible to aggregate accuracy.<br/>This is why the Proving Ground measures answer-flip rate and behavioural tests."]

  SRC --> P1 --> OUT
  SRC --> P2 --> OUT
  BOUND -.-> SRC
  OUT --> OBJ
  OUT --> GKD
  OUT --> FAIL

  classDef pri fill:#eaf0fb,stroke:#2a5caa,stroke-width:2px,color:#0f1d33
  classDef maj fill:#fdf0ea,stroke:#e8562a,stroke-width:2px,color:#20160f
  classDef gate fill:#fdecec,stroke:#c62026,stroke-width:2px,color:#2a1113
  class SRC,P1,OBJ,GKD pri
  class P2,OUT maj
  class BOUND,FAIL gate
```

*Prior art: Hinton et al. 1503.02531; Kim & Rush 1606.07947; GKD 2306.13649; MiniLLM 2306.08543; Orca 2306.02707.*

</details>

<details>
<summary><b>A-03 · LoRA — the adapter that becomes the cartridge</b> — why 500 customer models occupy 16 GB instead of 550 GB</summary>

```mermaid
flowchart LR
  X["Input x"]
  W["<b>FROZEN</b><br/>Base weight W<br/><small>d x d, never updated.<br/>Shipped once, shared by every customer.</small>"]
  BA["<b>TRAINABLE</b><br/>Low-rank branch B·A<br/><small>A is d x r, B is r x d, r typically 8-64.<br/>The only trained tensors.</small>"]
  SUM(["+"])
  H["Output h = Wx + BAx"]
  PROP["<b>THE PROPERTY THAT MATTERS</b><br/>Because the update is ADDED it can be merged for zero latency,<br/>or kept separate so thousands of adapters share one resident base.<br/>Majestic depends on the second property."]

  NAIVE["<b>NAIVE</b><br/>Full fine-tuning, 500 customers<br/><b>500 x 1.1 GB = 550 GB</b>"]
  REG["<b>MAJESTIC REGISTRY</b><br/>Shared base plus adapters<br/><b>1.1 GB + (500 x 30 MB) = ~16 GB</b><br/><small>34x reduction</small>"]
  LIMIT["<b>THE DOCUMENTED LIMIT</b><br/>LoRA learns less and forgets less. It underperforms full fine-tuning<br/>when a genuinely NEW domain must be learned, but preserves general<br/>ability far better — so the planner escalates to full fine-tune only above<br/>a declared novel-vocabulary threshold, and runs a regression suite either way."]

  X --> W --> SUM
  X --> BA --> SUM
  SUM --> H
  BA -.-> PROP
  PROP --> NAIVE
  PROP --> REG
  REG --> LIMIT

  classDef pri fill:#eaf0fb,stroke:#2a5caa,stroke-width:2px,color:#0f1d33
  classDef maj fill:#fdf0ea,stroke:#e8562a,stroke-width:2px,color:#20160f
  classDef gate fill:#fdecec,stroke:#c62026,stroke-width:2px,color:#2a1113
  classDef com fill:#f1f3f6,stroke:#8a94a6,stroke-width:1.5px,color:#1b2430
  class W,H,PROP pri
  class BA,REG maj
  class NAIVE,LIMIT gate
  class X,SUM com
```

*Prior art: Hu et al. 2106.09685; QLoRA 2305.14314; DoRA 2402.09353; Biderman et al. 2405.09673.*

</details>

<details>
<summary><b>A-04 · Multi-adapter serving — one GPU, many customers</b> — the published result that makes hosted unit economics work</summary>

```mermaid
flowchart LR
  subgraph TRAFFIC["INCOMING TRAFFIC — MIXED TENANTS"]
    direction TB
    TA["Tenant A<br/><small>invoice extractor</small>"]
    TB["Tenant B<br/><small>ticket triage</small>"]
    TC["Tenant C<br/><small>policy Q and A</small>"]
    TD["Tenant D<br/><small>reply drafting</small>"]
  end

  subgraph GPU["SINGLE GPU"]
    direction TB
    BASE["<b>Shared base model</b><br/><small>one copy resident in VRAM</small>"]
    POOL["<b>Adapter pool — paged in and out on demand</b><br/><small>thousands of LoRA deltas, ~30 MB each</small>"]
    KERN["<b>Batched heterogeneous kernel (SGMV)</b><br/><small>requests using different adapters execute in ONE batch</small>"]
    KV["KV cache paged, near-zero fragmentation"]
    BASE --> POOL --> KERN --> KV
  end

  OUT["<b>Per-tenant responses</b><br/><small>each request answered by that customer's<br/>own specialised model, from one shared GPU</small>"]
  PUB["<b>PUBLISHED RESULT</b><br/>~1000 concurrent adapters on a single A100,<br/>with order-of-magnitude throughput gains<br/>over one replica per fine-tune."]
  BREAK["<b>WHAT BREAKS IT</b><br/>Customers on DIFFERENT base models cannot share a GPU.<br/>Heterogeneous adapter ranks also reduce batching efficiency.<br/>Both push Majestic toward a narrow base catalogue."]
  ADD["<b>WHAT MAJESTIC BUILDS ON TOP</b><br/>A spec-hash-keyed adapter cache with tier-driven eviction,<br/>cross-tenant isolation, and scheduling that arbitrates between<br/>customer inference and internal teacher generation."]

  TA --> BASE
  TB --> BASE
  TC --> BASE
  TD --> BASE
  KV --> OUT
  OUT --> PUB
  OUT --> BREAK
  BREAK --> ADD

  classDef pri fill:#eaf0fb,stroke:#2a5caa,stroke-width:2px,color:#0f1d33
  classDef maj fill:#fdf0ea,stroke:#e8562a,stroke-width:2px,color:#20160f
  classDef gate fill:#fdecec,stroke:#c62026,stroke-width:2px,color:#2a1113
  classDef com fill:#f1f3f6,stroke:#8a94a6,stroke-width:1.5px,color:#1b2430
  class BASE,KERN,KV,PUB pri
  class POOL,OUT,ADD maj
  class BREAK gate
  class TA,TB,TC,TD com
```

*Prior art: S-LoRA 2311.03285; Punica 2310.18547; vLLM PagedAttention 2309.06180.*

</details>

<details>
<summary><b>A-05 · Post-training quantisation</b> — where a good fine-tune quietly dies</summary>

```mermaid
flowchart LR
  IN["<b>INPUT</b><br/>FP16 fine-tuned model<br/><small>passed its task eval at full precision</small>"]
  CAL["<b>CRITICAL CHOICE</b><br/>Calibration set<br/><small>a few hundred representative samples</small>"]
  Q["<b>ALGORITHM</b><br/>Quantiser<br/><small>GPTQ: Hessian-based rounding<br/>AWQ: activation-aware scaling</small>"]
  OUT["<b>OUTPUT</b><br/>INT4 model<br/><small>roughly a quarter of the size</small>"]
  GATE["<b>MANDATORY GATE — re-run the full eval suite</b><br/>· task metric on real held-out data<br/>· <b>answer-flip rate against FP16</b><br/>· regression suite for general ability<br/>· recalibrate confidence — quantisation shifts<br/>&nbsp;&nbsp;the probability distribution"]
  PASS["Pass — package for target"]
  ESC["fail: escalate to SpQR or a larger base"]

  MOD["<b>THE MODIFICATION THAT MATTERS</b><br/>Majestic draws calibration data from the<br/>CUSTOMER's task distribution, not generic web text.<br/>Off-distribution calibration is the most common<br/>cause of a fine-tune evaporating at 4-bit."]
  WHY["<b>WHY IT BREAKS</b><br/>Outlier features above roughly 6.7B parameters<br/>are the specific mechanism of failure.<br/>Below that scale it is subtler and easier to miss."]
  ONE["<b>WHY ONE NUMBER IS NOT ENOUGH</b><br/>A quantised model can hold identical aggregate accuracy while changing a large<br/>fraction of individual answers. Aggregate parity is an illusion — so the<br/>answer-flip rate goes on the customer scorecard."]

  IN --> CAL --> Q --> OUT --> GATE
  GATE --> PASS
  GATE -. "fail" .-> ESC
  ESC -.-> Q
  CAL -.-> MOD
  Q -.-> WHY
  GATE --> ONE

  classDef pri fill:#eaf0fb,stroke:#2a5caa,stroke-width:2px,color:#0f1d33
  classDef maj fill:#fdf0ea,stroke:#e8562a,stroke-width:2px,color:#20160f
  classDef gate fill:#fdecec,stroke:#c62026,stroke-width:2px,color:#2a1113
  classDef ok fill:#eaf6ef,stroke:#1b7f4b,stroke-width:2px,color:#0e2a1b
  class IN,Q,OUT,WHY pri
  class CAL,MOD maj
  class GATE,ESC,ONE gate
  class PASS ok
```

*Prior art: GPTQ 2210.17323; AWQ 2306.00978; LLM.int8(); "Accuracy is Not All You Need" 2407.09141.*

</details>

<details>
<summary><b>A-06 · Retrieval-augmented generation</b> — knowledge in the index, behaviour in the adapter</summary>

```mermaid
flowchart LR
  subgraph IDX["INDEX TIME — RUNS ONCE, THEN ON CHANGE"]
    direction LR
    D1["<b>SOURCE</b><br/>Customer documents<br/><small>manuals, catalogues, tickets</small>"]
    D2["Chunk and embed"]
    D3["<b>STORE</b><br/>Vector index<br/><small>FAISS / LanceDB</small>"]
    D1 --> D2 --> D3
  end

  subgraph QRY["QUERY TIME — EVERY REQUEST"]
    direction LR
    Q1["User query"]
    Q2["Embed query"]
    Q3["Retrieve top-k"]
    Q4["<b>Assemble context</b><br/><small>rerank, then place the strongest<br/>evidence at the context BOUNDARIES</small>"]
    Q5["<b>CARTRIDGE</b><br/>Small model generates<br/><small>grounded answer with citations</small>"]
    Q1 --> Q2 --> Q3 --> Q4 --> Q5
  end

  MORE["<b>MORE CONTEXT IS NOT BETTER</b><br/>A U-shaped positional curve means evidence placed<br/>mid-context is largely ignored — and small models are<br/>WORSE at this than the models studied. Majestic caps<br/>chunks aggressively and budgets context."]
  ADAPT["<b>ADAPTIVE RETRIEVAL</b><br/>Self-RAG trains the model to decide WHEN to retrieve.<br/>Majestic reuses those reflection tokens as the router's<br/>confidence signal — one mechanism serving both<br/>retrieval control and cloud escalation."]
  SPLIT["<b>THE ARCHITECTURAL SPLIT MAJESTIC DEPENDS ON</b><br/><b>Knowledge lives in the index. Behaviour lives in the adapter.</b><br/>A pharmacy adding 400 products reindexes: seconds, no GPU, no cost.<br/>Changing HOW the model answers is a rebuild: hours, forty dollars.<br/>Conflating the two turns every catalogue update into a retraining bill."]

  D3 -.-> Q3
  Q4 --> MORE
  Q3 --> ADAPT
  Q5 --> SPLIT

  classDef pri fill:#eaf0fb,stroke:#2a5caa,stroke-width:2px,color:#0f1d33
  classDef maj fill:#fdf0ea,stroke:#e8562a,stroke-width:2px,color:#20160f
  classDef gate fill:#fdecec,stroke:#c62026,stroke-width:2px,color:#2a1113
  classDef com fill:#f1f3f6,stroke:#8a94a6,stroke-width:1.5px,color:#1b2430
  class D2,D3,Q2,Q3,ADAPT pri
  class Q4,Q5,SPLIT maj
  class MORE gate
  class D1,Q1 com
```

*Prior art: Lewis et al. 2005.11401; DPR 2004.04906; Self-RAG 2310.11511; Lost in the Middle 2307.03172.*

</details>

<details>
<summary><b>A-07 · Sparse mixture-of-experts routing</b> — the ancestor of cartridge routing, and a warning about phones</summary>

```mermaid
flowchart LR
  TOK["Token"]
  GATE["<b>ROUTER</b><br/>Gating network<br/><small>scores every expert, selects the top k</small>"]
  E1["Expert 1<br/><small>selected</small>"]
  E2["Expert 2"]
  E3["Expert 3<br/><small>selected</small>"]
  EN["Expert N"]
  COMB["Weighted combine"]
  OUT["Output"]

  PRIN["<b>THE PRINCIPLE</b><br/>Capacity grows with the number of experts while<br/>compute per token stays constant. This is what<br/>Majestic borrows: many specialists, sparse<br/>activation, one gate."]
  COARSE["<b>COARSE ROUTING, ON PURPOSE</b><br/>MoE routes per TOKEN across jointly-trained experts.<br/>Majestic routes per REQUEST across whole cartridges.<br/>Coarser — but the experts become independently<br/>trainable, versionable, auditable and sellable."]
  EDGE["<b>WHY MoE IS NOT AN EDGE ANSWER</b><br/>Sparse activation does NOT mean sparse residency:<br/>the full expert set must be in memory. MoE bases are<br/>excluded by the planner from every memory-constrained<br/>target. Confusing active with resident parameters is a<br/>common and costly mistake."]

  TOK --> GATE
  GATE --> E1 --> COMB
  GATE -.-> E2
  GATE --> E3 --> COMB
  GATE -.-> EN
  COMB --> OUT
  GATE --> PRIN
  OUT --> COARSE
  OUT --> EDGE

  classDef pri fill:#eaf0fb,stroke:#2a5caa,stroke-width:2px,color:#0f1d33
  classDef maj fill:#fdf0ea,stroke:#e8562a,stroke-width:2px,color:#20160f
  classDef gate fill:#fdecec,stroke:#c62026,stroke-width:2px,color:#2a1113
  classDef com fill:#f1f3f6,stroke:#8a94a6,stroke-width:1.5px,color:#1b2430
  class GATE,COMB,PRIN pri
  class E1,E3,COARSE maj
  class EDGE gate
  class TOK,E2,EN,OUT com
```

*Prior art: Shazeer et al. 1701.06538; Switch Transformer 2101.03961; Mixtral 2401.04088.*

</details>

<details>
<summary><b>A-08 · LLM cascades and learned routing</b> — Majestic inverts the economics: the cheap tier is the user's own device</summary>

```mermaid
flowchart LR
  Q["Incoming query"]
  R["<b>DECISION</b><br/>Router / scorer<br/><small>predicts whether the small<br/>model will answer well</small>"]
  T1["<b>TIER 1</b><br/>Small / cheap model<br/><small>answers the majority</small>"]
  T2["<b>TIER 2</b><br/>Large / expensive model<br/><small>handles the hard tail</small>"]
  RESP["Response"]

  FRUGAL["<b>FRUGALGPT</b><br/>Up to 98% cost reduction while matching or beating<br/>the strongest single model — because most real<br/>queries do not need the strongest model."]
  QGAP["<b>QUALITY-GAP ROUTING</b><br/>Route on predicted quality GAP rather than absolute<br/>difficulty, exposing a tunable cost-quality knob.<br/>Majestic surfaces that knob to the customer:<br/><i>how often may this call the cloud?</i>"]
  CHANGE["<b>HOW MAJESTIC CHANGES THE PICTURE</b><br/>· <b>Tier 1 runs on the user's own device</b> — zero marginal cost, zero data egress<br/>· <b>Offline mode hard-locks the threshold</b> — escalation is architecturally disabled<br/>· <b>Every escalation is logged as a hard case</b> — the router closes the flywheel"]
  CONS["<b>THE CONSTRAINT ON THIS DESIGN</b><br/>Confidence-based deferral is provably suboptimal for specialist models under<br/>distribution shift — precisely Majestic's setting. The router must be a LEARNED<br/>deferral rule, not a softmax threshold, and its accuracy caps whole-system accuracy."]

  Q --> R
  R -- "high confidence" --> T1 --> RESP
  R -- "low confidence" --> T2 --> RESP
  T1 --> FRUGAL
  R --> QGAP
  RESP --> CHANGE --> CONS

  classDef pri fill:#eaf0fb,stroke:#2a5caa,stroke-width:2px,color:#0f1d33
  classDef maj fill:#fdf0ea,stroke:#e8562a,stroke-width:2px,color:#20160f
  classDef gate fill:#fdecec,stroke:#c62026,stroke-width:2px,color:#2a1113
  classDef com fill:#f1f3f6,stroke:#8a94a6,stroke-width:1.5px,color:#1b2430
  class R,T2,FRUGAL,QGAP pri
  class T1,CHANGE maj
  class CONS gate
  class Q,RESP com
```

*Prior art: FrugalGPT 2305.05176; RouteLLM 2406.18665; Hybrid LLM 2404.14618; 2307.02764.*

</details>

<details>
<summary><b>A-09 · The ReAct loop and tool calling</b> — web scraping is a tool the model calls, never a capability it contains</summary>

```mermaid
flowchart TB
  subgraph LOOP["THE LOOP"]
    direction LR
    RE["<b>REASON</b><br/>Thought<br/><small>what do I need next?</small>"]
    AC["<b>ACT</b><br/>Action<br/><small>call a registered tool</small>"]
    OB["<b>OBSERVE</b><br/>Observation<br/><small>tool result returns</small>"]
    RE --> AC --> OB
    OB -. "repeat until done" .-> RE
  end

  REG["<b>SIDE EFFECTS LIVE HERE</b><br/>Tool registry<br/><small>scraper, database, calculator,<br/>customer API, file system</small>"]
  SMALL["<b>SMALL BEATS LARGE HERE</b><br/>Gorilla is the clearest proof of the Majestic thesis<br/>inside the tool domain: a fine-tuned 7B model beat far<br/>larger general models at correct API invocation.<br/>Specialisation wins where the task is narrow."]
  ADOPT["<b>WHAT MAJESTIC ADOPTS</b><br/>Retriever-aware training. Train the model to USE<br/>retrieved tool documentation rather than memorise<br/>signatures, and an API change becomes a reindex<br/>instead of a retrain."]
  FAIL["<b>WHERE SMALL MODELS FAIL</b><br/>Hard capability ceiling: sub-2B models degrade badly<br/>across multi-step loops. Majestic trains cartridges to emit<br/>a CONSTRAINED tool-call schema rather than free-form<br/>reasoning, and the planner refuses specs whose step<br/>depth exceeds what the device tier can sustain."]
  SEC["<b>THE SECURITY CONSEQUENCE</b><br/>Indirect prompt injection means every scraped page and retrieved document is<br/>untrusted INPUT, not data. Any graph where untrusted content can reach a<br/>privileged tool is an exploit waiting to happen."]

  AC --> REG
  REG --> SMALL
  REG --> ADOPT
  REG --> FAIL
  REG --> SEC

  classDef pri fill:#eaf0fb,stroke:#2a5caa,stroke-width:2px,color:#0f1d33
  classDef maj fill:#fdf0ea,stroke:#e8562a,stroke-width:2px,color:#20160f
  classDef gate fill:#fdecec,stroke:#c62026,stroke-width:2px,color:#2a1113
  classDef com fill:#f1f3f6,stroke:#8a94a6,stroke-width:1.5px,color:#1b2430
  class RE,OB,SMALL pri
  class AC,ADOPT maj
  class FAIL,SEC gate
  class REG com
```

*Prior art: ReAct 2210.03629; Toolformer 2302.04761; Gorilla 2305.15334; indirect prompt injection 2302.12173.*

</details>

<details>
<summary><b>A-10 · Compiler architecture</b> — the template Majestic copies, and the answer to the founding question</summary>

```mermaid
flowchart LR
  subgraph FE["FRONT ENDS"]
    direction TB
    F1["C / C++"]
    F2["Rust"]
    F3["Swift"]
    F4["Fortran"]
  end

  P["Parser and lowering"]
  IR["<b>THE PIVOT</b><br/>Intermediate representation"]

  subgraph OPT["OPTIMISATION PASSES"]
    direction TB
    O1["Inlining"]
    O2["Dead-code elimination"]
    O3["Loop transforms"]
    O4["Register allocation"]
  end

  subgraph BE["BACK ENDS"]
    direction TB
    B1["x86-64"]
    B2["ARM64"]
    B3["RISC-V"]
    B4["WebAssembly"]
  end

  OUT["<b>OUTPUT</b><br/>Object"]
  XFER["<b>THE TRANSFER TO MAJESTIC</b><br/><b>LLVM does not contain every program. It contains an IR, a pass pipeline and a backend matrix.</b><br/>No single component of LLVM is universal, and yet the system compiles anything. Majestic is built the<br/>same way: a specification IR, a planner that runs verified passes, and a backend matrix that targets<br/>devices. Flexibility is a property of the architecture, not of any one part — which is exactly why the<br/>'one flexible model' framing has no answer and this one does."]

  F1 --> P
  F2 --> P
  F3 --> P
  F4 --> P
  P --> IR --> OPT --> BE --> OUT --> XFER

  classDef pri fill:#eaf0fb,stroke:#2a5caa,stroke-width:2px,color:#0f1d33
  classDef maj fill:#fdf0ea,stroke:#e8562a,stroke-width:2px,color:#20160f
  classDef com fill:#f1f3f6,stroke:#8a94a6,stroke-width:1.5px,color:#1b2430
  class P,IR,OUT pri
  class XFER maj
  class F1,F2,F3,F4,O1,O2,O3,O4,B1,B2,B3,B4 com
```

*Prior art: LLVM (Lattner & Adve, CGO 2004); TVM 1802.04799; MLIR 2002.11054.*

</details>

---

## Part B — the proposed architecture

### B-01 · Majestic system architecture — the seven subsystems

> **Master diagram.** Every other diagram in Part B expands one block of this one.
> **LLMs occupy exactly five roles; everything else is deterministic software.**

```mermaid
flowchart TB
  CUST["<b>ACTOR</b><br/>Customer<br/><small>non-technical owner, or developer</small>"]

  subgraph FRONT["FRONT END"]
    direction LR
    S1["<b>SUBSYSTEM 1</b><br/>FORGE — specification builder<br/><small>interview agent for non-technical users;<br/>direct IR editing for developers. Fills a TYPED<br/>SLOT SCHEMA, asking only about slots left<br/>empty or ambiguous. <b>LLM ROLE 1 — front end.</b></small>"]
    SPEC["<b>THE PIVOT — SPEC IR</b><br/><small>hash-addressed, versioned, typed JSON.<br/>Task primitive, I/O schema, device target, offline<br/>flag, latency budget, policy, data rights, quality bar.<br/><b>The contract everything downstream depends on.</b></small>"]
    S1 --> SPEC
  end

  S2["<b>SUBSYSTEM 2</b><br/>PLANNER — build plan compiler<br/><small>constrained search over a typed parts catalogue with HARD<br/>feasibility predicates. Deterministic by default; an LLM proposes<br/>only when no precedent matches, and a validator always disposes.<br/><b>LLM ROLE 5 — tie-breaker.</b></small>"]

  subgraph BUILD["BUILD PIPELINE — ASYNC, METERED, CACHED"]
    direction LR
    S3["<b>SUBSYSTEM 3</b><br/>DATA FACTORY<br/><small>seed-anchored synthetic generation from the<br/>customer's own documents. Dedup, quality filter,<br/>diversity monitor, PII scrub. Real held-out set<br/>locked and never trained on.<br/><b>LLM ROLE 2 — teacher.</b></small>"]
    S4["<b>SUBSYSTEM 4</b><br/>TRAINER<br/><small>LoRA / QLoRA by default, full fine-tune for paid<br/>tiers, DPO and KTO for preference. Parallel<br/>candidate builds where the plan is uncertain.<br/>Zero custom training loops.</small>"]
    S5["<b>SUBSYSTEM 5</b><br/>PROVING GROUND<br/><small>task metrics, calibrated judge, behavioural tests,<br/>regression, safety, post-quantisation re-eval.<br/><b>GATE-ENFORCED:</b> below threshold the model<br/>never reaches the customer.<br/><b>LLM ROLES 3 and 4 — judge and repairer.</b></small>"]
    S3 --> S4 --> S5
  end

  subgraph RUNTIME["OUTPUT AND RUNTIME"]
    direction LR
    S6["<b>SUBSYSTEM 6</b><br/>REGISTRY<br/><small>content-addressed cartridges. Base stored once,<br/>adapters are deltas. Model card, eval certificate,<br/>licence chain and provenance on every artefact.</small>"]
    S7["<b>SUBSYSTEM 7</b><br/>FABRIC<br/><small>typed DAG runtime. Nodes are cartridges, tools and<br/>control flow. Statically verified for offline closure,<br/>RAM and cost. Runs on server, on device, or hybrid.</small>"]
    DELIV["<b>WHAT IS SOLD — DELIVERABLE</b><br/><small>Not a model file — a SCORECARD proving a 1 GB<br/>specialist does one job as well as a 140 GB generalist,<br/>plus the installable artefact for the customer's device.</small>"]
    S6 --> S7 --> DELIV
  end

  CUST -- "spec" --> S1
  SPEC --> S2 --> S3
  S5 --> S6
  S5 -. "<b>REPAIR LOOP</b> — plan mutation on gate failure" .-> S3
  DELIV -. "field corrections feed the flywheel" .-> S1

  classDef maj fill:#fdf0ea,stroke:#e8562a,stroke-width:2px,color:#20160f
  classDef gate fill:#fdecec,stroke:#c62026,stroke-width:2px,color:#2a1113
  classDef con fill:#0f1723,stroke:#0f1723,stroke-width:2px,color:#ffffff
  classDef ok fill:#eaf6ef,stroke:#1b7f4b,stroke-width:2px,color:#0e2a1b
  classDef com fill:#f1f3f6,stroke:#8a94a6,stroke-width:1.5px,color:#1b2430
  class S1,S2,S3,S4,S6,S7 maj
  class S5 gate
  class SPEC con
  class DELIV ok
  class CUST com
```

### B-02 · The compiler mapping: LLVM to Majestic

```mermaid
flowchart LR
  subgraph L["LLVM / TVM"]
    direction TB
    L1["C, Rust, Swift source"]
    L2["Parser and lowering"]
    L3["Intermediate representation"]
    L4["Optimisation passes"]
    L5["Back ends: x86, ARM, WASM"]
    L6["Object file"]
    L7["Test suite"]
  end

  subgraph M["MAJESTIC"]
    direction TB
    M1["Plain-language description<br/><small>human intent enters the system</small>"]
    M2["FORGE interview agent<br/><small>intent becomes a typed artefact</small>"]
    M3["SPEC IR<br/><small>the single contract everything binds to</small>"]
    M4["PLANNER — base, data, method, bit-width<br/><small>choices made under hard constraints</small>"]
    M5["Targets: GGUF, ExecuTorch, CoreML, vLLM<br/><small>one plan, many deployment shapes</small>"]
    M6["CARTRIDGE<br/><small>the shippable, versioned artefact</small>"]
    M7["PROVING GROUND scorecard<br/><small>proof the artefact does what was asked</small>"]
  end

  L1 --> M1
  L2 --> M2
  L3 --> M3
  L4 --> M4
  L5 --> M5
  L6 --> M6
  L7 --> M7

  POINT["<b>THE POINT</b><br/>No component of LLVM is universal. The IR is not universal, no single pass is universal, no backend is universal.<br/>Flexibility is an emergent property of <b>IR x passes x backends</b>. Majestic inherits that property exactly:<br/>six bases, a few hundred adapters, a dozen tools and eight primitives compose into millions of distinct<br/>producible models without any one part needing to be magic."]
  M7 --> POINT

  classDef maj fill:#fdf0ea,stroke:#e8562a,stroke-width:2px,color:#20160f
  classDef con fill:#0f1723,stroke:#0f1723,stroke-width:2px,color:#ffffff
  classDef ok fill:#eaf6ef,stroke:#1b7f4b,stroke-width:2px,color:#0e2a1b
  classDef pri fill:#eaf0fb,stroke:#2a5caa,stroke-width:2px,color:#0f1d33
  class L1,L2,L3,L4,L5,L6,L7 pri
  class M1,M2,M4,M5,M6,POINT maj
  class M3 con
  class M7 ok
```

### B-03 · The three-level IR stack and its verification gates

```mermaid
flowchart LR
  L1["<b>LEVEL 1 — SPEC IR</b><br/><i>what the customer wants</i><br/><small>task_primitive · io_schema · languages · device_target<br/>offline_required · latency_budget_ms · quality_gate<br/>seed_data_ref · policy_rules · data_rights</small>"]
  G1["<b>GATE 1 — SPEC ADMISSIBILITY</b><br/>· Is the primitive inside the supported set of eight?<br/>· Does the seed-data volume clear the floor for this primitive?<br/>· Are the data rights sufficient to train and to deploy?<br/>· Is the quality bar achievable at this device tier at all?"]

  L2["<b>LEVEL 2 — BUILD PLAN IR</b><br/><i>how the system will make it</i><br/><small>base_ref · teacher_ref · distil_mode · peft_method · rank<br/>data_recipe · quantiser · bit_width · grammar_ref<br/>eval_suite_ref · budget_usd</small>"]
  G2["<b>GATE 2 — PLAN FEASIBILITY</b><br/>· Does base size at this bit-width fit device RAM, <b>with KV cache</b>?<br/>· Does predicted latency clear the budget on MEASURED hardware?<br/>· Tokenizer identity verified if logit distillation is requested.<br/>· <b>OFFLINE CLOSURE:</b> every node locally resolvable, zero API calls.<br/>· Licence chain composes for base, teacher, data and jurisdiction."]

  L3["<b>LEVEL 3 — ARTEFACT IR</b><br/><i>what actually exists on disk</i><br/><small>adapter_blob_hash · quantised_blob_hash · grammar_blob<br/>index_ref · tool_manifest · model_card<br/>eval_certificate · licence_chain</small>"]
  G3["<b>GATE 3 — ARTEFACT CERTIFICATION</b><br/>· Eval suite passed on REAL held-out customer data.<br/>· Re-evaluated after quantisation; answer-flip rate within bound.<br/>· Regression, safety and privacy suites all clean.<br/>· Only then does the artefact enter the registry."]

  OUT["<b>CARTRIDGE — admitted to registry</b><br/><small>immutable, hash-addressed, certified, licensed</small>"]
  DIFF["<b>WHY THIS IS THE DIFFERENTIATOR</b><br/>Every other fine-tuning service accepts a config file and starts a GPU job. Majestic type-checks the request<br/>against physics, law and its own catalogue first, and refuses impossible builds before they cost anything.<br/><b>A refusal in eight seconds is a better product than a failed build in ninety minutes.</b>"]

  L1 -. "checked by" .-> G1
  L1 -- "lower" --> L2
  L2 -. "checked by" .-> G2
  L2 -- "lower" --> L3
  L3 -. "checked by" .-> G3
  L3 --> OUT --> DIFF

  classDef con fill:#0f1723,stroke:#0f1723,stroke-width:2px,color:#ffffff
  classDef maj fill:#fdf0ea,stroke:#e8562a,stroke-width:2px,color:#20160f
  classDef gate fill:#fdecec,stroke:#c62026,stroke-width:2px,color:#2a1113
  classDef ok fill:#eaf6ef,stroke:#1b7f4b,stroke-width:2px,color:#0e2a1b
  class L1 con
  class L2,L3 maj
  class G1,G2,G3,DIFF gate
  class OUT ok
```

### B-04 · FORGE — from plain language to a typed specification

```mermaid
flowchart TB
  IN["<b>INPUT</b><br/>Customer describes the need<br/><small>free text. No ML vocabulary required.<br/>May be voice, typed, or an uploaded brief.</small>"]
  ST1["<b>STEP 1</b><br/>Parse into partial Spec IR<br/><small>extract every slot the description already<br/>determines. Score confidence per slot.</small>"]
  ST2["<b>STEP 2</b><br/>Detect ambiguity and gaps<br/><small>'Works offline' — always, or only when the<br/>network drops? <b>Silent defaulting on an ambiguous<br/>slot is the top cause of a wrong build.</b></small>"]
  ST3["<b>STEP 3</b><br/>Ask ONLY the unfilled slots<br/><small>ranked by information gain. Device specification<br/>offered as a QR auto-probe, not a question.</small>"]
  ANS["Customer answers<br/><small><b>four questions, not forty</b></small>"]
  OUT["<b>Emit hash-addressed Spec IR</b><br/><small>versioned, diffable, replayable. An identical hash<br/>means the build is served from cache at zero marginal cost.</small>"]

  SLOTS["<b>THE SPEC IR SLOT SCHEMA</b><br/>task_primitive · io_schema · languages · device_target<br/>offline_required · latency_budget_ms · quality_gate<br/>seed_data_ref · abstention_policy · policy_rules<br/>data_rights · budget_ceiling_usd"]
  WHY["<b>WHY TYPED SLOTS</b><br/>Every slot carries a type and a validator.<br/>A specification that cannot be type-checked cannot be<br/>compiled — which is precisely why the interview<br/><b>terminates instead of wandering</b>."]
  STAR["<b>HOW THE INTERVIEW IMPROVES OVER TIME</b><br/>STaR-GATE closes the loop: reward the questions that produced builds which passed their gate<br/>on the FIRST attempt. Majestic has a far stronger reward signal than the paper does, because the<br/>outcome is objective — did the build pass, and did the customer accept it."]

  IN --> ST1 --> ST2 --> ST3 --> ANS
  ANS -. "refill slots" .-> ST2
  ST3 --> OUT
  ST1 --> SLOTS
  SLOTS --> WHY
  OUT --> STAR

  classDef maj fill:#fdf0ea,stroke:#e8562a,stroke-width:2px,color:#20160f
  classDef gate fill:#fdecec,stroke:#c62026,stroke-width:2px,color:#2a1113
  classDef con fill:#0f1723,stroke:#0f1723,stroke-width:2px,color:#ffffff
  classDef com fill:#f1f3f6,stroke:#8a94a6,stroke-width:1.5px,color:#1b2430
  class ST1,ST3,WHY,SLOTS,STAR maj
  class ST2 gate
  class OUT con
  class IN,ANS com
```

### B-05 · PLANNER — deterministic first, LLM only at the edges

```mermaid
flowchart TB
  SPEC["<b>Spec IR arrives</b>"]
  WARM["<b>STEP 1 — WARM START</b><br/>Retrieve similar past builds<br/><small>embed the spec, search build history for outcomes</small>"]
  DEC{"<b>PRECEDENT FOUND?</b><br/><small>does a past plan match this<br/>spec shape and pass its gate?</small>"}
  COMMON["<b>COMMON PATH</b><br/>Reuse and adapt the plan<br/><small>deterministic. No LLM involved.</small>"]
  RARE["<b>RARE PATH</b><br/>LLM proposes a plan<br/><small>given the catalogue and past outcomes.<br/><b>LLM ROLE 5 — tie-breaker only.</b></small>"]

  VAL["<b>VALIDATOR — THE LLM PROPOSES, THIS DISPOSES</b><br/>· Base size at bit-width fits device RAM <b>including KV cache</b><br/>· Predicted latency clears budget on MEASURED hardware, not FLOPs<br/>· Tokenizer identity verified if logit distillation is requested<br/>· Offline closure holds — no node requires the network<br/>· Licence chain composes for base, teacher, data and jurisdiction<br/>· Predicted build cost is inside the customer's tier ceiling<br/>· Primitive is inside the supported set of eight"]
  OUT["<b>Build Plan IR — admitted</b><br/><small>every predicate holds. <b>Only now is a GPU allocated.</b></small>"]

  CAT["<b>THE CATALOGUE THE PLANNER SEARCHES</b><br/><b>Bases</b> 6 open-weight, 0.6B to 8B<br/><b>PEFT</b> LoRA, DoRA, QLoRA, full FT<br/><b>Distil</b> logit KD, sequence KD, GKD<br/><b>Quantisers</b> GPTQ, AWQ, k-quants, SpQR<br/><b>Targets</b> GGUF, ExecuTorch, CoreML, ONNX, vLLM<br/><b>Primitives</b> the eight structural task types"]
  REJ["<b>THE REJECTED ALTERNATIVE</b><br/>Majestic deliberately rejects search-based design in the build loop. Neural architecture search costs<br/>thousands of GPU-hours per result, irreconcilable with a $20-$120 build. <b>Rules plus meta-learning, not search.</b><br/>An LLM never freestyles a training plan — a hallucinated tokenizer mismatch costs $60 of GPU to discover."]

  SPEC --> WARM --> DEC
  DEC -- "yes" --> COMMON --> VAL
  DEC -- "no" --> RARE --> VAL
  VAL --> OUT
  VAL -. "rejected: mutate and re-validate" .-> RARE
  WARM --> CAT
  OUT --> REJ

  classDef maj fill:#fdf0ea,stroke:#e8562a,stroke-width:2px,color:#20160f
  classDef gate fill:#fdecec,stroke:#c62026,stroke-width:2px,color:#2a1113
  classDef con fill:#0f1723,stroke:#0f1723,stroke-width:2px,color:#ffffff
  classDef ok fill:#eaf6ef,stroke:#1b7f4b,stroke-width:2px,color:#0e2a1b
  class WARM,RARE,CAT maj
  class VAL,REJ gate
  class SPEC con
  class COMMON,OUT ok
```

### B-06 · DATA FACTORY — amplification with collapse guardrails

```mermaid
flowchart TB
  REAL["<b>REALITY</b><br/>Customer documents<br/><small>200 real forms, a product list, a ticket archive</small>"]
  SPLIT["<b>SPLIT — IMMUTABLE</b><br/>150 seed · <b>50 LOCKED</b><br/><small>the 50 are never trained on. Ever.</small>"]

  BT["<b>PRIMARY PATH</b><br/>Backtranslation<br/><small>generate instructions FOR the customer's existing<br/>documents. The response side is their REAL data,<br/>so the teacher cannot hallucinate the answer.</small>"]
  EV["<b>HARD-CASE COVERAGE</b><br/>Evolution<br/><small>deliberately evolve hard cases: smudged scans,<br/>missing fields, format drift, adversarial inputs.<br/>A flat generator never produces these.</small>"]
  RT["<b>CAPABILITY TRANSFER</b><br/>Rationale traces<br/><small>teacher explains its reasoning. Traces are<br/>training-time only and stripped from the<br/>deployed output contract.</small>"]

  POOL["<b>BEFORE FILTERING</b><br/>Raw generated pool — ~60,000 candidate examples"]
  QA["<b>QA GATES — EACH BLOCKS THE BUILD, NONE MERELY WARNS</b><br/><b>MinHash dedup</b> — exact and near-duplicate removal; also checks train and eval never overlap<br/><b>Semantic dedup</b> — lexically varied, semantically identical rows; embedding-cluster pruning<br/><b>Diversity entropy monitor</b> — kills the run on distribution collapse<br/><b>Quality classifier</b> — LLM rater calibrated against human labels per domain<br/><b>PII scrub</b> — before training, not after. Weights ship to devices and memorisation is measurable."]
  TRAIN["<b>TO TRAINER</b><br/>Curated training set — real seeds accumulated, never replaced"]
  EVAL["held-out set goes straight to evaluation"]

  INV["<b>REGISTRY INVARIANT</b><br/><b>Accumulate, do not replace.</b> Every retraining<br/>generation keeps the whole real corpus and mixes<br/>it in. This is what makes the flywheel safe."]
  BOUND["<b>THE BOUNDING CONSTRAINT</b><br/>The Curse of Recursion: training on generated data causes irreversible tail loss.<br/><b>Below a minimum real-seed floor, Majestic REFUSES the build</b> rather than shipping<br/>a model that will quietly degrade."]

  REAL --> SPLIT
  SPLIT -- "seeds" --> BT --> POOL
  SPLIT -- "seeds" --> EV --> POOL
  SPLIT -- "seeds" --> RT --> POOL
  SPLIT -.-> EVAL
  POOL --> QA --> TRAIN
  SPLIT --> INV
  QA --> BOUND

  classDef maj fill:#fdf0ea,stroke:#e8562a,stroke-width:2px,color:#20160f
  classDef gate fill:#fdecec,stroke:#c62026,stroke-width:2px,color:#2a1113
  classDef ok fill:#eaf6ef,stroke:#1b7f4b,stroke-width:2px,color:#0e2a1b
  classDef com fill:#f1f3f6,stroke:#8a94a6,stroke-width:1.5px,color:#1b2430
  class BT,EV,RT maj
  class QA,BOUND gate
  class SPLIT,TRAIN,INV,EVAL ok
  class REAL,POOL com
```

### B-07 · PROVING GROUND — the gate, and the repair loop behind it

> **This subsystem is the actual product.** A build that cannot be proven correct
> on the customer's own data is not sellable.

```mermaid
flowchart TB
  IN["<b>INPUT</b><br/>Trained candidate"]

  subgraph AXES["SEVEN AXES — ALL MUST PASS"]
    direction TB
    A1["<b>Task metric</b><br/><small>field-level F1 or exact match on the 50 REAL held-out<br/>documents. Partial credit, never binary.</small>"]
    A2["<b>Calibrated judge</b><br/><small>open-weight judge, position-swapped, length-controlled,<br/>calibrated against human labels per vertical.</small>"]
    A3["<b>Behavioural tests</b><br/><small>auto-generated invariance and directional tests per<br/>primitive. Catches bugs aggregate scores hide.</small>"]
    A4["<b>Regression suite</b><br/><small>general ability retained. Fine-tuning degrades it measurably.</small>"]
    A5["<b>Safety suite</b><br/><small>non-negotiable. Benign fine-tuning breaks alignment by default.</small>"]
    A6["<b>Privacy audit</b><br/><small>extraction probe and membership inference.<br/>Weights ship to devices; attacks are white-box.</small>"]
    A7["<b>Calibration / ECE</b><br/><small>confidence must be trustworthy — routing, abstention<br/>and human review all depend on it.</small>"]
  end

  GATE["<b>THE GATE</b><br/>Every axis at or above threshold?<br/><small>Failure never reaches the customer.<br/><b>Re-run in full after quantisation</b> — compression<br/>changes behaviour even when aggregate accuracy holds.</small>"]
  REG["<b>TO REGISTRY</b><br/>Certified cartridge plus scorecard<br/><small>eval certificate, model card, honest failure cases,<br/>side-by-side comparison against the teacher</small>"]
  REP["<b>LLM ROLE 4 — REPAIRER</b><br/>Reads the structured failure report"]
  CONS["<b>THE ARCHITECTURAL CONSTRAINT ON THE REPAIRER</b><br/><b>The repairer may only ever act on EXTERNAL evidence from the eval suite.<br/>Unaided self-reflection is forbidden.</b><br/>Without external feedback, self-correction DEGRADES performance, and apparent gains in earlier<br/>work came from oracle information leaking in. Any design where an agent 'fixes itself' by thinking<br/>harder is building on a result that does not hold."]

  IN --> AXES --> GATE
  GATE -- "pass" --> REG
  GATE -. "fail" .-> REP
  REP -. "rebuild with a mutated plan" .-> IN
  REP --> CONS

  classDef maj fill:#fdf0ea,stroke:#e8562a,stroke-width:2px,color:#20160f
  classDef gate fill:#fdecec,stroke:#c62026,stroke-width:2px,color:#2a1113
  classDef ok fill:#eaf6ef,stroke:#1b7f4b,stroke-width:2px,color:#0e2a1b
  classDef pri fill:#eaf0fb,stroke:#2a5caa,stroke-width:2px,color:#0f1d33
  class A1,A2,A3,A4,A7 pri
  class A5,A6,GATE,CONS gate
  class IN,REP maj
  class REG ok
```

### B-08 · The cartridge, and the registry that makes it cheap

```mermaid
flowchart LR
  subgraph CART["CARTRIDGE ANATOMY — FIVE SLOTS"]
    direction TB
    C1["<b>SLOT 1 — BASE REFERENCE</b><br/><small>pointer only, never a copy. Pinned hash. <b>~0 bytes stored</b></small>"]
    C2["<b>SLOT 2 — ADAPTER STACK</b><br/><small>the trained LoRA delta. <b>The customer's IP. ~30 MB</b></small>"]
    C3["<b>SLOT 3 — TOOL BINDINGS</b><br/><small>scoped permissions for scraper, database, calculator, API. ~KB</small>"]
    C4["<b>SLOT 4 — I/O CONTRACT</b><br/><small>prompt template plus a compiled grammar that makes<br/>schema violation <b>impossible</b>. ~KB</small>"]
    C5["<b>SLOT 5 — RUNTIME POLICY</b><br/><small>offline or hybrid, confidence threshold, abstention, fallback. ~KB</small>"]
  end

  CERT["<b>WHAT MAKES IT TRUSTWORTHY — ATTACHED CERTIFICATION</b><br/><small>eval certificate, model card, datasheet, licence chain, provenance record.<br/><b>Auto-generated from pipeline telemetry — never hand-authored.</b></small>"]

  NAIVE["<b>NAIVE — ONE FULL MODEL EACH</b><br/><b>550 GB</b> = 500 x 1.1 GB<br/><small>~$2.50 per model per month</small>"]
  CA["<b>CONTENT-ADDRESSED — BASE STORED ONCE</b><br/><b>~16 GB</b> = 1.1 GB + (500 x 30 MB)<br/><small>~$0.02 to $0.10 per model per month — <b>34x</b></small>"]
  DEDUP["<b>DEDUP IS NOT ONLY FOR WEIGHTS</b><br/>Datasets, eval suites, grammars and quantised artefacts are all hash-keyed.<br/>An identical spec hash skips the build entirely and serves the cached cartridge at<br/>zero marginal cost — which is why cache-hit rate should be tracked as closely as revenue."]
  LIN["<b>THE OPERATIONAL DIVIDEND</b><br/>Lineage is free once storage is content-addressed. When a defect is found in a base,<br/>the registry can name every cartridge derived from it and rebuild the affected fleet.<br/>Provenance turns homogenisation risk from an unbounded incident into a query."]
  CAV["<b>THE HONEST CAVEAT</b><br/>Base diversity limits blast radius but fragments the shared-base serving economics of A-04.<br/>This is a genuine strategic trade, not a solved problem."]

  CART --> CERT
  CERT --> NAIVE --> CA --> DEDUP --> LIN --> CAV

  classDef maj fill:#fdf0ea,stroke:#e8562a,stroke-width:2px,color:#20160f
  classDef gate fill:#fdecec,stroke:#c62026,stroke-width:2px,color:#2a1113
  classDef ok fill:#eaf6ef,stroke:#1b7f4b,stroke-width:2px,color:#0e2a1b
  classDef com fill:#f1f3f6,stroke:#8a94a6,stroke-width:1.5px,color:#1b2430
  class C2,C3,C4,C5,CA,DEDUP maj
  class NAIVE,CAV gate
  class CERT,LIN ok
  class C1 com
```

### B-09 · Runtime topology — server multi-tenant and on-device

```mermaid
flowchart TB
  subgraph SERVER["SERVER TIER — SINGLE GPU"]
    direction TB
    SV1["<b>Base model resident in VRAM</b><br/><small>one copy, shared</small>"]
    SV2["<b>Paged KV cache</b><br/><small>near-zero fragmentation</small>"]
    SV3["<b>Adapter pool — thousands of cartridges paged on demand</b>"]
    SV4["<b>Batched heterogeneous kernel</b><br/><small>requests using different adapters execute in one batch</small>"]
    SV5["<b>UNIT ECONOMICS</b><br/>~1000 tenants served per A100<br/><small>cost per 1k requests falls with tenancy</small>"]
    SV1 --> SV3 --> SV4 --> SV5
    SV2 --> SV4
  end

  subgraph DEVICE["DEVICE TIER — 4 GB ANDROID TABLET RAM BUDGET"]
    direction TB
    D1["Base model, 1.7B at 4-bit — <b>1.10 GB</b>"]
    D2["KV cache at planned context — <b>0.22 GB</b>"]
    D3["Embedder for the local index — <b>0.09 GB</b>"]
    D4["Grammar state and runtime — <b>0.06 GB</b>"]
    D5["Twenty adapters, resident — <b>0.60 GB</b>"]
    D6["Application and OS headroom — <b>1.40 GB</b>"]
    D7["<b>Total committed — 3.47 GB</b>"]
    D1 --> D2 --> D3 --> D4 --> D5 --> D6 --> D7
  end

  WHY["<b>WHY MANY CARTRIDGES FIT</b><br/>Twenty specialists live in 0.60 GB because they SHARE the one<br/>resident base. Adapter swap is roughly a hundred milliseconds —<br/>fast enough that the user experiences a single assistant,<br/>not a model chooser."]
  UNMEAS["<b>THE NUMBERS ABOVE ARE A DESIGN TARGET, NOT A MEASUREMENT</b><br/>Multi-adapter serving is characterised on datacentre GPUs. The claim that twenty specialists live in<br/>1.5 GB on a mid-range phone with 100 ms swaps is an <b>assumption in this design</b>. MELTing Point shows<br/>the binding constraints on mobile are <b>energy and thermal throttling</b>, not raw speed. Until a physical<br/>device lab measures swap latency, throughput and battery drain, this block is <b>unvalidated</b>."]
  MOAT["<b>THE MOAT IN THE COST</b><br/>The device lab is real capital expenditure and ongoing operations that most competitors will not fund.<br/>That is exactly why measured per-device numbers on the Build Card are defensible ground."]

  D7 --> WHY
  WHY --> UNMEAS --> MOAT

  classDef maj fill:#fdf0ea,stroke:#e8562a,stroke-width:2px,color:#20160f
  classDef gate fill:#fdecec,stroke:#c62026,stroke-width:2px,color:#2a1113
  classDef ok fill:#eaf6ef,stroke:#1b7f4b,stroke-width:2px,color:#0e2a1b
  classDef pri fill:#eaf0fb,stroke:#2a5caa,stroke-width:2px,color:#0f1d33
  classDef com fill:#f1f3f6,stroke:#8a94a6,stroke-width:1.5px,color:#1b2430
  class SV1,SV2,SV4,D1,D3,D4 pri
  class SV3,D5,WHY maj
  class D2,UNMEAS gate
  class SV5,D7,MOAT ok
  class D6 com
```

### B-10 · FABRIC — typed DAG, hybrid routing, and the flywheel

<p align="center">
  <img src="docs/diagrams/flywheel.svg" alt="Animated flywheel: field corrections queue locally, sync when online, feed KTO on unpaired binary signal, drive a rebuild, and a new version is offered only if it wins." width="100%">
</p>

```mermaid
flowchart TB
  subgraph GRAPH["A CUSTOMER GRAPH — THREE NODES"]
    direction LR
    N1["<b>NODE 1</b><br/>req-extractor v3<br/><small>Cartridge · offline</small>"]
    N2["<b>NODE 2</b><br/>urgency-classifier<br/><small>Cartridge · offline</small>"]
    N3["<b>NODE 3</b><br/>doctor-notify<br/><small>Tool · <b>REQUIRES NETWORK</b></small>"]
    N1 --> N2 --> N3
  end

  FLAG["<b>STATIC ANALYSER FLAGS THE GRAPH</b><br/><b>Offline closure BROKEN at node 3.</b><br/>Split into an offline core and an online tail?"]
  TYPE["<b>WHAT THE TYPE SYSTEM BUYS</b><br/>This check is only possible because Fabric is a TYPED DAG rather than free-form agent conversation.<br/>The same analysis bounds RAM and cost, and flags any path where untrusted retrieved content can reach<br/>a privileged tool — the indirect prompt injection surface. <b>Free-form multi-agent chat can prove none of this.</b>"]

  REQ["Incoming request"]
  ROUTER["<b>ROUTER</b><br/>Learned deferral rule<br/><small>NOT a softmax threshold — confidence deferral<br/>is provably weak for specialists</small>"]
  LOCAL["<b>MAJORITY</b><br/>Local cartridge answers<br/><small>zero cost, zero egress</small>"]
  CLOUD["<b>TAIL</b><br/>Escalate to cloud teacher<br/><small>only if the spec permits</small>"]
  OFFL["<b>OFFLINE MODE</b><br/><b>Escalation is architecturally disabled</b>,<br/>not merely discouraged.<br/>A customer who bought offline gets offline."]

  N3 --> FLAG --> TYPE
  REQ --> ROUTER
  ROUTER -- "confident" --> LOCAL
  ROUTER -- "uncertain" --> CLOUD
  CLOUD -.-> OFFL

  classDef maj fill:#fdf0ea,stroke:#e8562a,stroke-width:2px,color:#20160f
  classDef gate fill:#fdecec,stroke:#c62026,stroke-width:2px,color:#2a1113
  classDef ok fill:#eaf6ef,stroke:#1b7f4b,stroke-width:2px,color:#0e2a1b
  classDef pri fill:#eaf0fb,stroke:#2a5caa,stroke-width:2px,color:#0f1d33
  classDef com fill:#f1f3f6,stroke:#8a94a6,stroke-width:1.5px,color:#1b2430
  class N1,N2,N3,ROUTER,TYPE maj
  class FLAG,OFFL gate
  class LOCAL ok
  class CLOUD pri
  class REQ com
```

**Why this loop is the strategy, not a feature:** escalated queries and corrected
fields are the hard cases the local model could not handle. Accumulated across
years and customers, that corpus is the one asset a competitor cannot clone — the
interface is a month of work, the compiler is six months of lead, the per-vertical
eval library is eighteen months, and the failure corpus is permanent.

---

## Part C — comparison

### C-01 · What exists, what is assembly, and what is genuinely new

```mermaid
flowchart LR
  subgraph SOLVED["SOLVED — USE OFF THE SHELF"]
    direction TB
    S["LoRA and QLoRA training<br/>Multi-adapter serving (S-LoRA)<br/>GPTQ, AWQ, k-quant compression<br/>GGUF, CoreML, ONNX packaging<br/>vLLM serving and PagedAttention<br/>Vector indexing and retrieval<br/>Constrained decoding grammars<br/>DPO, ORPO, KTO preference training<br/>MinHash and semantic dedup<br/>Job orchestration and metering"]
  end

  subgraph ASSEMBLY["ASSEMBLY — REAL WORK, NO NOVELTY"]
    direction TB
    A["Wiring the seven subsystems together<br/>Device benchmark matrix and lab operations<br/>Per-vertical judge calibration<br/>Backend packaging and verification<br/>Billing, tiers and cache accounting<br/>Customer-facing scorecard rendering<br/>Tool manifest and permission scoping<br/>Registry service and lineage tracking"]
  end

  subgraph NEW["GENUINELY NEW — THE CONTRIBUTION"]
    direction TB
    N1["<b>The Spec IR itself</b><br/><small>no published schema carries task intent, device constraints,<br/>offline requirements, policy and licence terms in one<br/>type-checkable artefact</small>"]
    N2["<b>Gate-before-GPU validation</b><br/><small>every competitor accepts a config and starts a job.<br/>Type-checking against physics, law and catalogue first<br/>is unclaimed ground.</small>"]
    N3["<b>Offline-closure static analysis</b><br/><small>no agent framework can statically prove a graph runs<br/>without a network. Only a typed DAG can.</small>"]
    N4["<b>Build-outcome meta-learning</b><br/><small>predicting gate pass from a spec embedding turns build<br/>history into a compounding planner asset</small>"]
    N5["<b>Automated licence-chain solving</b><br/><small>composing base, teacher, data and jurisdiction terms per<br/>build is done by lawyers today, not software</small>"]
    N6["<b>The accumulated failure corpus</b><br/><small>years of real customer hard cases. Not clonable at any<br/>price. <b>This is the only permanent moat.</b></small>"]
  end

  READ["<b>THE STRATEGIC READ</b><br/>Moat ranking, honestly: <b>the interface is worth nothing</b> — clonable in a month. The compiler and planner are worth<br/>roughly <b>six months</b> of lead. The per-vertical evaluation library is worth <b>eighteen</b>. The accumulated customer failure<br/>corpus is <b>permanent</b>. Build toward the last one, and treat the first three as necessary costs of entry."]

  SOLVED --> ASSEMBLY --> NEW --> READ

  classDef maj fill:#fdf0ea,stroke:#e8562a,stroke-width:2px,color:#20160f
  classDef gate fill:#fdecec,stroke:#c62026,stroke-width:2px,color:#2a1113
  classDef pri fill:#eaf0fb,stroke:#2a5caa,stroke-width:2px,color:#0f1d33
  classDef com fill:#f1f3f6,stroke:#8a94a6,stroke-width:1.5px,color:#1b2430
  class S pri
  class A com
  class N1,N2,N3,N4,N5,N6 maj
  class READ gate
```

**Only the right-hand column is defensible.** Everything left of it is available
to any competent competitor within months.

### C-02 · End-to-end build — one customer, ten acts

> A diagnostics chain needs offline lab-requisition extraction on 4 GB Android tablets.

```mermaid
flowchart LR
  subgraph P1["PHASE ONE — SPECIFY AND BUILD"]
    direction LR
    ACT1["<b>ACT 01 · CUSTOMER</b><br/>Describe the need<br/><small>'Read lab forms into our system.<br/>Must work when the internet dies.'</small>"]
    ACT2["<b>ACT 02 · FORGE</b><br/>Interview, four questions<br/><small>device auto-probe returns 4 GB Android.<br/>Upload 200 forms. Flag or guess?<br/>Handwriting too?</small>"]
    ACT3["<b>ACT 03 · FORGE</b><br/>Plan in eight seconds<br/><small>offline closure verified. 4 GB caps the<br/>base at 1.7B — the 4B misses by 165 MB.<br/>Customer opted in to two candidates.</small>"]
    ACT4["<b>ACT 04 · BUILD</b><br/>Split, then amplify<br/><small>150 seed, 50 LOCKED. Teacher generates<br/>60k anchored forms with occlusion<br/>and format drift.</small>"]
    ACT5["<b>ACT 05 · BUILD</b><br/>Train two candidates<br/><small>QLoRA on a 1.7B and a 0.6B base, in<br/>parallel, on rented A100s. Forty minutes.<br/><b>Opt-in: faster, not cheaper.</b></small>"]
    ACT1 --> ACT2 --> ACT3 --> ACT4 --> ACT5
  end

  subgraph P2["PHASE TWO — PROVE, CERTIFY AND DEPLOY"]
    direction LR
    ACT6["<b>ACT 06 · PROVE</b><br/>Score both, pick one<br/><small>1.7B scores 0.94 F1 at ~13.8 s.<br/>0.6B scores 0.88 at ~4.5 s — below the<br/>0.93 gate. <b>1.7B wins, and the budget<br/>had to be relaxed to 15 s.</b></small>"]
    ACT7["<b>ACT 07 · PROVE</b><br/>Quantise, then re-score<br/><small>Q4_K_M gives 0.937, above gate.<br/>Flip rate inside bound. Regression and<br/>safety suites pass.</small>"]
    ACT8["<b>ACT 08 · REGISTRY</b><br/>Certify and package<br/><small>model card, eval certificate and licence<br/>chain attached. GGUF built for<br/>Android CPU.</small>"]
    ACT9["<b>ACT 09 · CUSTOMER</b><br/>Read the scorecard<br/><small>93.7% on their OWN 50 forms, twelve<br/>sample extractions, three honest<br/>failure cases.</small>"]
    ACT10["<b>ACT 10 · REGISTRY</b><br/>Deploy and run offline<br/><small>QR install. The cartridge downloads<br/>once, then runs with WiFi switched off.</small>"]
    ACT6 --> ACT7 --> ACT8 --> ACT9 --> ACT10
  end

  EXP["<b>WHAT THE CUSTOMER EXPERIENCES</b><br/>Total wall clock is roughly <b>fifty-two minutes</b>. Marginal cost is<br/><b>twenty to forty dollars</b>, or <b>zero</b> on a spec-hash cache hit.<br/>The customer never receives a model file — they receive a<br/>scorecard and an install link."]
  FIX["<b>TWO NUMBERS IN THIS FLOW WERE WRONG, AND THE PLANNER CAUGHT THEM</b><br/>This diagram once read <b>1.6 s</b> for act 6. The arithmetic says <b>~13.8 s</b> — a 500-token form is<br/>prefill-bound, and prefill is compute-bound at roughly 50 tok/s on a mid-range CPU.<br/>It also implied parallel candidates were free. They are <b>strictly worse in expected cost</b>:<br/>E[cost]₂/E[cost]₁ = 2/(2−p) > 1. They buy wall-clock time, so they are opt-in."]
  NEXT["<b>WHAT HAPPENS NEXT</b><br/>Act eleven, two months later: staff corrections accumulate, sync<br/>when online, and Majestic offers v4 <b>only if it beats v3</b> on a<br/>held-out set that has GROWN with real data.<br/>That is where the compounding begins."]

  ACT5 -- "continue" --> ACT6
  ACT10 --> EXP
  ACT10 --> NEXT
  ACT6 --> FIX

  classDef maj fill:#fdf0ea,stroke:#e8562a,stroke-width:2px,color:#20160f
  classDef gate fill:#fdecec,stroke:#c62026,stroke-width:2px,color:#2a1113
  classDef ok fill:#eaf6ef,stroke:#1b7f4b,stroke-width:2px,color:#0e2a1b
  classDef pri fill:#eaf0fb,stroke:#2a5caa,stroke-width:2px,color:#0f1d33
  classDef com fill:#f1f3f6,stroke:#8a94a6,stroke-width:1.5px,color:#1b2430
  class ACT2,ACT3 maj
  class ACT6,ACT7 gate
  class ACT4,ACT5 pri
  class ACT8,ACT10,EXP ok
  class ACT1,ACT9,NEXT com
  class FIX gate
```

---

## Layout
```
majestic-llm/
├── README.md · pyproject.toml · requirements.txt   (light core deps; heavy deps optional/lazy)
├── configs/        default · models · routing · devices
├── docs/           architecture · forge · planner · device-verification · USAGE
│   │               CONTRIBUTING · conformance · research-gaps
│   └── diagrams/   animated SVGs used above
├── modelrig/                       ← the compiler
│   ├── ir.py                       Spec / Build Plan / Artefact IR (hash-addressed)
│   ├── gates.py                    Gate 1/2/3 — checked before any GPU
│   ├── primitives.py · catalogue.py · licence.py   (the closed set, parts, legal)
│   ├── forge/                      ← the front end
│   │   ├── slots.py                the schema: elicited vs probed vs derived
│   │   ├── parser.py               plain language → a partial Spec IR
│   │   ├── posterior.py            parse_K: absence and ambiguity kept apart
│   │   ├── infogain.py             the Planner as the information-gain oracle
│   │   └── core.py                 the attrition rule; when to stop asking
│   ├── planner/                    enumeration + the predicates (optimisation passes)
│   ├── candidates.py               parallel builds → score both → pick one
│   ├── data_factory.py · proving_ground.py   amplification + the seven axes
│   ├── quantisation.py · grammar.py   calibration, flip rate, constrained decoding
│   ├── cartridge.py · registry.py  content-addressed artefacts + lineage
│   ├── feasibility.py              KV-cache-aware device predictor
│   ├── resources.py                the seven resource dimensions, as pure functions
│   ├── probe.py                    two-point calibration; the verification ladder
│   ├── weights.py                  hash pinning, BF16 merge, merged-vs-separate
│   ├── quantformat.py              SIMD flags select the quantisation format
│   ├── devicedb.py                 the compounding device model
│   ├── certification.py            per-device certification; the on-device eval subset
│   ├── conformance.py              model compatibility + architecture self-check
│   ├── pipeline.py                 end to end, with the repair loop
│   └── buildspec · compiler · planes · classifier · datasets · training_hf
├── majestic/                       ← the compound runtime
│   ├── orchestrator.py · factory.py   request lifecycle + assembly
│   ├── fabric/     graph · analyser · executor   (typed DAG that runs)
│   ├── serving.py                  adapter pool, batching, device RAM budget
│   ├── agent/react.py              ReAct with a constrained tool-call schema
│   ├── router/     rule_router · deferral   (capability routing, learned deferral)
│   ├── flywheel.py                 corrections → KTO signal → rebuild-if-wins
│   ├── core · perception · experts · retrieval · verification · representation
├── api/ · cli/ · demo/ · tests/
```

## Honest bounds
Two load-bearing promises are **unvalidated** and the code says so everywhere it
touches them: on-device swap latency and thermals (GAP-10, every device profile
ships `measured: false`), and multi-generation flywheel quality (GAP-05). Only
the right-hand column of C-01 is defensible; everything else is available to a
competent competitor within months.

## Notes
- Lint targets `ruff`. Where an OS application-control policy blocks unsigned
  binaries, `python -m pyflakes majestic modelrig api cli demo tests` is a
  correctness fallback.
- Optional dependency groups: `.[api]`, `.[ml]`, `.[retrieval]`, `.[compress]`, `.[dev]`.
- The animated diagrams are hand-written SVG in [docs/diagrams/](docs/diagrams/).
  GitHub strips animation from inline Mermaid, so the three flows that benefit
  most from motion are drawn as SVG and the full set is Mermaid.
