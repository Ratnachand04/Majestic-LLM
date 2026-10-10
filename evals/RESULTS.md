# Review response — measured results

Four studies, one per review item. Each is a module in this package that
writes its own JSON under `outputs/evals/`; the numbers below are copied from
those files. `data/` and `outputs/` are untracked, so reproduce with:

```
python -m evals.datasets             # fetch and normalise the corpora
python -m evals.multitask            # item 1
python -m evals.adaptation_tradeoffs # item 2
python -m evals.planner_validation   # item 3
python -m evals.integrity_robustness # item 4
```

Throughout: **MEASURED** means taken on this machine or read out of an
artefact's bytes. **DERIVED** means computed from the implemented equations.
The distinction is load-bearing in item 2 and is marked on every row there.

---

## Item 1 — tasks beyond SMS spam

Five corpora, each with a provenance record carrying the download digest, the
post-processing digest, row counts and label histogram.

| corpus | rows | classes | task | relation to the paper |
|---|---|---|---|---|
| `sms` | 5,159 | 2 | SMS spam | the published baseline |
| `youtube_spam` | 1,741 | 2 | comment spam | same task, different customer |
| `sentiment` | 2,979 | 2 | polarity | different task |
| `review_source` | 2,978 | 3 | source attribution | three classes (derived corpus) |
| `trec_questions` | 5,380 | 6 | question answer-type | six classes, different task shape |

### A1 — what FORGE settles from the description alone

| corpus | spec emitted unaided | primitive inferred |
|---|---|---|
| `sms` | no | — (asks `task_primitive`) |
| `youtube_spam` | no | — (asks `task_primitive`) |
| `sentiment` | yes | `classify` |
| `review_source` | no | — (asks `task_primitive`) |
| `trec_questions` | yes | `answer` |

2 of 5 descriptions yielded a spec unaided, and 1 of 5 inferred `classify`
correctly. The three refusals are `must_ask` slots: FORGE declines to guess
the primitive rather than defaulting. `trec_questions` was read as `answer`
from "route … to the right lookup". **A spec FORGE declines to emit is not a
compiler refusal**, which is why this is reported separately from A2.

### A2 — the full gated path, interview answered

900-row stratified seed per corpus, 225 held out, quality gate 0.80, cache
disabled so every task is built and scored rather than looked up.

| corpus | Gate 1 | Gate 2 | proving ground | Gate 3 | held-out accuracy |
|---|---|---|---|---|---|
| `sms` | pass | pass | pass | pass | **0.9689** |
| `youtube_spam` | pass | pass | pass | pass | **0.9244** |
| `sentiment` | pass | pass | refused | — | 0.7067 |
| `review_source` | pass | pass | refused | — | 0.6800 |
| `trec_questions` | pass | pass | refused | — | 0.5911 |

All five cleared Gate 1 and Gate 2, so the planner found a feasible plan for
every task including the six-class one. Three were refused by the proving
ground for missing the requested quality, with nothing offered to the
customer. The plan selected in all five cases was `Qwen/Qwen3-1.7B @ int8 →
gguf, peft=qlora`.

**The executed backend is not that plan.** Every certificate records
`{"backend": "tfidf_centroid", "planned_base_executed": false,
"device_measured": false, "scope": "local classifier demonstration; neural
plan is analytical only"}`. The accuracies above are the centroid backend's,
not a figure for Qwen3-1.7B.

### A3 — was it the gate or the architecture?

| corpus | measured | gate | relaxed gate | outcome |
|---|---|---|---|---|
| `sentiment` | 0.7067 | 0.80 | 0.65 | admitted |
| `review_source` | 0.6800 | 0.80 | 0.60 | admitted |
| `trec_questions` | 0.5911 | 0.80 | 0.50 | admitted |

Each recompiles and is admitted at a gate its measured accuracy clears. The
refusals in A2 are threshold decisions, not capability limits — the
architecture compiles all five tasks; this backend is not accurate enough on
the three harder ones to clear 0.80.

### B — the paper's protocol, unchanged, per task

`modelrig.review.benchmark`: three fixed stratified 80/20 splits,
majority/centroid/3-NN × fp32/int8/int4, full corpora. Means over three splits.

| corpus | cls | majority | centroid fp32 | int8 | int4 | macro-F1 fp32 | int4 flip | warm p50 |
|---|---|---|---|---|---|---|---|---|
| `sms` | 2 | 0.876 | 0.945 | 0.944 | 0.926 | 0.889 | 0.0281 | 0.630 ms |
| `youtube_spam` | 2 | 0.519 | 0.906 | 0.907 | 0.909 | 0.906 | 0.0279 | 0.314 ms |
| `sentiment` | 2 | 0.501 | 0.749 | 0.746 | 0.681 | 0.749 | 0.1636 | 0.361 ms |
| `review_source` | 3 | 0.334 | 0.745 | 0.748 | 0.693 | 0.745 | 0.1714 | 0.351 ms |
| `trec_questions` | 6 | 0.232 | 0.641 | 0.641 | 0.619 | 0.615 | 0.1288 | 0.591 ms |

The SMS headline reproduces. Lift over always-guessing is **larger** on the
new corpora (+0.387 to +0.411) than on SMS (+0.070), because SMS is 87.6% ham
and the majority baseline is already strong there. Absolute accuracy falls as
the task gets harder: 0.945 at two classes on spam, 0.641 at six classes.

int8 is free everywhere (|Δ| ≤ 0.003). int4 is not: flip rates rise from 2.8%
on the spam corpora to 16–17% on sentiment and source attribution, costing up
to 6.8 accuracy points. Split replicates overlap, so the spread across seeds
is not an independent-sample confidence statement.

### C — transfer to data the build never saw

Fit on a held-out split of the source corpus, then test that same fitted model
on the target corpus. Both corpora are binary spam sharing label names, so
only the data's owner changes. This is the only well-posed pair; the other
corpora share no label set.

| source → target | method | in-domain | out-of-domain | Δ accuracy | Δ macro-F1 |
|---|---|---|---|---|---|
| `sms` → `youtube_spam` | centroid | 0.945 | 0.590 | **−0.355** | −0.346 |
| `sms` → `youtube_spam` | 3-NN | 0.971 | 0.538 | **−0.433** | −0.518 |
| `youtube_spam` → `sms` | centroid | 0.922 | 0.544 | **−0.379** | −0.444 |
| `youtube_spam` → `sms` | 3-NN | 0.905 | 0.608 | **−0.297** | −0.404 |

**This is the sharpest negative result in the response.** A model that scores
0.97 in-domain falls to 0.54 on another customer's data for the same task —
near the majority baseline. The compiler generalises across tasks; a *built
cartridge* does not transfer across customers, and nothing in the current
pipeline detects that at build time. Per-customer seed data is not an
onboarding inconvenience, it is load-bearing.

---

## Item 2 — LoRA against QLoRA

Four recorded builds: three LoRA seeds and one QLoRA seed, DistilBERT-base
(66,955,010 parameters) on SMS spam, 1,031 test rows.

`import torch` fails on this host:

```
OSError: [WinError 4551] An Application Control policy has blocked this file.
Error loading "...\torch\lib\shm.dll" or one of its dependencies.
```

So no forward pass could be timed, no resident set sampled, and no power rail
read. Accuracy, sizes, dtypes and training cost are **MEASURED** from the
artefacts and records; latency, memory and energy are **DERIVED**.

### Accuracy — MEASURED

| build | seed | accuracy | macro-F1 | 95% Wilson |
|---|---|---|---|---|
| LoRA | 0 | 0.991271 | 0.979863 | [0.9835, 0.9954] |
| LoRA | 1 | 0.989331 | 0.974878 | [0.9810, 0.9940] |
| LoRA | 2 | 0.985451 | 0.965978 | [0.9761, 0.9912] |
| QLoRA | 0 | 0.991271 | 0.979863 | [0.9835, 0.9954] |

At seed 0 the two methods emitted **the same label on all 1,031 test rows** —
zero discordant pairs, so there is no accuracy difference to test and a paired
test is not applicable. LoRA spans 0.9855–0.9913 over three seeds (mean
0.9887) and the single QLoRA point sits inside that range. **One QLoRA seed
against three LoRA seeds cannot estimate a difference between the methods**;
it can only show containment, which is what is claimed.

The QLoRA deployment's post-quantisation answer-flip rate is 0.00194 — 2 rows
of 1,031 — measured against its own pre-quantisation reference.

### Size — MEASURED, from safetensors headers

Both adapters contain the **identical 739,586 trainable parameters** across 28
tensors of identical shape, with a byte-identical `adapter_config.json`.

| | LoRA | QLoRA |
|---|---|---|
| adapter tensors | 28 (24×F32, 4×BF16) | 28 (28×F32) |
| adapter bytes | 1,784,430 | 2,968,674 |
| merged model | 133,910,020 (BF16, 2.000 B/param) | same |
| **shipped deployment** | **134,867,669** | **73,841,388** |

The QLoRA adapter is 1.67× larger **only because four tensors were serialised
F32 where the LoRA run wrote BF16** — a saving decision, not a property of
QLoRA. Quoting that ratio as a QLoRA cost would be wrong.

The shipped deployment is **45.2% smaller (1.83×)**. Where those bytes go:

| group | bytes | share |
|---|---|---|
| embeddings, left at fp16 | 47,671,296 | **65.4%** |
| transformer weights, NF4 | 21,252,096 | 29.2% |
| quantisation scales and maps | 2,659,368 | 3.6% |
| classifier head, left at fp16 | 1,184,260 | 1.6% |

**NF4 reaches under a third of the shipped bytes.** The config excludes
`classifier`, `pre_classifier` and `score`, and the embedding table — 65% of a
small encoder's bytes — is not quantised at all. So the whole-model rate is
**8.71 bits per weight, not 4**, and the deployment shrinks by 1.8×, not ~4×.
On an encoder this size, quantising the transformer is not where the size is.

### Training cost — MEASURED

| build | runtime | total FLOPs | samples/s |
|---|---|---|---|
| LoRA seed 0 | 34.03 s | 1.98997×10¹⁴ | 363.9 |
| LoRA seed 1 | 37.51 s | 1.96088×10¹⁴ | 330.1 |
| LoRA seed 2 | 37.96 s | 1.97326×10¹⁴ | 326.2 |
| QLoRA seed 0 | **93.22 s** | **1.98997×10¹⁴** | 132.8 |

The seed-0 pair recorded **identical `total_flos` to the unit**, so the two
runs did arithmetically the same work. The **2.74× wall-clock slowdown** is
therefore a clean measure of NF4 dequantisation cost per step, with compute
held fixed.

### The planner's NF4 constant, checked against the artefact — MEASURED

`modelrig.trainer.NF4_BITS_EFFECTIVE = 4.5` underwrites the whole memory
model. Counting the logical weights in the transformer's 2-D matrices from the
unquantised reference (42,467,328) against the bytes the quantised file spends
on those same tensors including scales and maps (23,911,464):

**4.504 measured against 4.5 assumed — 0.10% error.** The width term the
memory predictions rest on is sound.

### Memory, latency, energy — DERIVED, not measured

Parameter counts come from the artefacts, not the rank formula (which is
fitted to a 1.7B Qwen geometry). The activation constant is recomputed for
DistilBERT from `2(d + 3d + 2·d_ffn)/d` = **24.0** at d_ffn/d = 4, rather than
the module default of 19.0, which is the Qwen value.

Training peak at batch 1, 128 tokens, gradient checkpointing on:

| | frozen base | adapter + optimiser | activations | peak |
|---|---|---|---|---|
| LoRA (bf16 base) | 133,910,020 | 11,833,376 | 5,779,071 | **151,522,467** |
| QLoRA (NF4 base) | 37,662,193 | 11,833,376 | 5,779,071 | **55,274,640** |

**2.74× lower peak** under QLoRA; full fine-tuning would need 1,077,059,231
bytes. Inference residency: 148,065,796 vs 87,024,172 bytes, **1.70× lower**.

Latency and energy at the device profile (9.24 GB/s, 3.5 W, thermal derate
0.60), 128 tokens in, **0 tokens out** — a sequence classifier runs one
forward pass and keeps no KV cache, so there is no bandwidth-bound decode term
and quoting one would invent a generation loop the artefact does not have:

| | prefill | derated | energy | requests/charge |
|---|---|---|---|---|
| LoRA | 99.7 ms | 166.1 ms | 0.581 J | 83,449 |
| QLoRA | 99.7 ms | 166.1 ms | 0.581 J | 83,449 |

**Identical, and that is the finding.** Prefill is compute-bound in the
parameter count, `t = 2·P·n_in/F_eff`, which quantisation does not reduce. On
this profile quantisation buys footprint, not speed. Energy is
`P_draw × predicted latency` — an arithmetic restatement of the latency
prediction, not an independent measurement.

The 2.74× memory ratio and the 2.74× training slowdown are **independent
quantities that coincide to two decimals**: the first from 16 against 4.5 bits
per frozen weight, the second from measured wall-clock. Neither is derived
from the other.

### Measured quantisation evidence that needs no framework

Not a LoRA/QLoRA result and not offered as a substitute for one — it is the
one quantisation trade-off on this machine that is measured rather than
predicted. On the centroid backend across the five item-1 tasks, int8 costs
|Δaccuracy| ≤ 0.003 everywhere; int4 ranges from −0.0683 to +0.0029 with flip
rates of 2.8% to 17.1%.

### What would close the gap

A host without the Application Control policy, or a CPU-only torch build that
does not load `shm.dll`; both methods at the same three seeds so the accuracy
comparison has more than one QLoRA point; a wall-power or battery-rail reading
for the energy row; and an int4 deployment that also quantises the embedding
table, which is where the measured bytes actually are.

---

## Item 3 — planner feasibility and refusal decisions

73 cases over the composed **Gate 1 → Planner** admission path, because that is
what decides admission; the planner alone is not the surface a spec meets.
Expectations were written from the design documents before running the code.

**100% agreement (73/73), zero disagreements.** 30 refusals, split by layer
`{gate1: 15, planner: 15}`. Per-predicate precision 1.0 for `P_cost` (3 firings),
`P_lat` (6), `P_ram` (6), `P_seed` (6) and Gate 1 (15).

Aggregate refusal precision is **1.0 but inflatable** — padding the corpus with
memory-infeasible specs raises it without evidence, because `P_tok`, `P_lic`,
`P_ram`, `P_off` and `P_storage` are sound by construction and no experiment
can inform them. It is reported only to be discounted. The meaningful stratum
is the one Part 2 §16 names: specs that clear Gate 1 and are decided by a
**soft** predicate — a prediction a measurement could overturn. That stratum is
20 cases, **100% agreement, 10 refusals, precision 1.0, recall 1.0**, all
decided by the planner.

**Headline finding — `P_seed` is unreachable for six of eight primitives.**
Gate 1's engineering seed floor is at or above the planner's coverage floor
`n_min = ⌈−ln(η)/μ_min⌉` everywhere except two primitives, so Gate 1 refuses
first and the planner's seed predicate never binds:

| primitive | Gate 1 floor | coverage floor | `P_seed` window |
|---|---|---|---|
| `extract` | 120 | 100 | none |
| `classify` | 80 | 60 | none |
| `summarise` | 150 | 120 | none |
| `rewrite` | 150 | 120 | none |
| `generate` | 200 | 200 | none |
| `answer` | 150 | 100 | none |
| `route` | 80 | 150 | **[80, 149]** |
| `toolcall` | 200 | 250 | **[200, 249]** |

Any evaluation of `P_seed` that does not target `route` or `toolcall` measures
Gate 1 and reports it as the planner.

---

## Item 4 — integrity checks and Fabric validation

### Malformed artefacts — 15/15 as specified

Against a fixture with a real manifest: untouched (accept), single-byte flip,
truncated file, zero-length file, deleted file, extra file, renamed file,
manifest entry dropped, manifest emptied, manifest missing, manifest not JSON,
manifest path traversal (`../../escaped.bin`), manifest absolute path
(`/etc/passwd`), manifest zero hash — all rejected — and **consistent forgery,
expected to ACCEPT and accepted**.

### Unsafe data flow — 11/11 as specified

Beyond the existing chain sweep: chain unconfirmed/confirmed,
**branch-one-arm-confirmed** (two arms into one privileged sink, only one
passing a human), branch-both-arms-confirmed, two untrusted sources, trusted
only, untrusted-to-unprivileged, network in an offline graph, cycle,
16-hop laundering chain, and **diamond-shortcut** (a confirmed long path
alongside an unconfirmed shortcut). The branch and diamond cases are the ones a
first-path-only analysis would wave through; both were caught.

### These are not a security sandbox — three measured demonstrations

1. **The manifest does not authenticate.** The consistent-forgery case passes
   by design: rewrite the payload and the manifest together and the check
   succeeds. `verify_manifest` establishes integrity *relative to a manifest*,
   which is a statement about accidental corruption and transport damage, not
   about an adversary who can write the artefact directory.
2. **Fabric validates declarations, not implementations.** A node declaring
   `taint_role="clear"` and no network use is accepted on that declaration.
   Whether its handler honours it is outside what analysing a graph can see.
3. **Coverage is of planted templates and declared effects only.** The corpus
   establishes that the checks fire on constructed violations. It does not
   quantify behaviour on real user graphs, and it is not a containment
   boundary: nothing here confines code at runtime, restricts syscalls, or
   survives an attacker with write access to the artefact.
