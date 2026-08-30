# Weight compilation and device verification

Two questions, answered separately because they are different kinds of hard.

> **How is the SLM created?** Download a frozen `W`, learn a rank-`r` correction,
> merge, quantise, package. Four tensor operations. Nothing is invented.
>
> **How is device specificity verified?** Not by prediction. **The device
> certifies the model, rather than the vendor certifying the device.**

```bash
python -m cli.main probe --device-id sm-a536b-8f3a \
    --small-mb 200 --large-mb 400 --small-toks 42 --large-toks 22 \
    --sustained-toks 13.2 --ram-free-mb 2600 --simd neon,dotprod,i8mm \
    --out profile.json
```

## Part 1 — how the cartridge is physically produced

```
W ──train──▶ ΔW = (α/r)·B·A ──merge──▶ W' = W + ΔW ──quantise──▶ Q(W') ──▶ cartridge
└ downloaded, frozen        └ ~30 MB, learned
```

Stating it that plainly matters: the mystification of this step is where the
"model that builds models" misconception lives. **No architecture is generated,
no layers are designed.**

### The two quantisations that get confused

They are separate types in the code ([`weights.py`](../modelrig/weights.py)),
because treating them as one step produces the worst bug available here.

| | Training quantisation | Deployment quantisation |
|---|---|---|
| what | NF4 (QLoRA) | k-quant / AWQ / GPTQ |
| why | fit the frozen base in VRAM | fit the shipped artefact on the device |
| applied to | the frozen base only | the **merged** model |
| calibrated on | nothing — data-free | the customer's task distribution |
| reaches the device | **no** | yes |

`QuantSetting` refuses to be constructed with the wrong combination: a training
quantisation cannot be marked calibrated, and a deployment quantisation cannot be
uncalibrated.

### Merge in BF16, never into the 4-bit base

```
WRONG:   NF4(W) + ΔW → quantise → device      ← two errors compounded, silently
RIGHT:   dequantise NF4(W) → BF16
         BF16(W) + ΔW = W'                    ← merge in high precision
         quantise(W', calibration)             ← one quantisation, task-calibrated
```

`MergePlan` raises if `merge_precision != "bf16"`. The failure it prevents is
silent — every intermediate step reports success while most of the fine-tune's
gains evaporate.

### Merged or separate

One cartridge on the device → **merge** (better runtime support, nothing to share
with). Multiple cartridges sharing a base → **keep separate**, because N merged
models cost `N × full` while N adapters cost `base + N × 30 MB`. That is the
entire multi-cartridge memory argument. `choose_merge_strategy()` encodes it,
including the cases where sharing does not help.

### Pinning

Base weights are pinned by **content hash, never by name or tag**. Checkpoints on
public hubs get updated in place, renamed and removed; if a result says
"Qwen3-1.7B" and that checkpoint changes, the work becomes unreproducible with no
error anywhere. `WeightMirror.pin()` refuses to re-pin a name at a new digest —
which is exactly the event pinning exists to catch.

### The format is a device choice, not just a bit width

On ARM cores with `dotprod` or `i8mm`, some k-quant variants run substantially
faster than others **at identical bit-width**, because their block structure maps
onto the available SIMD instructions.

> ⚠ §13 is explicit that this mapping is **empirical folklore, not a published
> table**. [`quantformat.py`](../modelrig/quantformat.py) encodes it as a
> falsifiable hypothesis: every selection carries `evidence="hypothesis"`, and a
> measured sweep overrides it entirely. The probe already runs two models, so
> sweeping formats is nearly free — and produces a genuinely useful public
> artefact.

## Part 2 — verifying hardware you do not own

The honest problem: **there are hundreds of Android SoCs and you will own two or
three.** Every certification you issue is for a device you have never touched.

| approach | cost | memory | speed | sound? |
|---|---|---|---|---|
| analytical (roofline) | free | yes | bound only | **refusal only** |
| emulated (cgroups) | cheap | **yes** | no | partially |
| reference device | one/tier | yes | that device | interpolation unsound |
| **probe the actual device** | one round trip | **yes** | **yes** | **sound** |

The fourth is the answer, and nobody offers it because it requires the customer's
device to participate in the build. Majestic can require it, because the device is
where the model is going to live anyway.

### The three-tier ladder

Every claim carries a tier, and nothing is presented as measured when it is
predicted.

**Tier 1 — analytical.** `tok/s ≲ BW_eff / |Q(W')|`. Real throughput is always at
or below this, since throttling, framework overhead and contention only subtract.
Hence the asymmetry: *roofline says infeasible → refuse (sound); roofline says
feasible → cannot promise (unsound).*

**Tier 2 — emulated.** Run the packaged artefact under `systemd-run
-p MemoryMax=…`. **Catches OOM, never slowness** — memory ceilings are enforceable
with cgroups, memory *bandwidth* is not. `EmulationPlan.VERIFIES` and
`.CANNOT_VERIFY` state both halves.

**Tier 3 — measured.** The only tier that permits a promise.

### The two-point calibration

Decode at batch 1 is memory-bandwidth bound, but per-token overhead is not zero,
so the honest model is affine:

$$t_{\text{token}}(S) = \frac{S}{BW_{\text{eff}}} + c_{\text{overhead}}$$

Two unknowns, so two probe points solve it exactly:

$$BW_{\text{eff}} = \frac{S_2 - S_1}{t_2 - t_1}, \qquad
c_{\text{overhead}} = t_1 - \frac{S_1}{BW_{\text{eff}}}$$

**Two measurements on a 200 MB and a 400 MB model calibrate latency prediction
for every base in the catalogue on that device.** You never ship the real model to
find out how fast the real model will be.

Worked, and reproduced by `cli.main probe`:

```
200 MB → 42 tok/s,  400 MB → 22 tok/s
  BW_eff     = 9.24 GB/s
  overhead   = 2.16 ms/token
  1120 MB cartridge → 8.1 tok/s burst
```

A larger model measuring *faster* raises `ProbeError`: decode is bandwidth-bound,
so that is a broken benchmark — thermal state, page cache, contention — not a
device that beats the roofline. It must not be silently fitted.

### Thermal derate

A 30-second benchmark will not reveal throttling that appears at four minutes:

$$\rho_{\text{thermal}} = \frac{\text{tok/s at }180\text{s}}{\text{tok/s at }30\text{s}}$$

Routinely **0.5–0.8** on passively cooled phones. **All latency promises use the
sustained figure** — `DeviceProfile.tokens_per_s()` applies it by default. Ignoring
it is how a demo that works becomes a deployment that does not.

### Extrapolation limits

The affine model holds within roughly one decade. Probing at 200/400 MB and
predicting a 4 GB model is a 10× reach, and beyond that **even a measured profile
reverts to a bound** — `assess_latency()` downgrades the tier and says why.

## Probe before plan

The reordering is what converts device specificity from a guess into a
constraint:

```
spec (device_class = "android_mid")      ← a guess
   │
   ▼  PROBE — 3-5 min on the actual device, once
DeviceProfile (measured)                 ← replaces the guess
   │
   ▼  PLANNER — predicates now run on MEASURED facts
        P_ram  uses ram_free_mb (already net of the app)
        P_lat  uses bw_eff × thermal_derate
        q      selected from simd flags
   │
   ├──▶ REFUSE, with real numbers
   ▼
build → merge → quantise → package → DEVICE VERIFY
   │
   ▼
certified for THIS device, or not certified at all
```

**`P_lat` moves from Tier 1 to Tier 3 and may now promise.** The `devices.yaml`
deadlock — every tier marked `unmeasured` — is resolved not by buying twenty
phones but by making the customer's own device the measurement instrument.

Without a probe, the fallback is honest and clearly worse: the reference device
for that tier, `latency_source: interpolated`, and a Build Card that says
**unvalidated for your specific hardware** — which is itself an argument for
running the probe.

## Device facts → plan choices

Six coordinates, not one:

| device fact | determines |
|---|---|
| `ram_free_mb` | max base size |
| `bw_eff × thermal_derate` | max base size under latency |
| `simd` flags | **quantisation format** |
| `accelerator` | **runtime container** (GGUF / CoreML / ONNX / ExecuTorch) |
| `ram_free − weights` | **context budget** |
| `storage_free_mb` | merged vs separate |

The **context budget** is the one most often missed. KV cache is linear in context
length, so on a tight device the plan does not merely pick a smaller model — it
**caps the context**, which caps how many retrieved chunks the cartridge may use,
which feeds back into the retrieval design. `context_budget()` derives it; it is
never elicited. **Device constraints propagate upward into task design, not only
downward into model size.**

Two of these are separate questions that get collapsed. The **quantiser** comes
from the SIMD flags; the **container** comes from the accelerator. Selecting the
container from the device's *name* — `android_*` implies GGUF — is exactly the
guess the probe exists to replace, since two units in the same class can differ
in whether an NPU delegate is usable at all. `target_formats_for()` maps it, and
a measured accelerator overrides the class prior:

| accelerator | containers, best-supported first | offline |
|---|---|---|
| `cpu` | `gguf`, `onnx`, `executorch` | all three |
| `gpu` | `onnx`, `gguf`, `vllm` | `vllm` dropped — it needs a server |
| `npu` | `executorch`, `coreml`, `onnx` | all three |
| `ane` | `coreml`, `executorch` | both |

So an android tablet whose probe reports an NPU is no longer offered GGUF at all,
where the device-name prior would have ranked it first. An accelerator nobody has
mapped falls back to the CPU containers — conservative rather than correct, and
logged as such.


## The compounding device model

Every probe returns a `(SoC, bandwidth, overhead, thermal_derate)` tuple.
Accumulated across customers, [`devicedb.py`](../modelrig/devicedb.py) fits a
device-class predictor **with no device lab at all**.

Three deliberate choices:

- **Median bandwidth, minimum derate.** One throttled outlier should not drag the
  bandwidth estimate, but a thermal promise must hold for the *worst* unit seen.
- **Only measured profiles enter the database.** Admitting an interpolated one
  would let the model train on its own output.
- **An interpolated profile never reaches the measured tier**, however many
  observations back it. It is not a measurement of *this* unit.

A new customer on a previously-seen SoC can skip the probe — cutting build latency
at no cost to soundness, provided the tier is honestly marked. *A measurement made
for one customer improves planning for every subsequent one:* the strongest moat
available here, because it requires deployments rather than cleverness.

`DeviceDB.coverage()` also flags **high dispersion** — when units sharing an SoC
name disagree, the label is not predicting the thing it is being asked to predict.

## Schema changes (§14)

| change | field |
|---|---|
| **add** | `device_profile` — the measured profile, nullable |
| **add** | `profile_source` — `probe` \| `device_lab` \| `reference_device` \| `interpolated` \| `assumed` |
| **add** | `context_budget` — **derived**, not elicited |
| **relax** | `latency_source` may now be `measured` when a probe exists |
| **demote** | `devices.yaml` entries become **priors for un-probed devices**, not the source of truth |

`profile_source` is the single field that decides whether `P_lat` may promise or
only refuse.

## Failure modes

- **Probe/production mismatch.** The probe runs idle; production shares RAM with
  other apps. `headroom_factor` (0.85) discounts it, and re-probing periodically
  is the real answer.
- **Extrapolation beyond a decade** is unreliable — reported, and it downgrades
  the tier.
- **Thermal derate is workload-dependent.** A 180 s sustained benchmark does not
  characterise 10-second bursts all day. It is a conservative single number
  standing in for a curve.
- **Format choice is under-determined.** Currently folklore; the probe should
  measure it.
- **Merged and separate are different artefacts** with different numerics.
  Certify whichever one actually ships.

## What is certified

Implemented in [`certification.py`](../modelrig/certification.py); the array lives
on the cartridge manifest as `measured_performance`.

```
DeviceCertificate
  device_id, source          customer_device | device_lab | emulated | interpolated
  artefact_kind              merged | separate
  tokens_per_s               burst
  tokens_per_s_sustained     after the thermal derate — what promises use
  peak_ram_mb
  eval_subset_score          ran ON the device
  eval_subset_size
  output_parity              against the workstation's outputs
  energy_per_1k_tokens_j
```

Three properties, and each is the difference between a certificate and a
decoration.

**A build certificate is not a device certificate.** `Cartridge.certified` says
the build was evaluated and licensed. `Cartridge.certified_for(device_id)` says it
has been proven to run on *this* silicon. They are different questions with
different answers, and the manifest keeps them apart.

**Certification is per device.** The same cartridge may be certified for one SoC
and uncertified for another. A device absent from the array is reported as
unverified — never as passing, never as failing, and never silently defaulted to
the nearest device that happens to be present. A ledger that answers "yes" for a
device it never saw is worse than no ledger at all.

**Certification is per artefact kind.** A merged model and a base-plus-adapter
pair are different artefacts with different numerics, so a certificate earned by
one says nothing about the other:

```
UNVERIFIED on sm-a536b for the separate artefact — the merged artefact was
certified there, and the two have different numerics
```

### The eval subset is what closes the loop

Throughput alone certifies nothing, so a run that only measured tokens per second
is **refused a certificate** and reported as `THROUGHPUT ONLY`. Compilation and
format conversion introduce numerical differences: an artefact that quantised
cleanly on a workstation can produce different outputs through a different runtime
on a different SoC — same weights, same grammar, different answers.

`output_parity` catches it. Aggregate score can hold steady while individual
answers change, which is the same failure the answer-flip rate catches after
quantisation, reappearing at the runtime boundary. Below 95% parity the eval
certificate does not transfer, and the run is refused with that reason. Twenty
held-out examples on the actual device is the floor, and it is the only check that
closes the loop between "we built it" and "it works there".

A run is also refused when the on-device score falls more than 0.03 below the
workstation's (*the artefact degraded in transit, not in training*), when peak RAM
exceeds what the device makes available in production, or when the profile behind
it was never measured. A benchmark whose sustained rate *exceeds* its burst rate
is rejected outright rather than fitted — that is a broken run, not a device that
defies the roofline, and fitting it would launder the error into a promise.

### One discrepancy

§11's illustrative record pairs `tokens_per_s_sustained: 5.4` with
`energy_per_1k_tokens_j: 41.2`. Those cannot both hold: 1000 tokens at 5.4 tok/s
takes 185 seconds, so 41.2 J implies a package draw of **0.22 W** — roughly 16×
below the 3.5 W Part 4 §13.6 establishes for sustained mobile inference. The same
pair at 3.5 W is 648 J.

The implementation computes `E = P·t` from the *measured* draw, which is
consistent with §13.6 and with the physics. The spec's figure is treated as a typo
in an example rather than a constant to reproduce.
