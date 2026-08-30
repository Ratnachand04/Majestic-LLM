# The Data Factory — 150 documents into a training set

*Part 8 of the architecture. Implemented in [`modelrig/augment.py`](../modelrig/augment.py),
[`modelrig/diversity.py`](../modelrig/diversity.py) and
[`modelrig/data_factory.py`](../modelrig/data_factory.py).*

This subsystem exists to solve one pain point — *"I have 40 examples, not
40,000"* — the single most common reason a build fails.

---

## The reframe: three operations, not one

Every earlier document called this "synthetic data generation". That conflates
three operations with **radically different risk**, and the conflation was hiding
the fact that the most common primitive barely needs synthesis at all.

| operation | form | label provenance | collapse risk | teacher cost |
|---|---|---|---|---|
| **augmentation** | (d, y) → (T(d), y) | **real, by construction** | very low | **≈ 0** |
| **backtranslation** | y → (G(y), y) | **real output**, generated input | medium | moderate |
| **synthesis** | ∅ → (G₁, G₂) | both generated | **high** | high |

**Priority: augmentation → backtranslation → synthesis.**

### Why this matters more than it looks

For `extract`, the customer usually already has (document, label) pairs **and does
not know it**. A clinic that has processed lab requisitions manually for years has
the extracted fields sitting in its operational database. The document is the
input; the database record is the label.

So the task is not to invent pairs. It is to generate realistic variations of real
inputs *whose labels are known to be unchanged* — occlusion, format drift, scanner
noise, layout shift. That is augmentation, and its label is real by construction.

**This explains the seed-floor differences as a mechanism**: the floors differ
because label provenance differs, not because "some tasks need more data".

**Economic consequence:** for augmentation-dominated primitives the Data Factory
costs approximately nothing. The dollar figures quoted for generation apply to
synthesis-dominated primitives only.

---

## The operator monoid

An operator is admissible iff it is label-preserving: `y(T(d)) = y(d)`. Admissible
operators are closed under composition, so a small set gives combinatorial
coverage — but **depth is capped at 3**, because a document blurred, rotated,
occluded and re-encoded five times is not a document anyone will ever scan.

**Realism is not seed-proximity.** The obvious check — "is T(d) near the seed
distribution?" — is *wrong*, because the point of hard-case generation is to
produce inputs the seed set does not contain. Realism comes from parameterising
operators with domain knowledge (blur radius from the scanner's MTF, occlusion
shapes from real staple marks), plus a human spot-check of ~100 samples per build.
That is the one place in the pipeline where human judgement is not replaceable,
and it takes ten minutes.

> Field reordering generates a genuinely hard case for free, and it exercises
> exactly the invariance the behavioural tests check. **Augmentation operators and
> behavioural test generators are the same objects seen from two directions.**

---

## Pseudonymisation: privacy and amplification are one operation

For extraction there is a hard tension: **the PII *is* the label.** You cannot
redact patient identifiers from a patient-identifier extractor's training data —
redaction destroys the task.

Consistent pseudonymisation resolves it, and does two jobs at once:

- **label-preserving under relabelling** — only surface tokens change;
- **an augmentation operator** — one real document yields *m* correctly labelled
  variants.

It also directly lowers verbatim-extraction rate, because no single real identity
appears often enough to be memorised. **This should be the default for every
extraction build carrying PII, not an option.**

---

## Effective modes, not example count

Sixty thousand near-identical rows is one example repeated.

```
N_eff(S) = exp(H(S))        the Hill number of order 1
```

| | N_eff |
|---|---|
| uniform over K clusters | K |
| everything in one cluster | 1 |
| 60k rows collapsed to 100 modes | ~100 |

The **Vendi score** — exponential of the von Neumann entropy of the similarity
matrix — avoids the arbitrary clustering step and is preferred when the set admits
an n×n matrix.

### The discount factor is the diversity ratio itself

```
n_eff = n_r + η₀ · N_eff(S)
```

| set | n_s | N_eff | contribution at η₀ = 0.3 |
|---|---|---|---|
| diverse | 60,000 | 8,000 | **2,400** |
| collapsed | 60,000 | 100 | **30** |

Same row count, eighty times the contribution. The old parametric form needed two
invented constants; this needs one, and `ρ₀` — the more speculative of the pair —
**disappears entirely**, because `N_eff` is measured per build rather than assumed
globally.

---

## Saturation stopping

`synthetic_target` was a number someone guessed. Generation now stops when
diversity saturates:

```
stop ⟺ (N_eff(b) − N_eff(b−1)) / N_eff(b) < ε
```

**Wall clock is the binding constraint**, not dollars — generation dominates build
latency. Saturating at 20k instead of a planned 60k cuts generation from ~3.3
hours to ~1.1 on a rented A100. And the count becomes a *reported measurement*
rather than an input.

This subsumes the separate diversity monitor: collapse in this regime is simply
saturation at a low `N_eff`, which then fails the seed predicate.

---

## The citation was at the wrong subsystem

Every document in this series justified these guardrails by appeal to model
collapse (the Curse of Recursion, 2305.17493). **That citation was misplaced, and
the misplacement was distorting the design.**

The paper describes **recursive** training: generation *k+1* trains on generation
*k*'s outputs, iterated, with tails progressively lost. Single-build amplification
here is **not recursive** — every call is anchored on the customer's real seeds
and no generated example is ever fed back as a source. The mechanism that produces
collapse is structurally absent.

| | collapse (recursive) | saturation (anchored) |
|---|---|---|
| cause | tails lost, compounding | bounded space fully covered |
| more data is | **actively harmful** | **useless**, not harmful |
| signature | drift across generations | flat N_eff vs n_s |
| remedy | inject real data | **stop generating** |

**The distinction is not academic.** Under a collapse model, over-generating
damages the build and the guardrail must abort. Under saturation it merely wastes
money, and the right response is to stop. The design already did the right thing;
the stated reason was wrong, and the reason determines what happens when a
guardrail fires.

**The citation belongs to the flywheel** — where training generation *g+1* on
generation *g*'s outputs *would* be the recursive structure, which is why I-03
forbids it. It has been moved there.

---

## The amplification ceiling

Combining the effective-sample formula with saturation:

```
n_eff_max = n_r (1 + κ)
```

**Amplification multiplies effective data by at most (1 + κ), however much is
generated.** Fifty seeds cannot become five thousand examples' worth of
information.

This turns the seed floor from a table lookup into a derivation, and the refusal
into an explanation:

> *"40 real examples amplify to at most 88 effective examples for this task, and
> the bar needs 200. You need about 91 real examples — **more seeds, not more
> generation**."*

κ is primitive-dependent for exactly the reason above: augmentation-dominated
primitives have high κ (labels are real, transforms are many); synthesis-dominated
ones have low κ.

---

## Coverage, measured without leaking

`N_eff` measures *internal* diversity. It says nothing about whether the set covers
**reality** — a generator can produce 8,000 diverse modes all unlike the deployment
distribution.

**The leakage trap:** measuring coverage against the *held-out* set and
regenerating until it improves is fitting to the holdout. It would silently
invalidate every number the Proving Ground produces.

The fix is K-fold **inside the seed set**. Every seed serves both purposes, the
holdout is never touched, and no third split is needed — which matters, because at
n = 200 a three-way split leaves nothing usable anywhere.

---

## Deduplication is a privacy control

Memorisation probability rises sharply with duplication count, and Majestic
**ships weights to devices** — so the attacker has white-box access, the strongest
possible setting.

**Augmentation raises the stakes**: twenty transforms of one seed are twenty
near-copies of that seed's content. Naive augmentation therefore *increases* `k`
by design and can undo the privacy benefit of a small training set.

Three requirements, all implemented: dedup runs **after** augmentation, per-source
multiplicity is capped using the `source_id` provenance that contamination
checking already requires, and `max_i k_i` is reported as a prior for the privacy
audit. A build at `k_max = 40` should expect a worse extraction rate than one at
`k_max = 5`, and knowing that in advance turns a surprise into a prediction.

**Contamination must be checked at the source-document level.** `T(d)` for a seed
`d` is fine, but an augmented variant of a *held-out* document leaks even though
the two examples differ byte for byte. Only provenance catches that.

---

## Cost

Self-hosted, batched vLLM on a rented A100 at ~$2/hour with a 32B teacher at
4-bit producing ~1500 tok/s:

```
$2 / (1500 × 3600) ≈ $0.4 per million generated tokens
```

The 90 micro-USD/1k figure quoted elsewhere is an **API price** and is wrong for
self-hosted generation by roughly two orders of magnitude. `KAPPA_GEN_PER_TOKEN`
has been corrected to 0.4.

| scenario | tokens | cost | wall clock |
|---|---|---|---|
| 60k synthesised @ 300 tok | 18 M | ~$7 | 3.3 h |
| 20k after saturation stop | 6 M | ~$2.4 | 1.1 h |
| augmentation-dominated | ~0 | **~$0** | minutes |

---

## Novelty, honestly

| claim | grade |
|---|---|
| Self-Instruct / Evol-Instruct / backtranslation | none — used as-is |
| **Label-provenance ordering** | moderate — a reframe that changes the cost model |
| **Diversity ratio as the discount factor** | moderate — removes an invented constant |
| **Saturation stopping** | moderate — replaces a guessed target |
| **Pseudonymisation as joint privacy + augmentation** | moderate — obvious once stated, absent from practice |
| **Measured collapse floors (η₀, κ)** | **strong** — R4; the numbers do not exist in the literature |

Grade B+, carried by the measured floors.

## Still open

- **η₀ and κ(p) are hypotheses.** R4 measures them: fix `n_r`, sweep `n_s` to
  saturation, read off the ceiling. That is a cleaner experiment than hunting for
  a collapse threshold which, per §19, does not exist in this regime.
- `data.synthetic_target` should become **advisory** in the Spec IR, with the
  actual count determined by saturation and reported. Not changed here — it is a
  schema change with customer-facing consequences.
- **Forge should ask whether the customer has an operational record of past
  outputs.** For extraction that single question converts a synthesis problem into
  an augmentation problem and collapses both the cost and the risk of the build.
  This is the most immediately valuable item in Part 8 and it belongs in the slot
  table, not here.
