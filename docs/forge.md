# FORGE — elicitation as an information problem

*Part 4 of the architecture. Implemented in [`modelrig/forge/`](../modelrig/forge/)
and [`modelrig/resources.py`](../modelrig/resources.py).*

FORGE turns "read our lab forms on the front-desk tablet" into a typed,
hash-addressed Spec IR. It is the module with the least prior art to borrow
from. Slot-filling dialogue is solved for *closed* domains where every slot is
enumerable and every question is cheap. Neither holds here: the schema mixes
human preference, device physics, licence law and sampling statistics in one
artefact, and every question loses customers.

So the question FORGE answers is not "how do I fill this form?" but:

> **Given a free-text description, which questions are worth asking?**

---

## 1. A parse is a posterior

`parse_K` runs the parser *K* times and keeps the distribution rather than the
argmax. That separates two things a point estimate conflates, and they need
opposite fixes:

| | what it means | the fix |
|---|---|---|
| **absence** `is_empty` | no sample produced a value; the description is silent | *supply* it — if the gain justifies a question |
| **ambiguity** `A_i > 0` | samples produced *different* values, confidently | *disambiguate* it — the text admits two readings |

Entropy `H_i` covers the full support including absence, which is exactly why
thresholding on `H` alone is wrong. `A_i` is the disagreement *conditional on a
value being produced* — the half the customer can actually resolve.

> "Must work when the internet dies" is the canonical case. Always offline, or
> merely resilient to the network dropping? The two produce different feasible
> sets. Silently defaulting it is the single most expensive thing FORGE can get
> wrong, so it is surfaced as a question instead of guessed.

**Sampling a deterministic parser.** The rule parser has no temperature, so
resampling it would give *K* identical draws and a spuriously confident
posterior of zero entropy. Instead the *evidence* is resampled: each clause is
dropped with probability `ABLATION_RATE` and the remainder re-parsed. A slot
supported by several redundant cues survives; a slot resting on one ambiguous
phrase flips. That measures ambiguity **in the text**, which is what `A_i` is
supposed to be — and it degrades gracefully to an LLM front end, where
temperature sampling replaces the ablation and nothing downstream changes.

**Constrained decoding.** Every sample is projected onto the slot's declared
domain, so an out-of-domain value is structurally impossible. A hallucinated
value can never win a vote.

---

## 2. The Planner is the information-gain oracle

**This is the novel piece.** Classical active learning ranks questions by the
model's own uncertainty. That criterion fails here, and not subtly: tone can be
maximally ambiguous in an extraction description and is worth exactly zero
questions, because no coordinate of an extraction plan reads it. Entropy would
put it near the top.

MAJESTIC has something active learning normally lacks: a **deterministic
downstream consumer**. So instead of guessing which uncertainties matter, push
each candidate answer through the Planner and look at what comes out.

```
IG(σ_i) = H( Π | σ_i ~ p̂_i )
```

The identity is exact, not a heuristic. The information the answer carries about
the plan is the mutual information `I(A; Π)`; the plan is a *deterministic*
function of the answer once everything else is fixed, so `H(Π | A) = 0` and
`I(A; Π) = H(Π)`. The entropy of the induced distribution over plans **is** the
information gain.

Two consequences fall straight out:

* **All values give the same plan ⇒ IG = 0 ⇒ never ask.** However uncertain the
  slot is. A proof, not a rule of thumb.
* **The values straddle admitted and refused ⇒ the highest stake there is.** That
  answer decides whether the customer gets anything at all, so it is worth the
  whole of `V + κ` rather than the difference between two working models.

### The plan signature

Outcomes are compared at the granularity a customer would notice. Two plans
differing only in LoRA rank are the *same* answer, and counting them apart would
manufacture gain out of implementation detail. But `grammar_ref` and
`eval_suite_ref` stay in: they decide what shape comes out and what the model is
certified against. Drop them and the task primitive — the single highest-gain
slot in the table — measures as worth nothing to ask.

### Scoring counterfactuals

Measuring `IG(σ_i)` means varying slot *i* and watching the plan. If some
*other* unknown refuses every plan on its own, every plan is identical, every
gain is zero, and the interview concludes it should ask nothing at all. The
masking unknown hides the informative one. Two stand-ins fix that, and **neither
ever reaches an emitted spec**:

1. the other slots take their `scoring_default`;
2. an unprobed device is scored under the analytical latency bound, because
   `P_lat` refuses outright on an unmeasured profile.

The second is a finding rather than a workaround: **while the device is
unprobed, no answer the customer can give changes the outcome.** The probe is the
highest-value next action and it is not a question.

So FORGE emits a **probe token instead of a question** (`Interview.probe_request`).
One round trip collapses the whole device group from a posterior to a point mass
and costs zero questions, which is why the attrition budget should never be spent
on a slot that can be measured.

### Marginal, not ceteris paribus

`information_gain` holds the other slots at a stand-in and varies only `σ_i`.
`marginal_information_gain` — §3's estimator, and the default — draws full specs
`s ~ ∏ᵢ pᵢ` from every slot posterior instead, so the number is the reduction in
plan diversity attributable to learning `σ_i` *while everything else stays
uncertain*.

That distinction is not academic. Pinning makes the answer depend on where the
pin was put: with a generous latency budget standing in, input length looks
irrelevant because nothing binds. Marginalising removes the hand-picked constant
from the measurement. It also discriminates better — the pinned estimator
normalises by `log₂(n)` and ties slots at the ceiling, where the marginal one
produces a real ordering:

```
slot                     pinned  marginal
latency_budget_ms         0.579     0.486
expected_input_tokens     0.579     0.375
data_rights               0.579     0.375
seed_data_count           0.579     0.278
device_target             1.000     0.172
budget_ceiling_usd        0.000     0.000     <- zero survives, which is the point
```

---

## 3. Every question costs attrition

```
P(complete | q questions) = e^(−γq)
```

At `γ = 0.05`, 82% of customers finish a four-question interview, 61% finish
ten, and 37% finish twenty. "Just ask everything" is not the conservative choice
— it is the choice that guarantees no spec at all. Those three points *pin* γ, so
the conformance check asserts the curve through them rather than a vague shape.

### The marginal rule

```
IG(σ_i) · stake_i · Λ  >  γ,        Λ = (V + κ) / V
```

The asymmetry in `Λ` is the whole point. Attrition costs a share of the *deal* —
the customer walks and you lose `V`. Bad information costs the deal *and* the
trust damage of shipping a failure, `V + κ`. So the ratio decides, and it couples
the interview to the refusal threshold `θ* = (C + κ)/(V + κ)` through the same
two parameters:

| tier | Λ | θ* at C=$40 | behaviour |
|---|---|---|---|
| experimental | 1.03 | 0.29 | builds readily, asks almost nothing |
| commercial | 1.67 | 0.48 | the default |
| regulated | 11.0 | 0.92 | refuses most things, and asks a lot first |

### Three reasons to ask, and they are different

§8 unions three criteria, and dropping any one of them loses a real case:

| reason | trigger | the case it catches |
|---|---|---|
| `REQUIRED` | `must_ask` and unfilled | `quality_gate` moves no plan but Gate 3 certifies against it |
| `DECISION` | `IG · stake · Λ > γ` | the marginal rule |
| `AMBIGUOUS` | `A_i > A_max` | the description reads two ways |

The third is not a special case of the second. A slot can be maximally ambiguous
and score zero gain, and it still has to be asked — the alternative is silently
picking one reading of a sentence that supports two.

**Two ambiguity signals, unioned.** `A_i` is measured by resampling and counting
disagreement, which is how you detect ambiguity when nothing knows better. But
the parser sometimes *does* know better: it recognises "must work when the
internet dies" as supporting two readings outright. That is stronger evidence
than a vote, not weaker, so `effective_ambiguity` floors a declared ambiguity at
an even split. Without that floor the canonical case scores `A = 0` — every
resample happened to agree — and would never be asked.

**Raise κ and the system both refuses more and asks more.** Both are caution, and
they move together rather than trading off — which is the property you want of a
regulated-domain setting. `check_elicitation_conformance` asserts they never
diverge.

### Termination — four conditions, not one

A spec that type-checks is not finished. `Interview.complete` requires all of:

1. every `must_ask` slot filled,
2. no remaining question clears the marginal rule,
3. no unresolved ambiguity — `A_i ≤ A_max`,
4. **Gate 1 passes**.

The fourth is the one that changes behaviour. A well-formed spec carrying
`data_rights = unknown` is inadmissible, and the old path emitted it silently —
exactly the default the interview exists to prevent. §11 splits the two refusals
by whose fault they are: a Gate 1 failure is remediable by the user, so FORGE
owns explaining it, while Gate 2 is about physics and budget and belongs to the
Planner.

When conditions 3 or 4 fail at the cap, FORGE emits a **partial spec plus an
explicit `unresolved_ambiguities` list**. It never guesses to complete.

The strongest form of condition 2 deserves a name: when every candidate answer to
every remaining uncertainty yields the *same plan*, what is left unknown is
provably irrelevant and stopping is a proof rather than a heuristic
(`Stop.PLAN_INVARIANT`). `MAX_QUESTIONS` is a policy backstop; if it ever fires,
something upstream is miscalibrated.

---

## 4. The slot table is the economy

> A field is elicited only if it encodes a human preference that cannot be
> measured or computed.

| source | cost | policy |
|---|---|---|
| `ELICITED` | attrition | **minimise** |
| `PROBED` | one round trip | prefer |
| `DERIVED` | free | **maximise** |

That ordering is what makes four questions enough. Eleven slots are `must_ask`,
but the probe collapses the device group at once, derivation handles six more,
and a decent description settles most of the rest — so what actually reaches a
human is three to five questions.

**`MUST_ASK` bypasses the marginal rule**, and it has to. `quality.gate_threshold`
moves no plan: the Planner ranks predicted quality against `θ*`, and the
customer's bar is what *Gate 3* certifies against. Its value shows up after the
plan rather than in it, so the marginal rule cannot see it — and a build with no
agreed acceptance bar cannot be certified however good the plan looks.

Two fields are deliberately distinguished on every slot:

* **`domain`** — a *constraint*. Constrained decoding projects onto it.
* **`probe_values`** — representative points for the oracle on a continuous slot.
  Not a constraint. Without them a continuous slot would score zero gain for the
  circular reason that nothing was enumerated from it.

Ordering matters inside `domain` too, since the oracle only tries the first few:
`data.rights` lists `no_training` **second**, because omitting the absorbing
element would make the whole slot read as zero-gain — the one reading where the
answer matters most.

---

## 5. The seven resource dimensions

[`modelrig/resources.py`](../modelrig/resources.py) implements all seven as pure
functions. No LLM, no device, no GPU: it is the layer that can be verified before
anything expensive exists, and the layer everything expensive depends on.

| # | dimension | driver | source |
|---|---|---|---|
| 1 | memory `M` | model size, context | probed |
| 2 | storage `S` | artefact + index | probed |
| 3 | latency `l` | size, **input length** | probed + elicited budget |
| 4 | compute `F_eff` | prefill | probed |
| 5 | thermal `ρ` | sustained use | probed |
| 6 | energy `E` | battery life | probed + derived |
| 7 | network `B` | download, escalation | derived from mode |

**Six of seven are probed. One budget is elicited. None is guessed.**

Two were missing entirely before this module — storage and energy — and both bind
independently of memory. A phone with 4 GB RAM and 8 GB free storage can hold an
artefact it cannot load; a device with ample RAM and a full disk cannot install
one.

### Prefill dominates

Latency is `prefill + decode`, and for document work **prefill leads**:

```
n_in / n_out  >  tok/s_prefill / tok/s_decode        (≈ 4–6)
```

A 1000-token lab requisition producing 80 tokens of JSON gives a ratio of 12.5.
Prefill is the dominant term, not a correction to it — costing decode alone
understates true latency by roughly **3×**, which is exactly how a build passes
Gate 2 and then disappoints in the field.

Three things follow, and all three are implemented:

1. `expected_input_tokens` is a **first-order** Spec IR field, not a detail
   (§17). It is derived from sample documents where possible and asked
   otherwise.
2. A decode-only probe **cannot promise** a prefill-bound workload.
   `assess_latency` downgrades the tier to analytical and says so:
   *"decode was measured but prefill was not, and prefill is 67% of this
   workload."*
3. When prefill dominates and latency fails, `P_lat`'s remedy is to **shorten the
   input**, not to drop a base tier. Prefill is linear in `n_in` and cropping
   costs no quality on the task itself.

### Resources are not independent

Base size and context trade against each other in the same RAM budget; context
caps retrieved chunks; chunks set `n_in`; `n_in` drives prefill; prefill drives
latency and energy. **Retrieving more evidence makes the model slower, not merely
better informed.** `ResourceEnvelope` evaluates all seven at once and names the
binding one, because solving them in sequence gets the wrong answer.

---

## 6. Using it

```python
from modelrig.forge import Interviewer
from modelrig.probe import build_profile

iv = Interviewer(tier=Tier.REGULATED).conduct(
    "Read lab requisition forms on an Android tablet. "
    "Must work when the internet dies. We have 200 real forms.",
    # One round trip settles the whole device group. Not a question.
    device_profile=build_profile(...).to_dict(),
)

iv.questions             # what to ask, ranked by measured gain
iv.spec                  # a hash-addressed Spec IR, or None if it refused to guess
iv.report()              # the full transcript, including every omission
```

```console
$ majestic forge "..." --tier regulated --explain
still to ask (regulated tier, lambda=11.00, P(complete)=55%):
  [        must] Is the example data yours to train on?
                 the answers straddle admitted and refused — this one decides
                 whether there is a model at all

not asked — the planner proved the answer cannot move the plan:
  - budget_ceiling_usd: every candidate answer produces the same plan
```

Every question carries the arithmetic that justified it, and every omission
carries the proof that it was pointless. Both are in `iv.report()`, so the
interview is auditable after the fact — which matters, because the interview is
the only part of the compiler a customer sees.

---

## 7. Did the questions turn out to be worth asking?

§6 closes the loop with a reward that is **observed, never judged**:

```
R = 1[gate passed first attempt] − γ·q − 1[customer rejected]
```

Elicitation research optimises against proxy rewards because nothing downstream
can settle the question. Here the build settles it.

The diagnostic matters more than the reward. `OutcomeLog.diagnose()` reports
first-attempt pass rate **and** mean question count, and refuses to report one
without the other:

```
FORGE OUTCOMES (§6)
  first  10   pass 0.50   q_bar 4.0    reward +0.300
  last   10   pass 0.70   q_bar 14.0   reward -0.000

  ASKING_MORE: pass rate rose +20.0% but mean questions rose +10.0 and mean
  reward did not improve — the attrition cost is being paid and not counted.
  This is not learning
```

A pass rate that rose because the interview got longer is indistinguishable from
real learning on any dashboard that plots pass rate alone. An abandoned interview
scores `−γq` and nothing else: no gate was ever attempted, which is the outcome
the whole budget exists to avoid.

## Open questions

* **γ is a hypothesis.** No published elicitation study measures per-question
  attrition, because their subjects are paid to finish. 0.05 is the specified
  value and the conformance check pins it to three stated points, but those
  points are themselves an assumption, not a measurement.
* **`DELTA_SHARE = 0.25`** — what a plan *change* is worth relative to a plan's
  *existence* — is a stake weight, not a measurement.
* **`(V, κ)` per tier** remains the Planner's open question 2, and now governs
  the interview length as well as the refusal threshold. Both inherit its
  uncertainty.
* **GAP-09.** Eval research optimises correlation with expert raters, not whether
  a pharmacy owner *believes* the number. Customer trust and acceptance are
  first-class product metrics and are still not measured here.
