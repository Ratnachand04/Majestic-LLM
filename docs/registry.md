# The Registry — storage, caching, and the two things that are unclaimed

*Part 5 of the architecture. Implemented in [`modelrig/registry.py`](../modelrig/registry.py),
[`cachekey.py`](../modelrig/cachekey.py), [`economics.py`](../modelrig/economics.py)
and [`corpus.py`](../modelrig/corpus.py).*

## Honest framing first

The Registry is **Grade C** for novelty and that grade is right. Content-addressed
storage is textbook, refcounting is textbook, model cards are a published format.
Two things here are genuinely unclaimed — the licence solver and the
cache-privacy constraint — and neither is a paper on its own.

What the Registry *is* is the subsystem whose arithmetic determines **gross
margin**. Every equation below maps to a line in a P&L, which is why it gets the
same rigour as the Planner despite never being publishable.

---

## The correction: the cache as specified leaks models between customers

This is a data-protection bug, not a refinement, and it arrives through an
optimisation that looks free.

The cache needs a key that is *not* the full spec hash — otherwise two requests
identical in everything that matters, differing only in owner or budget ceiling,
key differently and rebuild from scratch. So you hash a semantic subset. The
obvious next step is to drop `data.seed_ref` too: it names *data*, not
*requirements*.

Do that and two customers with identical requirements and different confidential
corpora collide — and the second is served **a model trained on the first
customer's documents**. For a system whose central promise is that data never
leaves the customer's control, that is the worst failure available.

**The invariant:**

```
data.seed_ref ∈ h_cache    always
```

`h_cache` raises rather than returning a key if a required field ever goes
missing — a guard, not a comment, because a schema change that dropped it would
otherwise silently stop distinguishing builds it must distinguish.

### Two barriers, deliberately redundant

```python
CACHE_REQUIRED  = {task_primitive, seed_data_ref, data_rights, io_schema, …}
CACHE_EXCLUDED  = {owner_id, created_at, notes, budget_ceiling_usd, …}
```

1. **The key** stops the collision forming.
2. **The owner check** (`lookup_allowed`) stops it being served if one ever forms
   anyway — through a hash collision, a restored index, or a future optimisation
   written by somebody who has not read this page.

Either alone would prevent the leak. Both are required so that a mistake in one
is not sufficient to cause it.

Only **public-domain** data is shareable across owners. `licensed` is not: a
licence to *use* data is not a licence to serve somebody else a model trained on
it. An entry admitted without a spec is never served from the cache at all — the
safe default, since provenance nobody recorded cannot be shown to be shareable.

### What this bounds

| | |
|---|---|
| cross-customer exact hits | **only** on shared or public corpora |
| within-customer hits | the common case — rebuilds, retries, edits to excluded fields |
| the compose tier | unaffected: it builds a *new* adapter rather than serving someone else's |

The compose tier is how cross-customer value gets captured legitimately.

---

## The other correction: nothing could act on a recall

Lineage answers *who is affected*. Before this, nothing let you **act** on the
answer — a defective cartridge stayed certified forever and the runtime kept
loading it.

```
Status:  active | deprecated | recalled | superseded
```

`certified` and `servable` are different questions. Certification records that
the cartridge passed its gate; status records that a defect was found
*afterwards*. Both can be true at once, and only `recalled` stops it being
served. A deprecated cartridge still works — it is merely not the newest.

```python
reg.blast_radius("Qwen/Qwen3-1.7B")   # {"affected": 3, "share": 0.75, …}
reg.recall_base("Qwen/Qwen3-1.7B", "tokenizer defect")
```

A fleet recall is one query and one call. That is what lineage is *for*: "which
customers are affected?" has to be a query, not an investigation, because the
alternative is an incident with an unknown boundary. A recall must state its
reason — one with no stated defect cannot be reviewed, appealed or lifted.

---

## Deduplication

```
D(k) = k·S_B / (S_B + k·S_A)        k = cartridges per base
```

**Dedup is governed by `k`, not by `n`.** Adding customers helps; adding bases
hurts proportionally. That is the quantitative form of the base-diversity
tension below.

| k | D | |
|---|---|---|
| 1 | 0.97 | **worse than naive** |
| 2 | 1.90 | |
| 100 | 27.18 | |
| 500 | 34.74 | |
| ∞ | 37.3 | ceiling = S_B / S_A |

Two things worth knowing:

- **The ceiling is just the size ratio.** No amount of scale exceeds it. The only
  way to raise it is a smaller adapter or a larger base, and the latter costs
  device memory — a trade the Planner is already making.
- **CAS is worse for the first cartridge on a base** (break-even `k ≈ 1.03`),
  because it stores base *and* adapter where naive stores one merged model. A
  demo with three customers across six bases will measure CAS as worse than
  naive, and **that measurement is correct rather than a bug**.

> §2's table quotes `D(10) = 8.6`. The formula, at the sizes the table itself
> uses, gives **7.89** — and its other five rows reproduce exactly. No size pair
> yields 8.6 at k=10 while also giving a 37.3 ceiling, so that cell is a slip.
> The formula is authoritative.

---

## The cache hierarchy and margin

| tier | cost | p |
|---|---|---|
| exact | 0 | 0.15 |
| compose | 0.05 c | 0.20 |
| warm-start | 0.60 c | 0.25 |
| miss | c | 0.40 |

At `c = $40` and `P = $199`: `c̄ = $22.4` → **88.7%** margin, against **79.9%**
with no caching. **The nine points between those numbers is the entire economic
argument for the Registry**, and it is why cache-hit rate belongs on the same
dashboard as revenue.

**The compose tier is inert at cold start.** `p_compose = 0` until the library
holds tens of adapters per primitive, so a margin projection assuming it from
month one is wrong. It arrives about when the meta-learner does, for the same
reason: both need history.

---

## Base diversity is insurance, not fewer defects

The intuitive argument — more bases, smaller blast radius — is **wrong on
expectation**:

```
b · λ · (n/b) = λn        independent of b
```

More bases means more defects, each affecting proportionally fewer cartridges,
and the product does not move. Diversity pays only because loss is **superlinear**
in the number simultaneously affected — the reputational damage from "every
customer's model is broken at once" genuinely exceeds `n` times the damage from
one:

```
E[loss] = λ · n^γ · b^(1−γ)      decreasing in b only for γ > 1
```

At γ = 1 diversity buys exactly nothing. Against this, storage grows as `b·S_B`
and dedup falls as `D(n/b)`, so the optimum is finite. `diversity_tradeoff()`
returns the rows rather than picking a winner — the exchange rate between a
storage megabyte and a unit of reputational loss is a business judgement.

**Say it correctly in review: base diversity buys protection against correlated
catastrophic failure, not fewer defects.**

---

## The append-only corpus, and what it costs

```
R_g ⊆ R_{g+1}        for all g          (I-03)
```

If the corpus can shrink, a later generation can lose the real examples anchoring
it and the flywheel becomes a drift. `check_monotone()` names the records that
went missing, because "the corpus shrank" is not actionable and "these ids were
dropped between generations 4 and 5" is.

**The cost nobody had written down.** Append-only means unbounded growth, and
there is no code path that reduces it:

```
Storage(t) = |R_0| + r·V·t
```

At 200 requests/day and a 6% correction rate: 12 corrections/day ≈ **17 MB/year
per customer**. Negligible individually, real at scale, and permanent.

| permitted | why |
|---|---|
| `compress` | cold generations are text; every record survives |
| `content_dedup` | the *content* dedups, the *record* does not |
| `tier` | slower to read, still present |

| forbidden | why |
|---|---|
| `delete` | removes records outright |
| `sample` | deletion with extra steps |
| `window` | drops old generations, so `R_g ⊄ R_{g+1}` |
| `ttl` | deletion on a timer |

`apply_mitigation()` refuses the forbidden ones by name and states the reason.

---

## Synthetic data is a recipe, not an artefact

The largest storage win available, and it is free:

```
S_g = generate(R_g, recipe, teacher_hash, rng_seed)
```

Store the four arguments. Sixty thousand generated examples might be 200 MB; the
tuple that produces them is **under 1 KB** — a >99.9% reduction, lossless in the
only sense that matters.

**The honest caveat.** Regeneration is not bit-identical across hardware, because
floating-point reduction order differs between GPU models. So the original set's
content hash and summary statistics are stored too, and a regenerated set is
accepted on **eval equivalence** within tolerance rather than bit equality —
the same standard already adopted for build reproducibility. Demanding bit
equality would make the recipe useless on any machine but the one that ran it.

---

## Novelty, honestly

| claim | grade |
|---|---|
| Content-addressed dedup of base + adapters | none — textbook |
| Cache hierarchy with compose tier | none — LoraHub applied |
| **Licence composition as an obligation lattice** | moderate — unclaimed, mechanisable |
| **The cache-privacy constraint** | moderate — a real constraint nobody has written down |
| Synthetic-as-recipe | none — good engineering |
| Lineage / blast radius | none — standard provenance |

Registry stays **Grade C**. The two moderate rows belong inside the system paper,
not as standalone results, and neither should be oversold.
