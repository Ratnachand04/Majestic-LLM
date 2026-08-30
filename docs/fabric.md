# Fabric — the typed runtime

*Part 6 of the architecture. Implemented in [`majestic/fabric/`](../majestic/fabric/).*

## Why typed, and why that is the contribution

Every agent framework is dynamic and untyped by design — nodes are functions,
edges are whatever the last node returned, the graph is discovered at execution
time. That flexibility is exactly what makes it **impossible to prove anything
about them**.

Fabric gives it up deliberately. In exchange it answers three questions before a
single token is generated:

- Will this run with the network off?
- Can attacker-controlled text reach a side-effecting tool?
- Will this fit in 4 GB and finish in 30 seconds?

None is answerable by testing, because testing samples paths and these are
properties of *all* paths.

> **The contribution is not the analyses — taint analysis is fifty years old. It
> is the type system that makes a model-and-tool graph analysable at all.**

Analysis runs at **construction**, not first execution, so there is no code path
where an unverified graph runs even once.

---

## The result: a decoding grammar is a taint sanitiser

Binary taint is too coarse, and it forbids useful designs for no security gain. A
five-way classifier that reads a scraped page and emits one of five labels *is*
tainted — and the attacker controls at most **2.3 bits**. Rejecting that graph
buys nothing.

So measure the channel instead of detecting it:

```
cap(v) = log₂ |D_out(v)|              bits a node can carry
cap(p) = min over v ∈ p of cap(v)     the bottleneck governs
safe_c(G) ⟺ every source→sink path has cap(p) ≤ c_max
```

**The new part**: the grammar compiled from a cartridge's `output_schema` —
which exists for output *validity* — already determines its channel capacity.
The mechanism is a security primitive for free.

| output | \|D\| | capacity | role |
|---|---|---|---|
| 5-way classifier | 5 | 2.3 bits | **sanitiser** at c_max ≥ 3 |
| enum {urgent, normal, low} | 3 | 1.6 bits | **sanitiser** |
| all-enum extraction | ~10⁴ | ~13 bits | narrow propagator |
| **one free-text field** | ∞ | **∞** | **propagator** |
| summarisation | ∞ | ∞ | propagator |

**One free-text field converts a sanitiser into a propagator.** That is a
design-time fact somebody can act on — constraining `doctor_name` to a lookup
over the known referrer list rather than free text collapses the channel from
unbounded to about ten bits, and may be the difference between a graph that
verifies and one that does not.

This is why capacity beats binary taint: it does not just say *unsafe*, it says
which node and by how much.

```
scrape -> summarise -> notify: unbounded, narrowed at 'summarise'
  constrain 'summarise': its output schema has an unconstrained field…
```

The remedy deliberately names a node whose schema *can* be changed. Naming the
scraper would be useless advice: its output is the attacker's, and there is no
schema to tighten.

### Linear, not exponential

The safety question is "does any path exceed `c_max`" — the **widest** path, the
max over paths of the min over nodes. Enumerating paths to find it is exponential
in the worst case and unnecessary: on a DAG the widest path to every node falls
out of one topological sweep, because every predecessor resolves before its
successor.

```
width(v) = max over predecessors u of min(width(u), cap(v))
```

**O(|V| + |E|), no iteration.** That is C5's linearity claim, and it holds for the
quantitative analysis and not only the binary one. Cyclic agent graphs need
fixpoint iteration; acyclic ones need one sweep. The DAG restriction — the
flexibility Fabric gave up — is what buys it.

### Two caveats that travel with any claim made from this

- **Capacity bounds one pass.** Repeated queries accumulate: a graph invoked `n`
  times has effective capacity `n·cap` against a patient attacker.
- **Capacity says nothing about *which* bits.** One bit that flips an
  authorisation is worse than ten that pick a label, so `c_max` belongs per sink,
  set by consequence.

### Cartridges propagate

The tempting error: *the model read the scraped page and wrote its own summary,
so the output is the model's, not the attacker's.* False — a model given injected
instructions emits attacker-chosen content. A cartridge is a propagator by
default, and a `confirm` node is the only thing that clears taint, because a
human has interposed judgement.

---

## Offline closure: rewriting, not configuration

The naive predicate rejects every hybrid-capable graph, because a router that
escalates *only when confidence is low* has `net(v) = true`. The tempting fix is
to set the escalation threshold to infinity and declare it offline.

That proves nothing. A threshold is a runtime value — mutable, restorable by a
bug or a flag flip, and exactly the failure where a customer who bought offline
discovers a network call at 2 a.m.

```
G_offline = G \ {net nodes} \ {edges into them}
```

**Offline mode deletes the branch and the proof is over the residue.** There is
nothing to misconfigure because there is nothing there. If the deletion
disconnects the graph, the task genuinely cannot be done offline and the
violation names the sink that became unreachable.

Reachability is measured from the **original** inputs. A sink fed only by the
deleted node becomes a root of the residue, and a naive check would call it
reachable when it has in fact lost its only input.

---

## The mixed-base RAM trap

```
M_peak = Σ over DISTINCT bases S_B  +  Σ adapters  +  KV  +  buffers
```

The first term dominates, and it is a sum over distinct bases:

| graph | bases | peak |
|---|---|---|
| 20 cartridges, **one** base | 1120 MB | **1720 MB** ✓ |
| 3 cartridges, **three** bases | 3360 MB | **3450 MB** ✗ on 4 GB |

**Three cartridges can cost twice as much as twenty.** So device-targeted graphs
must be **base-homogeneous**, which makes base selection a *graph-level* decision
rather than a per-cartridge one — a real interface change back into the Planner,
since a spec destined for a multi-cartridge graph must inherit the graph's base
constraint.

---

## Adapter paging is the *offline* problem

Everyone reaches for LRU because swapping looks like caching. But the graph is a
DAG and execution order is a topological sort, so the reference string is known
**before execution begins**. That is offline paging, where **Belady's MIN is
optimal** — no heuristic needed, and no online policy can do better.

Two refinements, both implemented: branch arms are weighted by probability, and
`grouped_order()` picks the topological order that keeps same-adapter nodes
adjacent, which shrinks the working set for free at construction time.

---

## Where to invest: abstention, not the router

```
A_sys = ρ·A_correct + (1−ρ)·A_fallback
```

The earlier claim — a 91% router over 96% experts gives ~87% — silently assumed
`A_fallback = 0`, i.e. a misroute always produces a wrong answer:

| A_fallback | A_sys | |
|---|---|---|
| 0.00 | **87.4%** | misroute ⇒ wrong |
| 0.70 | **93.7%** | wrong specialist still partly useful |
| abstain + reroute | ~95.9% | wrong specialist **detects** it is wrong |

**Seven points of system accuracy live entirely in what happens on a misroute.**
Improving the router 0.91 → 0.95 buys 3.8 points; making misroutes graceful buys
6.3, and it helps at *every* router accuracy rather than only at the margin
improved. That is why `behaviour.abstention_policy` is trained and evaluated
rather than decorative.

---

## Execution

The runtime **recomputes taint and asserts it matches the static proof**. A cheap
dynamic check whose only job is to catch drift between the analyser and the
executor — two implementations of one rule, edited by different people at
different times, where a divergence would otherwise surface as a security failure
in the field rather than a test failure here. A mismatch is fatal: the graph was
admitted on the static proof, so disagreement means admission used a different
rule than execution, and neither verdict can be trusted.

---

## Novelty, honestly

| claim | grade |
|---|---|
| Typed DAG for model-and-tool graphs | moderate — the enabling type system |
| Offline closure by syntactic disabling | moderate — unclaimed, and the product promise depends on it |
| Taint analysis on a DAG | none — textbook dataflow |
| **Capacity from the decoding grammar** | **strong** — new, and it upgrades C5 |
| Belady for adapter paging | none — but nobody has noticed it applies |
| Quality-gap routing | none — Hybrid LLM |
| Misroute penalty analysis | moderate — reframes where to invest |

Fabric holds **Grade A**, and the capacity result is the reason: it converts C5
from "we can detect unsafe graphs" into "we can quantify how unsafe, and tell you
which field to constrain to fix it".

## Still open

- `output_domain_bits` is derived from a schema here; it should be computed from
  the **compiled GBNF grammar** at package time and stored in the cartridge
  manifest, so the analysis reads a measured number rather than re-deriving one.
- Swap latency is assumed at 100 ms (GAP-10). §19's telemetry is where that gets
  confirmed or falsified in the field, at no extra cost.
- `NUMERIC_BITS = 32` is a placeholder until grammars declare numeric ranges.
