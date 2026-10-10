"""Does the compiler generalise past the one dataset the paper reports?

The paper's evidence is SMS spam: one public corpus, one task, two classes.
Three things are therefore unestablished, and this study measures each.

  A. the COMPILER on other tasks — not the classifier in isolation but the
     whole gated path, plain-language description to certified cartridge,
     once per corpus;
  B. the PAPER'S PROTOCOL on other tasks — the identical three-split sweep,
     so every number here is comparable to the published table;
  C. TRANSFER to data the build never saw — fit on one corpus, test on
     another corpus of the same task. The paper never does this, and it is
     the half of the question that asks about new customer data.

Corpora come from ``evals.datasets``: SMS spam (the published baseline),
YouTube comment spam (same task, different customer), sentiment (different
task), review source (three classes), TREC question type (six classes).

    python -m evals.datasets && python -m evals.multitask
"""
from __future__ import annotations

import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modelrig import classifier  # noqa: E402
from modelrig.datasets import load_dataset, split_dataset  # noqa: E402
from modelrig.review import metrics, write_json  # noqa: E402

OUT = Path("outputs/evals/multitask")
DATA = Path("data")

# The seed corpus a customer would actually upload, not the whole archive.
# Gate 1's engineering floor for `classify` is 80 examples and the planner's
# coverage floor is 60, so this sits far above both: the cap is not what
# decides admission, which is what makes the per-task comparison fair.
SEED_CAP = 900
MIN_PER_CLASS = 12

TASKS: list[dict[str, Any]] = [
    {"name": "sms", "description":
        "Flag incoming SMS messages that are spam so they never reach the inbox.",
     "note": "the published baseline, re-run here for comparability"},
    {"name": "youtube_spam", "description":
        "Flag spam comments posted on our YouTube videos so moderators see them first.",
     "note": "same task as the baseline, different customer and channel"},
    {"name": "sentiment", "description":
        "Sort short product and restaurant reviews into positive or negative sentiment.",
     "note": "a different task: polarity, not abuse"},
    {"name": "review_source", "description":
        "Tell us which site a review sentence came from: amazon, imdb or yelp.",
     "note": "three classes over the same text distribution as sentiment"},
    {"name": "trec_questions", "description":
        "Route an incoming question by the kind of answer it wants, so it reaches "
        "the right lookup: an abbreviation, a description, an entity, a person, "
        "a place or a number.",
     "note": "six classes, and a label about the answer rather than the text"},
]

# Transfer is only meaningful between corpora that share a label set. These are
# the pairs that do; anything else would be comparing different alphabets.
TRANSFER = [("sms", "youtube_spam"), ("youtube_spam", "sms")]


def _stratified(corpus: list[tuple[str, str]], cap: int, seed: int = 0
                ) -> list[tuple[str, str]]:
    """Proportional sample that never starves a small class below the minimum."""
    by_label: dict[str, list[tuple[str, str]]] = {}
    for row in corpus:
        by_label.setdefault(row[1], []).append(row)
    rng = random.Random(seed)
    for rows in by_label.values():
        rng.shuffle(rows)
    total = len(corpus)
    picked: list[tuple[str, str]] = []
    for _label, rows in sorted(by_label.items()):
        want = max(MIN_PER_CLASS, round(cap * len(rows) / total))
        picked.extend(rows[:min(want, len(rows))])
    rng.shuffle(picked)
    return picked


# ====================================================================== #
# A. the compiler, end to end, once per task
# ====================================================================== #
def _interview(task: dict[str, Any], corpus: list[tuple[str, str]],
               quality_gate: float, *, answer_primitive: bool) -> Any:
    """Conduct the FORGE interview for one task.

    ``answer_primitive`` is the difference between the two things worth
    measuring. Left False, FORGE must infer everything from the description
    alone, and what it refuses to infer is the result. Set True, the slots a
    customer would fill at the prompt are filled, which is what the product
    does — the interview is a dialogue, and abandoning it at the first
    question is not a verdict on the compiler.
    """
    from api.studio import _corpus_ref, _probe_profile
    from modelrig.forge import Interviewer

    answers: dict[str, Any] = {
        "data_rights": "customer_owned",
        "quality_gate": quality_gate,
        "seed_data_count": len(corpus),
        "offline_required": True,
        "seed_data_ref": _corpus_ref(corpus),
        "latency_budget_ms": 30_000,
        "expected_input_tokens": 120,
        "io_schema": {"label": "str", "accept_unmeasured_latency": True},
    }
    if answer_primitive:
        # Every corpus here is a labelled single-label classification set, so
        # this is the honest answer to the question FORGE asks, not a nudge
        # towards the one primitive the local backend happens to execute.
        answers["task_primitive"] = "classify"
        answers["abstention_policy"] = "flag"
    return Interviewer().conduct(task["description"], device_profile=_probe_profile(),
                                 answers=answers)


def study_interview() -> dict[str, Any]:
    """A1 — what can FORGE settle from the description alone?

    This is reported separately because a spec FORGE declines to emit is not a
    compiler refusal. ``must_ask`` slots are the design working: FORGE would
    rather ask than guess the primitive.
    """
    records = []
    for task in TASKS:
        path = DATA / f"{task['name']}.jsonl"
        if not path.is_file():
            continue
        corpus = _stratified(load_dataset(str(path)), SEED_CAP)
        interview = _interview(task, corpus, 0.80, answer_primitive=False)
        inferred = interview.spec.task_primitive.value if interview.spec else None
        pending = [{"slot": q.as_dict()["slot"],
                    "must_ask": q.as_dict()["must_ask"],
                    "reason": q.as_dict()["reason"]} for q in interview.pending]
        records.append({
            "task": task["name"], "description": task["description"],
            "spec_emitted": interview.spec is not None,
            "inferred_primitive": inferred,
            "correct_primitive": "classify",
            "inference_agrees": inferred == "classify",
            "pending": pending,
        })
        print(f"  {task['name']:<16}"
              f"{'spec' if interview.spec else 'asks':<6}"
              f"primitive={inferred or '-':<9}"
              f"pending={','.join(p['slot'] for p in pending) or 'none'}")

    return {
        "question": "what does FORGE settle from a plain description, and what "
                    "does it insist on asking",
        "emitted": sum(r["spec_emitted"] for r in records),
        "agreed_on_primitive": sum(r["inference_agrees"] for r in records),
        "tasks": len(records),
        "records": records,
    }


def _compile_task(task: dict[str, Any], quality_gate: float, registry: Path
                  ) -> dict[str, Any]:
    """Run one task through Gate 1, the planner, data, training and Gate 3."""
    from modelrig.pipeline import MajesticCompiler

    path = DATA / f"{task['name']}.jsonl"
    corpus = _stratified(load_dataset(str(path)), SEED_CAP)
    labels = sorted({lab for _t, lab in corpus})
    interview = _interview(task, corpus, quality_gate, answer_primitive=True)
    if interview.spec is None:
        return {"task": task["name"], "admitted": False, "stage_reached": "interview",
                "refusal": "FORGE still would not emit a spec with the interview answered"}

    # The cache is a production optimisation and would quietly turn a second
    # run of this study into a lookup: an admitted cartridge with no scorecard,
    # no gate record and no accuracy, reported as a pass. Every task here is
    # built and scored for real.
    started = time.perf_counter()
    result = MajesticCompiler(base_path=registry).compile(
        interview.spec, corpus, use_cache=False)
    elapsed = time.perf_counter() - started

    # On a refusal at the proving ground there is no winner, so the winner's
    # scorecard is never set. The score that caused the refusal is on the
    # candidates, and reporting the refusal without it would hide the one
    # number that explains it.
    card = result.scorecard
    scorecard_from = "winner" if card is not None else None
    if card is None and result.selection is not None:
        card = next((c.scorecard for c in result.selection.candidates
                     if c.scorecard is not None), None)
        scorecard_from = "rejected_candidate" if card is not None else None
    axes = [{"name": a.name, "score": a.score, "threshold": a.threshold,
             "passed": a.passed, "blocking": a.blocking}
            for a in (card.axes if card else [])]
    accuracy = next((a["score"] for a in axes if a["name"] == "task_metric"), None)
    plan = result.plan
    return {
        "task": task["name"], "note": task["note"],
        "quality_gate": quality_gate,
        "n_seed": len(corpus), "n_classes": len(labels), "labels": labels,
        "admitted": bool(result.admitted),
        "stage_reached": result.stage_reached,
        "refusal": result.refusal,
        "cartridge_id": result.cartridge_id,
        "cache_hit": result.cache_hit,
        "gates": [{"gate": g.gate, "passed": g.passed, "reasons": list(g.reasons)}
                  for g in result.gates],
        "plan": {"base_ref": plan.base_ref, "bit_width": plan.bit_width,
                 "quantiser": plan.quantiser, "peft_method": plan.peft_method,
                 "target": plan.target} if plan else None,
        "held_out_accuracy": accuracy,
        "scorecard_from": scorecard_from,
        "n_held_out": card.n_held_out if card else None,
        "status": card.status.value if card else None,
        "axes": axes,
        # Carries the certificate's own `planned_base_executed` flag. The plan
        # names a Qwen base; the local backend that actually runs is the
        # centroid classifier, and an accuracy here must not be read as a
        # figure for the planned base.
        "execution": (result.cartridge.eval_certificate.get("execution", {})
                      if result.cartridge is not None else None),
        "repair_attempts": result.repair_attempts,
        "compile_seconds": round(elapsed, 2),
    }


def study_compiler() -> dict[str, Any]:
    """A2 — the full gated path per task, at the paper's 0.80 quality gate."""
    records = []
    for task in TASKS:
        if not (DATA / f"{task['name']}.jsonl").is_file():
            continue
        row = _compile_task(task, 0.80, OUT / "registry_080")
        records.append(row)
        state = "admitted" if row["admitted"] else f"refused at {row['stage_reached']}"
        acc = row.get("held_out_accuracy")
        shown = f"  held-out {acc:.3f}" if isinstance(acc, (int, float)) else ""
        print(f"  {task['name']:<16}{row.get('n_classes', '?')} cls  {state}{shown}")

    return {
        "question": "does the compiler architecture work on tasks other than SMS spam",
        "path": "FORGE interview answered, then MajesticCompiler.compile — "
                "Gate 1, planner, Gate 2, data factory, training, proving ground, Gate 3",
        "quality_gate": 0.80,
        "tasks": len(records),
        "admitted": sum(bool(r["admitted"]) for r in records),
        "records": records,
    }


def study_gate_binds(compiled: dict[str, Any]) -> dict[str, Any]:
    """A3 — for tasks refused on quality, is the threshold what refused them?

    A refusal at the proving ground says the model missed the gate the customer
    asked for. That is a different claim from "the compiler cannot do this
    task", and the two are distinguished by moving the gate to a level the
    measured accuracy can clear and seeing whether the same build is admitted.
    """
    records = []
    for row in compiled["records"]:
        if row["admitted"] or row["stage_reached"] != "prove":
            continue
        task = next(t for t in TASKS if t["name"] == row["task"])
        measured = row.get("held_out_accuracy")
        if not isinstance(measured, (int, float)):
            continue
        # A gate the measured score clears with room to spare, rounded down so
        # the number is not fitted to the result it is testing.
        relaxed = max(0.50, (int(measured * 20) - 1) / 20)
        retry = _compile_task(task, relaxed, OUT / "registry_relaxed")
        records.append({
            "task": row["task"],
            "refused_at_gate": row["quality_gate"],
            "measured_accuracy": measured,
            "relaxed_gate": relaxed,
            "admitted_at_relaxed_gate": retry["admitted"],
            "stage_reached": retry["stage_reached"],
            "refusal": retry["refusal"],
            "held_out_accuracy": retry.get("held_out_accuracy"),
        })
        print(f"  {row['task']:<16}measured {measured:.3f}  gate {row['quality_gate']:.2f}"
              f" -> {relaxed:.2f}  "
              f"{'admitted' if retry['admitted'] else 'still refused'}")

    return {
        "question": "when the proving ground refuses, is it the threshold or the "
                    "architecture that refused",
        "method": "recompile the same corpus at a quality gate the measured "
                  "held-out accuracy clears",
        "records": records,
    }


# ====================================================================== #
# B. the paper's protocol, once per task
# ====================================================================== #
def study_protocol() -> dict[str, Any]:
    """The published sweep, unchanged, on each corpus.

    ``review.benchmark`` is the function that produced the paper's table: three
    fixed stratified 80/20 splits, majority against centroid and 3-NN, each at
    fp32, int8 and int4, with Wilson intervals, answer-flip rates and warm
    latency. Running it verbatim is the point — a reimplemented harness would
    make the comparison with the published numbers worthless.
    """
    from modelrig.review import benchmark

    per_task = []
    for task in TASKS:
        path = DATA / f"{task['name']}.jsonl"
        if not path.is_file():
            continue
        out = OUT / task["name"] / "review.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        report = benchmark(path, out)
        records = report["classification"]

        def pick(method: str, quant: str,
                 rows: list[dict[str, Any]] = records) -> list[dict[str, Any]]:
            return [r for r in rows
                    if r["method"] == method and r["quantization"] == quant]

        n_rows = sum(1 for line in path.open(encoding="utf-8") if line.strip())
        summary: dict[str, Any] = {
            "task": task["name"], "note": task["note"], "n_rows": n_rows,
            "n_classes": len(records[0]["labels"]),
            "majority_accuracy": statistics.mean(
                r["accuracy"] for r in pick("majority", "none")),
        }
        for method in ("centroid", "knn"):
            for quant in ("none", "int8", "int4"):
                got = pick(method, quant)
                if not got:
                    continue
                summary[f"{method}_{quant}"] = {
                    "accuracy_mean": statistics.mean(r["accuracy"] for r in got),
                    "accuracy_min": min(r["accuracy"] for r in got),
                    "accuracy_max": max(r["accuracy"] for r in got),
                    "macro_f1_mean": statistics.mean(r["macro_f1"] for r in got),
                    "flip_rate_mean": statistics.mean(r["flip_rate"] for r in got),
                    "warm_p50_ms_mean": statistics.mean(r["warm_p50_ms"] for r in got),
                    "accuracy_ci95_seed0": got[0]["accuracy_ci95"],
                }
        best = summary["centroid_none"]["accuracy_mean"]
        summary["lift_over_majority"] = best - summary["majority_accuracy"]
        per_task.append(summary)
        print(f"  {task['name']:<16}{summary['n_classes']} cls  "
              f"centroid {best:.3f}  majority {summary['majority_accuracy']:.3f}  "
              f"lift {summary['lift_over_majority']:+.3f}")

    return {
        "question": "do the paper's numbers hold on other tasks under the same protocol",
        "protocol": "3 fixed stratified 80/20 splits; majority/centroid/knn "
                    "x none/int8/int4; identical to the published run",
        "limits": "split replicates overlap, so the spread across seeds is not an "
                  "independent-sample confidence statement",
        "per_task": per_task,
    }


# ====================================================================== #
# C. transfer to a corpus the build never saw
# ====================================================================== #
def study_transfer() -> dict[str, Any]:
    """Fit on one customer's data, test on another's.

    This is the sharpest form of the reviewer's second question. Both corpora
    are binary spam with the same label names, so the comparison is well
    posed: the only thing that changes is whose data the test rows came from.
    The in-domain figure is measured on a held-out split of the SOURCE corpus
    using the same fitted model, so the pair differs in nothing else.
    """
    records = []
    for source, target in TRANSFER:
        src_path, tgt_path = DATA / f"{source}.jsonl", DATA / f"{target}.jsonl"
        if not (src_path.is_file() and tgt_path.is_file()):
            continue
        src, tgt = load_dataset(str(src_path)), load_dataset(str(tgt_path))
        shared = sorted({lab for _t, lab in src} & {lab for _t, lab in tgt})
        if len(shared) < 2:
            continue

        train, held = split_dataset(src, 0.2, 0)
        labels = sorted({lab for _t, lab in train})
        for method, fit in (("centroid", classifier.fit_centroid),
                            ("knn", classifier.fit_knn)):
            model = fit(train, labels)
            in_dom = metrics(classifier.predict(model, [t for t, _ in held]),
                             [g for _t, g in held])
            # Only target rows whose label the source model can emit at all.
            usable = [(t, lab) for t, lab in tgt if lab in labels]
            out_dom = metrics(classifier.predict(model, [t for t, _ in usable]),
                              [g for _t, g in usable])
            records.append({
                "source": source, "target": target, "method": method,
                "shared_labels": shared, "n_train": len(train),
                "in_domain": {k: in_dom[k] for k in
                              ("accuracy", "macro_f1", "accuracy_ci95", "n_test")},
                "out_of_domain": {k: out_dom[k] for k in
                                  ("accuracy", "macro_f1", "accuracy_ci95", "n_test",
                                   "labels", "confusion_matrix")},
                "accuracy_drop": in_dom["accuracy"] - out_dom["accuracy"],
                "macro_f1_drop": in_dom["macro_f1"] - out_dom["macro_f1"],
            })
            print(f"  {source} -> {target:<16}{method:<9}"
                  f"in {in_dom['accuracy']:.3f}  out {out_dom['accuracy']:.3f}  "
                  f"drop {records[-1]['accuracy_drop']:+.3f}")

    return {
        "question": "does a model built on one customer's data hold up on another's",
        "method": "fit on a held-out split of the source corpus, then test that same "
                  "fitted model on every target row whose label it can emit",
        "pairs_available": [list(p) for p in TRANSFER],
        "pairs_skipped_reason": "transfer needs a shared label set; the sentiment, "
                                "source-attribution and question-type corpora share "
                                "no labels with the spam corpora or with each other",
        "largest_accuracy_drop": max((r["accuracy_drop"] for r in records), default=0.0),
        "records": records,
    }


def _findings(interview: dict[str, Any], compiler: dict[str, Any],
              gate: dict[str, Any], protocol: dict[str, Any],
              transfer: dict[str, Any]) -> list[str]:
    out = [f"FORGE emitted a spec unaided for {interview['emitted']} of "
           f"{interview['tasks']} descriptions, and inferred the right primitive "
           f"for {interview['agreed_on_primitive']}; the rest it insisted on asking",
           f"with the interview answered the compiler admitted "
           f"{compiler['admitted']} of {compiler['tasks']} tasks through the full "
           f"gated path at a {compiler['quality_gate']:.2f} quality gate"]
    for r in compiler["records"]:
        if not r.get("admitted"):
            out.append(f"{r['task']} refused at {r['stage_reached']}: {r['refusal']}")
    for r in gate["records"]:
        verdict = "admitted" if r["admitted_at_relaxed_gate"] else "still refused"
        out.append(f"{r['task']} measured {r['measured_accuracy']:.3f}; at a "
                   f"{r['relaxed_gate']:.2f} gate it is {verdict} — the threshold "
                   f"refused it, not the architecture")
    for task in protocol["per_task"]:
        out.append(f"{task['task']} ({task['n_classes']} classes): centroid fp32 "
                   f"{task['centroid_none']['accuracy_mean']:.3f}, majority "
                   f"{task['majority_accuracy']:.3f}, lift "
                   f"{task['lift_over_majority']:+.3f}")
    flips = [t["centroid_int4"]["flip_rate_mean"] for t in protocol["per_task"]
             if "centroid_int4" in t]
    if flips:
        out.append(f"int4 answer-flip rate across tasks: {min(flips):.4f} to "
                   f"{max(flips):.4f}")
    for r in transfer["records"]:
        out.append(f"{r['source']} -> {r['target']} ({r['method']}): accuracy falls "
                   f"{r['in_domain']['accuracy']:.3f} to "
                   f"{r['out_of_domain']['accuracy']:.3f} "
                   f"({r['accuracy_drop']:+.3f}) on data the build never saw")
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("A1. what FORGE settles from the description alone")
    interview = study_interview()
    print("\nA2. the compiler, end to end, per task")
    compiler = study_compiler()
    print("\nA3. is it the gate or the architecture that refuses")
    gate = study_gate_binds(compiler)
    print("\nB. the paper's protocol, per task")
    protocol = study_protocol()
    print("\nC. transfer to unseen customer data")
    transfer = study_transfer()

    report = {
        "study": "generalisation beyond the single reported dataset",
        "reviewer_item": "evaluate the compiler architecture on additional task "
                         "types beyond SMS spam classification",
        "interview": interview, "compiler": compiler, "quality_gate": gate,
        "protocol": protocol, "transfer": transfer,
        "findings": _findings(interview, compiler, gate, protocol, transfer),
    }
    write_json(OUT / "report.json", report)
    print("\nfindings")
    for line in report["findings"]:
        print(f"  - {line}")
    print(f"\nwritten to {OUT / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
