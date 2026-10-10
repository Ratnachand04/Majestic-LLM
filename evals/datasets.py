"""Fetch and normalise the public corpora the multi-task study runs on.

Every corpus is written as JSONL of ``{"text": ..., "label": ...}`` beside a
provenance record carrying the download digest, the post-processing digest, the
row counts and the label histogram — the same shape ``data/sms.provenance.json``
already uses, so a reader can check any of them the same way.

Deduplication matches the existing SMS treatment: casefold-and-whitespace key,
drop exact duplicates, and discard any group whose rows disagree on the label.
A corpus that cannot be deduplicated cleanly is reported, not silently kept.

    python -m evals.datasets
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA = Path("data")
_WS = re.compile(r"\s+")


def _get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "majestic-eval/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return fh.read()


def _key(text: str) -> str:
    return _WS.sub(" ", text.strip().casefold())


def _dedupe(rows: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Exact-duplicate removal, and discard groups that disagree on the label."""
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for text, label in rows:
        groups[_key(text)].append((text, label))
    kept: list[tuple[str, str]] = []
    conflicting = 0
    for members in groups.values():
        labels = {lab for _t, lab in members}
        if len(labels) > 1:
            conflicting += 1
            continue
        kept.append(members[0])
    return kept, {"raw_rows": len(rows), "unique_rows": len(kept),
                  "conflicting_groups": conflicting,
                  "removed_rows": len(rows) - len(kept)}


def _write(name: str, rows: list[tuple[str, str]], meta: dict[str, Any]) -> dict[str, Any]:
    DATA.mkdir(exist_ok=True)
    path = DATA / f"{name}.jsonl"
    body = "\n".join(json.dumps({"text": t, "label": lab}, ensure_ascii=False)
                     for t, lab in rows) + "\n"
    path.write_text(body, encoding="utf-8")
    meta = dict(meta)
    meta["dataset_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    meta["labels"] = dict(sorted(Counter(lab for _t, lab in rows).items()))
    meta["n_classes"] = len(meta["labels"])
    meta["preprocessing"] = ("casefold and whitespace key; deduplicate; "
                             "discard conflicting groups")
    (DATA / f"{name}.provenance.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# =========================================================================== #
def youtube_spam() -> dict[str, Any]:
    """UCI 380 — comment spam across five YouTube channels.

    The same task as SMS on different customer data, which is the second half
    of the reviewer's question: not only other tasks, but this task on a corpus
    the system has never seen.
    """
    url = "https://archive.ics.uci.edu/static/public/380/youtube+spam+collection.zip"
    blob = _get(url)
    rows: list[tuple[str, str]] = []
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for member in sorted(zf.namelist()):
            if not member.lower().endswith(".csv"):
                continue
            text = zf.read(member).decode("utf-8", errors="replace")
            for rec in csv.DictReader(io.StringIO(text)):
                content = (rec.get("CONTENT") or "").strip()
                tag = (rec.get("CLASS") or "").strip()
                if content and tag in {"0", "1"}:
                    rows.append((content, "spam" if tag == "1" else "ham"))
    kept, counts = _dedupe(rows)
    return _write("youtube_spam", kept, {
        "dataset": "UCI YouTube Spam Collection", "url": url,
        "citation": "https://doi.org/10.24432/C58885",
        "license": "CC BY 4.0 (UCI listing)",
        "task": "binary comment-spam detection",
        "download_sha256": hashlib.sha256(blob).hexdigest(), **counts})


def sentiment_sentences() -> dict[str, Any]:
    """UCI 331 — sentiment over Amazon, IMDb and Yelp sentences.

    A different task from spam: polarity rather than abuse, and the surface
    cues that carry it are different.
    """
    url = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
    blob = _get(url)
    rows: list[tuple[str, str]] = []
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for member in sorted(zf.namelist()):
            if not member.endswith("_labelled.txt"):
                continue
            for line in zf.read(member).decode("utf-8", errors="replace").splitlines():
                if "\t" not in line:
                    continue
                sentence, _sep, tag = line.rpartition("\t")
                tag = tag.strip()
                if sentence.strip() and tag in {"0", "1"}:
                    rows.append((sentence.strip(),
                                 "positive" if tag == "1" else "negative"))
    kept, counts = _dedupe(rows)
    return _write("sentiment", kept, {
        "dataset": "UCI Sentiment Labelled Sentences", "url": url,
        "citation": "https://doi.org/10.24432/C57604",
        "license": "CC BY 4.0 (UCI listing)",
        "task": "binary sentiment polarity",
        "download_sha256": hashlib.sha256(blob).hexdigest(), **counts})


def sentiment_domain() -> dict[str, Any]:
    """The same sentences, relabelled by their source site.

    A three-class topic/provenance task built from an already-downloaded
    corpus. It is derived rather than found, and is marked as such in the
    provenance so nobody mistakes it for an established benchmark. Its value
    here is that it exercises a three-class decision over the same text
    distribution, isolating class count from domain shift.
    """
    url = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
    blob = _get(url)
    rows: list[tuple[str, str]] = []
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for member in sorted(zf.namelist()):
            if not member.endswith("_labelled.txt"):
                continue
            site = Path(member).name.split("_")[0].lower()
            for line in zf.read(member).decode("utf-8", errors="replace").splitlines():
                sentence, _sep, tag = line.rpartition("\t")
                if sentence.strip() and tag.strip() in {"0", "1"}:
                    rows.append((sentence.strip(), site))
    kept, counts = _dedupe(rows)
    return _write("review_source", kept, {
        "dataset": "UCI Sentiment Labelled Sentences, relabelled by source site",
        "url": url, "citation": "https://doi.org/10.24432/C57604",
        "license": "CC BY 4.0 (UCI listing)",
        "task": "three-class source attribution (DERIVED, not a published benchmark)",
        "derived": True,
        "download_sha256": hashlib.sha256(blob).hexdigest(), **counts})


def trec_questions() -> dict[str, Any]:
    """TREC question classification — six coarse answer types.

    The multiclass case, and a genuinely different task shape: the label is
    what kind of answer the question wants, not a property of the text's tone
    or intent.
    """
    url = "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label"
    blob = _get(url)
    rows: list[tuple[str, str]] = []
    for line in blob.decode("latin-1").splitlines():
        line = line.strip()
        if not line or " " not in line:
            continue
        tag, _sep, question = line.partition(" ")
        coarse = tag.split(":")[0].strip()
        if question.strip() and coarse:
            rows.append((question.strip(), coarse))
    kept, counts = _dedupe(rows)
    return _write("trec_questions", kept, {
        "dataset": "TREC question classification (coarse labels)", "url": url,
        "citation": "Li and Roth, COLING 2002; Hovy et al., HLT 2001",
        "license": "public research distribution (UIUC CogComp)",
        "task": "six-class question answer-type classification",
        "download_sha256": hashlib.sha256(blob).hexdigest(), **counts})


BUILDERS = {
    "youtube_spam": youtube_spam,
    "sentiment": sentiment_sentences,
    "review_source": sentiment_domain,
    "trec_questions": trec_questions,
}


def main() -> int:
    print(f"{'corpus':<18}{'rows':>7}{'classes':>9}  labels")
    for name, build in BUILDERS.items():
        try:
            meta = build()
        except Exception as exc:                               # noqa: BLE001
            print(f"{name:<18}{'FAILED':>7}  {type(exc).__name__}: {exc}")
            continue
        labels = ", ".join(f"{k}={v}" for k, v in meta["labels"].items())
        print(f"{name:<18}{meta['unique_rows']:>7}{meta['n_classes']:>9}  {labels[:60]}")
    existing = DATA / "sms.provenance.json"
    if existing.is_file():
        meta = json.loads(existing.read_text(encoding="utf-8"))
        labels = ", ".join(f"{k}={v}" for k, v in meta["labels"].items())
        print(f"{'sms (existing)':<18}{meta['unique_rows']:>7}{2:>9}  {labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
