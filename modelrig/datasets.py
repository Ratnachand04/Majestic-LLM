"""Tiny built-in datasets + a loader, so a factory build runs with no downloads.

External datasets can be loaded from a ``.jsonl`` (objects with ``text`` and
``label``) or ``.csv`` (``text,label`` header) file path.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

# A tiny, clearly separable sentiment set. Every example carries several shared
# "anchor" words (good/great/love/happy/nice vs bad/hate/awful/terrible/sad) so
# that a held-out example always overlaps the training vocabulary — enough for a
# nearest-centroid/kNN classifier to clear a modest quality gate.
_SENTIMENT: list[tuple[str, str]] = [
    ("i love this, so good and great", "positive"),
    ("great and good, a happy wonderful choice", "positive"),
    ("love it, nice and good and happy", "positive"),
    ("wonderful and great, i am so happy", "positive"),
    ("good great love, a nice happy day", "positive"),
    ("nice and good, i really love this great thing", "positive"),
    ("happy and good, great and lovely work", "positive"),
    ("i love how good and nice and happy it is", "positive"),
    ("great good wonderful, love and happy", "positive"),
    ("so good, so great, i love it, very happy", "positive"),
    ("a good and great and nice experience i love", "positive"),
    ("love love love, good great and happy", "positive"),
    ("nice good happy, wonderful and i love it", "positive"),
    ("great and happy and good, truly love this", "positive"),
    ("good great happy love nice all around", "positive"),
    ("i really love this good great happy product", "positive"),
    ("i hate this, so bad and awful", "negative"),
    ("awful and bad, a sad terrible choice", "negative"),
    ("hate it, bad and awful and sad", "negative"),
    ("terrible and bad, i am so sad", "negative"),
    ("bad awful hate, a sad terrible day", "negative"),
    ("awful and bad, i really hate this terrible thing", "negative"),
    ("sad and bad, terrible and hateful work", "negative"),
    ("i hate how bad and awful and sad it is", "negative"),
    ("bad awful terrible, hate and sad", "negative"),
    ("so bad, so awful, i hate it, very sad", "negative"),
    ("a bad and awful and terrible experience i hate", "negative"),
    ("hate hate hate, bad awful and sad", "negative"),
    ("awful bad sad, terrible and i hate it", "negative"),
    ("bad and sad and awful, truly hate this", "negative"),
    ("bad awful sad hate terrible all around", "negative"),
    ("i really hate this bad awful sad product", "negative"),
]

_BUILTIN = {"sentiment": _SENTIMENT}


def load_dataset(source: str) -> list[tuple[str, str]]:
    """Load ``(text, label)`` pairs from a builtin name or a file path."""
    if source.startswith("builtin:"):
        name = source.split(":", 1)[1]
        if name not in _BUILTIN:
            raise ValueError(f"unknown builtin dataset {name!r}; have {list(_BUILTIN)}")
        return list(_BUILTIN[name])

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {source}")
    rows: list[tuple[str, str]] = []
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                obj = json.loads(line)
                rows.append((str(obj["text"]), str(obj["label"])))
    elif path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as fh:
            for obj in csv.DictReader(fh):
                rows.append((str(obj["text"]), str(obj["label"])))
    else:
        raise ValueError(f"unsupported dataset format: {path.suffix} (use .jsonl or .csv)")
    if not rows:
        raise ValueError(f"dataset is empty: {source}")
    return rows


def split_dataset(
    rows: list[tuple[str, str]], test_split: float, seed: int
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Deduplicate normalized inputs, reject conflicting labels, then stratify.

    Duplicate messages cannot inflate evidence or occur on both sides of the split.
    """
    if not 0 < test_split < 1:
        raise ValueError("test_split must be between zero and one")
    unique: dict[str, tuple[str, str]] = {}
    for text, label in rows:
        key = " ".join(text.casefold().split())
        if not key or not label.strip():
            raise ValueError("dataset contains empty text or label")
        if key in unique and unique[key][1] != label:
            raise ValueError("identical normalized input has conflicting labels")
        unique.setdefault(key, (text, label))
    rng = random.Random(seed)
    train, test = [], []
    for label in sorted({label for _, label in unique.values()}):
        group = [row for row in unique.values() if row[1] == label]
        if len(group) < 2:
            raise ValueError(f"label {label!r} needs at least two distinct examples")
        rng.shuffle(group)
        n_test = min(len(group) - 1, max(1, int(len(group) * test_split)))
        train.extend(group[n_test:])
        test.extend(group[:n_test])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test
