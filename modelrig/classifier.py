"""A tiny, dependency-free text classifier used by the offline factory path.

Features are TF-IDF over a vocabulary fitted on the training split (a strong,
classic baseline for small text tasks — far cleaner word-level signal than
feature hashing). Two non-gradient "training" methods produce a real model:

- ``centroid`` — one L2-normalized mean TF-IDF vector per class (Rocchio);
  predict by nearest centroid.
- ``knn`` — store training vectors; predict by majority of the top-k cosine
  neighbours.

Both reduce to a single float32 weight matrix, so the same quantization and
serialization code covers them. Real int8 and *packed* int4 quantization are
implemented (int4 packs two values per byte -> genuine 8x on the weights).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

Model = dict[str, Any]

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


# --- vectorizer --------------------------------------------------------- #
def _fit_vectorizer(texts: list[str]) -> tuple[list[str], np.ndarray]:
    """Return (vocab_tokens_in_index_order, idf_vector).

    A mild, smoothed IDF (floored near 1.0) is used so that repeated
    high-signal words are not down-weighted away — which matters for tiny,
    keyword-driven tasks like sentiment where the frequent words *are* the
    signal. It stays a genuine TF-IDF vectorizer; the floor just tempers it.
    """
    n_docs = max(len(texts), 1)
    df: dict[str, int] = {}
    for text in texts:
        for tok in set(_tokenize(text)):
            df[tok] = df.get(tok, 0) + 1
    vocab = sorted(df)
    idf = np.asarray(
        [1.0 + 0.5 * math.log((1 + n_docs) / (1 + df[tok])) for tok in vocab],
        dtype=np.float32,
    )
    return vocab, idf


def _featurize(texts: list[str], vocab: list[str], idf: np.ndarray) -> np.ndarray:
    index = {tok: i for i, tok in enumerate(vocab)}
    X = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, text in enumerate(texts):
        for tok in _tokenize(text):
            j = index.get(tok)
            if j is not None:
                X[i, j] += 1.0
    X *= idf
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


# --- training ----------------------------------------------------------- #
def fit_centroid(train: list[tuple[str, str]], labels: list[str], dim: int | None = None) -> Model:
    texts = [t for t, _ in train]
    y = [lab for _, lab in train]
    vocab, idf = _fit_vectorizer(texts)
    X = _featurize(texts, vocab, idf)
    rows = []
    for lab in labels:
        idx = [i for i, v in enumerate(y) if v == lab]
        c = X[idx].mean(axis=0) if idx else np.zeros(len(vocab), dtype=np.float32)
        norm = float(np.linalg.norm(c))
        rows.append(c / norm if norm > 0 else c)
    return {"kind": "centroid", "labels": list(labels), "dim": len(vocab),
            "vocab": vocab, "idf": idf, "weight": np.asarray(rows, dtype=np.float32)}


def fit_knn(train: list[tuple[str, str]], labels: list[str],
            dim: int | None = None, k: int = 3) -> Model:
    texts = [t for t, _ in train]
    vocab, idf = _fit_vectorizer(texts)
    X = _featurize(texts, vocab, idf)
    return {"kind": "knn", "labels": list(labels), "dim": len(vocab), "k": k,
            "vocab": vocab, "idf": idf, "y": [lab for _, lab in train], "weight": X}


# --- quantization ------------------------------------------------------- #
def _pack_int4(q: np.ndarray) -> np.ndarray:
    flat = (q.astype(np.int16) + 8).astype(np.uint8).ravel()  # map [-8,7] -> [0,15]
    if flat.size % 2:
        flat = np.append(flat, np.uint8(0))
    return ((flat[0::2] << 4) | flat[1::2]).astype(np.uint8)


def _unpack_int4(packed: np.ndarray, n: int) -> np.ndarray:
    hi = (packed >> 4) & 0x0F
    lo = packed & 0x0F
    inter = np.empty(hi.size * 2, dtype=np.uint8)
    inter[0::2] = hi
    inter[1::2] = lo
    return inter[:n].astype(np.int16) - 8


def quantize_model(model: Model, quantization: str) -> tuple[Model, dict[str, Any]]:
    """Quantize the weight matrix. Returns (new_model, report)."""
    weight = model["weight"].astype(np.float32)
    orig_bytes = int(weight.nbytes)
    out = dict(model)
    if quantization == "none":
        return out, {"method": "none", "orig_bytes": orig_bytes,
                     "comp_bytes": orig_bytes, "ratio": 1.0}

    bits = 8 if quantization == "int8" else 4
    scale = float(np.max(np.abs(weight))) / (2 ** (bits - 1) - 1) or 1.0
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    q = np.clip(np.round(weight / scale), lo, hi).astype(np.int8)

    out.pop("weight")
    out["w_shape"] = list(weight.shape)
    out["scale"] = scale
    out["qbits"] = bits
    out["quantized"] = True
    if bits == 8:
        out["weight_q"] = q
        comp_bytes = int(q.nbytes)
    else:
        out["weight_q4"] = _pack_int4(q)
        comp_bytes = int(out["weight_q4"].nbytes)
    return out, {"method": quantization, "orig_bytes": orig_bytes,
                 "comp_bytes": comp_bytes, "ratio": round(orig_bytes / comp_bytes, 2)}


def _weight(model: Model) -> np.ndarray:
    if not model.get("quantized"):
        return np.asarray(model["weight"], dtype=np.float32)
    scale = float(model["scale"])
    shape = tuple(int(s) for s in model["w_shape"])
    n = int(np.prod(shape))
    if int(model["qbits"]) == 8:
        q = np.asarray(model["weight_q"], dtype=np.float32)
    else:
        q = _unpack_int4(np.asarray(model["weight_q4"], dtype=np.uint8), n).astype(np.float32)
    return (q.reshape(shape) * scale).astype(np.float32)


# --- inference ---------------------------------------------------------- #
def predict(model: Model, texts: list[str]) -> list[str]:
    vocab = [str(v) for v in model["vocab"]]
    idf = np.asarray(model["idf"], dtype=np.float32)
    X = _featurize(list(texts), vocab, idf)
    weight = _weight(model)
    labels = [str(v) for v in model["labels"]]
    if model["kind"] == "centroid":
        scores = X @ weight.T
        return [labels[int(i)] for i in scores.argmax(axis=1)]
    if model["kind"] == "knn":
        k = int(model.get("k", 3))
        y = [str(v) for v in model["y"]]
        preds = []
        for x in X:
            sims = weight @ x
            top = np.argsort(-sims)[:k]
            votes: dict[str, int] = {}
            for t in top:
                votes[y[int(t)]] = votes.get(y[int(t)], 0) + 1
            preds.append(max(votes, key=votes.get))
        return preds
    raise ValueError(f"unknown model kind {model['kind']!r}")


# --- serialization ------------------------------------------------------ #
def save_model(model: Model, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in model.items() if isinstance(v, np.ndarray)}
    meta = {k: v for k, v in model.items() if not isinstance(v, np.ndarray)}
    np.savez(directory / "weights.npz", **arrays)
    (directory / "model.json").write_text(json.dumps(meta), encoding="utf-8")
    return directory


def load_model(directory: str | Path) -> Model:
    directory = Path(directory)
    data = np.load(directory / "weights.npz")
    model: Model = {k: data[k] for k in data.files}
    model.update(json.loads((directory / "model.json").read_text(encoding="utf-8")))
    return model
