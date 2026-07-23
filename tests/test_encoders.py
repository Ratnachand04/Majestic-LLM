"""Tests for the offline hashing text encoder."""
from __future__ import annotations

import math

from majestic.perception.encoders import HashingTextEncoder


def test_deterministic_and_normalized():
    enc = HashingTextEncoder(dim=64)
    a = enc.encode("the quick brown fox")
    b = enc.encode("the quick brown fox")
    assert a == b
    assert len(a) == 64
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, rel_tol=1e-6)


def test_similar_texts_closer_than_dissimilar():
    enc = HashingTextEncoder(dim=256)

    def cos(u, v):
        return sum(x * y for x, y in zip(u, v))

    base = enc.encode("machine learning models on device")
    near = enc.encode("on device machine learning models")
    far = enc.encode("a completely unrelated sentence about oranges")
    assert cos(base, near) > cos(base, far)


def test_empty_text_is_zero_vector():
    enc = HashingTextEncoder(dim=16)
    assert enc.encode("") == [0.0] * 16
