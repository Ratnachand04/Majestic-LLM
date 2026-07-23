"""Multimodal encoders that map raw inputs onto the shared representation bus.

The abstract :class:`Encoder` is the contract. :class:`HashingTextEncoder` is a
dependency-free, deterministic text embedder used offline and in tests. Phase 3
adds an optional embedding-model-backed encoder for higher-quality retrieval.
"""
from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import Any

from majestic.types import Modality

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Encoder(ABC):
    """Encodes one modality into a latent vector on the shared bus."""

    modality: Modality

    @abstractmethod
    def encode(self, data: Any) -> list[float]:
        """Return an embedding for the input."""
        raise NotImplementedError


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HashingTextEncoder(Encoder):
    """Feature-hashing text encoder: deterministic, offline, no dependencies.

    Tokens are hashed into a fixed-dimensional vector (the "hashing trick") and
    the result is L2-normalized so cosine similarity is meaningful. Crude but
    real — good enough to exercise retrieval and vector-store code paths without
    downloading a model.
    """

    modality = Modality.TEXT

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def encode(self, data: Any) -> list[float]:
        text = data if isinstance(data, str) else str(data)
        vec = [0.0] * self.dim
        for tok in _tokenize(text):
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class SentenceTransformerEncoder(Encoder):
    """Optional embedding-model encoder (lazy imports ``sentence-transformers``).

    Higher-quality retrieval than the hashing encoder, at the cost of a model
    download. Selected via ``retrieval.embedding_model`` (any non-"hash" value is
    treated as a model id).
    """

    modality = Modality.TEXT

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self._model = SentenceTransformer(model_id)

    def encode(self, data: Any) -> list[float]:
        text = data if isinstance(data, str) else str(data)
        vec = self._model.encode([text], normalize_embeddings=True)[0]
        return [float(x) for x in vec]


# Backwards-compatible alias for the scaffold's ``TextEncoder`` name.
TextEncoder = HashingTextEncoder


# TODO: ImageEncoder, AudioEncoder, VideoEncoder follow the same pattern.
