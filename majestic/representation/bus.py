"""Shared representation bus — the common latent space.

This is where a JEPA-style world model and the modality encoders interoperate:
experts consume and produce vectors here, enabling reasoning in representation
space rather than only in tokens.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class RepresentationBus(ABC):
    @abstractmethod
    def put(self, key: str, vector: list[float]) -> None: ...

    @abstractmethod
    def get(self, key: str) -> list[float]: ...
