"""API request/response schemas (dataclasses; framework-agnostic)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerateRequest:
    prompt: str
    modality: str = "text"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerateResponse:
    output: Any
    trace: list[str] = field(default_factory=list)
    verified: bool = False
