"""Shared data types for the Majestic LLM compound system."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"


@dataclass
class Request:
    """A user request entering the system."""
    content: Any
    modality: Modality = Modality.TEXT
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Step:
    """A single step routed to an expert or tool."""
    description: str
    target: str                                  # expert/tool name
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    """A decomposition of a request into ordered steps."""
    steps: list[Step] = field(default_factory=list)


@dataclass
class RouteDecision:
    """Router output: which target handles a step, and why."""
    target: str
    confidence: float
    estimated_cost: float
    escalate: bool = False


@dataclass
class Response:
    """A verified response leaving the system."""
    content: Any
    trace: list[str] = field(default_factory=list)   # path taken through the system
    verified: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
