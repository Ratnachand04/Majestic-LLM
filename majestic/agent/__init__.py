"""Constrained tool use (A-09).

Web scraping is a tool the model CALLS, never a capability the model contains.
Side effects live in the tool registry, and every retrieved page is untrusted
input rather than data.
"""
from majestic.agent.react import (
    ReActLoop,
    StepLimitExceeded,
    ToolCall,
    ToolRegistry,
    ToolSpec,
    parse_tool_call,
)

__all__ = [
    "ReActLoop",
    "StepLimitExceeded",
    "ToolCall",
    "ToolRegistry",
    "ToolSpec",
    "parse_tool_call",
]
