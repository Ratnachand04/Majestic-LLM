"""Structured logging helpers (stdlib ``logging`` only)."""
from __future__ import annotations

import logging
import os

_CONFIGURED = False


def configure_logging(level: str | int | None = None) -> None:
    """Configure root logging once. Level comes from ``MAJESTIC_LOG_LEVEL`` or arg."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    lvl = level or os.environ.get("MAJESTIC_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for ``name``."""
    configure_logging()
    return logging.getLogger(name)
