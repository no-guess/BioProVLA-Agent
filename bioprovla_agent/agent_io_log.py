"""Shared helpers for concise agent input/output logging (English)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("bioprovla_agent.io")


def truncate(text: str | None, max_chars: int = 320) -> str:
    if text is None:
        return ""
    t = " ".join(str(text).split())
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + f"... ({len(str(text))} chars total)"


def log_block(title: str, lines: list[tuple[str, Any]]) -> None:
    """Print a bordered block for high-visibility console tests."""
    sep = "=" * 72
    logger.info("%s", sep)
    logger.info("%s", title)
    for key, val in lines:
        logger.info("  %-22s %s", key + ":", val)
    logger.info("%s", sep)
