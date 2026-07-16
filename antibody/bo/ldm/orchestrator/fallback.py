"""bo/ldm/orchestrator/fallback.py — fallback strategies when LLM fails."""
from __future__ import annotations

from typing import Any


def fallback_to_original_antbo(_status: Any) -> tuple[None, None]:
    """Return ``(None, None)``: trust region resets to AntBO default, bias preserved.

    The Orchestrator keeps the previous bias atom by NOT clearing it.
    """
    return None, None