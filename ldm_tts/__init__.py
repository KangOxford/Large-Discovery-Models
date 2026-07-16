"""Shared primitives for Large Discovery Model test-time search.

The task folders keep their domain-specific generators, scorers, and prompts.
This package holds the small pieces that should be common across tasks:
budgeted search-loop control, JSON trajectory writing, and score utilities.
"""

from ldm_tts.loop import LDMSearchLoopResult, LDMSearchRoundResult, run_budgeted_search
from ldm_tts.scoring import (
    as_float,
    best_item,
    finite_or_none,
    is_finite_number,
    ranked_items,
)
from ldm_tts.trajectory import AtomicJsonLog, JsonlTrajectoryRecorder, load_jsonl

__all__ = [
    "AtomicJsonLog",
    "JsonlTrajectoryRecorder",
    "LDMSearchLoopResult",
    "LDMSearchRoundResult",
    "as_float",
    "best_item",
    "finite_or_none",
    "is_finite_number",
    "load_jsonl",
    "ranked_items",
    "run_budgeted_search",
]
