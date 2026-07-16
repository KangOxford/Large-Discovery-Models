"""Reusable budgeted loop for LDM test-time search."""

from __future__ import annotations

from collections.abc import Callable, MutableSequence, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

HistoryItem = TypeVar("HistoryItem")


@dataclass
class LDMSearchRoundResult(Generic[HistoryItem]):
    """Result emitted by one task-specific LDM-TTS round.

    ``history_delta`` is appended to the loop history. ``record`` is handed to
    the caller-provided recorder even for empty rounds, preserving complete
    traces for debugging and publication plots.
    """

    history_delta: Sequence[HistoryItem] = field(default_factory=tuple)
    record: dict[str, Any] | None = None
    empty_reservoir: bool = False
    stop_reason: str | None = None


@dataclass
class LDMSearchLoopResult(Generic[HistoryItem]):
    """Final state from :func:`run_budgeted_search`."""

    history: list[HistoryItem]
    rounds_run: int
    early_stop_reason: str | None = None
    empty_reservoir_rounds: int = 0


def run_budgeted_search(
    history: MutableSequence[HistoryItem],
    *,
    budget: int,
    build_round: Callable[[int, list[HistoryItem]], LDMSearchRoundResult[HistoryItem]],
    record_round: Callable[[dict[str, Any]], None] | None = None,
    on_empty_reservoir: Callable[[int, int], None] | None = None,
    start_round: int = 0,
    max_empty_reservoir_rounds: int = 10,
    allow_early_stop: bool = True,
) -> LDMSearchLoopResult[HistoryItem]:
    """Run a task-specific LDM loop until the shared budget policy is met.

    The caller owns domain behavior through ``build_round``. The shared policy
    handles round numbering, history growth, empty-reservoir accounting, and
    standardized early-stop reasons.
    """

    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}")
    if start_round < 0:
        raise ValueError(f"start_round must be non-negative, got {start_round}")
    if max_empty_reservoir_rounds < 1:
        raise ValueError(
            "max_empty_reservoir_rounds must be >= 1, "
            f"got {max_empty_reservoir_rounds}"
        )

    current_history = list(history)
    round_idx = int(start_round)
    rounds_run = 0
    empty_reservoir_rounds = 0
    early_stop_reason: str | None = None

    while len(current_history) < budget:
        result = build_round(round_idx, list(current_history))
        if result.record is not None and record_round is not None:
            record_round(result.record)
        rounds_run += 1

        if result.stop_reason:
            early_stop_reason = result.stop_reason
            break

        if result.empty_reservoir:
            empty_reservoir_rounds += 1
            if on_empty_reservoir is not None:
                on_empty_reservoir(round_idx, empty_reservoir_rounds)
            if empty_reservoir_rounds >= max_empty_reservoir_rounds and allow_early_stop:
                early_stop_reason = "empty_reservoir_limit"
                break
            round_idx += 1
            continue

        if not result.history_delta:
            early_stop_reason = "empty_selection"
            break

        current_history.extend(result.history_delta)
        empty_reservoir_rounds = 0
        round_idx += 1

    history[:] = current_history
    return LDMSearchLoopResult(
        history=current_history,
        rounds_run=rounds_run,
        early_stop_reason=early_stop_reason,
        empty_reservoir_rounds=empty_reservoir_rounds,
    )
