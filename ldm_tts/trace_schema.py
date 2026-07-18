"""Shared JSON shapes for LDM-TTS round traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CandidateTraceRecord:
    """Serializable candidate row used inside a round trace."""

    candidate_id: str
    payload: Any
    source: str = ""
    prediction: dict[str, Any] | None = None
    true_scores: tuple[float | None, ...] = ()
    selected: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LDMRoundTrace:
    """Task-neutral round record for future shared trajectory analysis."""

    round_idx: int
    task: str
    history_size_before: int
    history_size_after: int
    response_space: str
    acquisition: str
    candidates: tuple[CandidateTraceRecord, ...] = ()
    selected_candidate_ids: tuple[str, ...] = ()
    llm_attempts: tuple[dict[str, Any], ...] = ()
    fallback_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
