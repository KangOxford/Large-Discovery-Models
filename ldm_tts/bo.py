"""Lightweight BO interface records for LDM task adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, Sequence

from ldm_tts.spaces import AcquisitionSpec, CandidateSpaceSpec


@dataclass(frozen=True)
class FeatureVector:
    """Encoded candidate representation for surrogate models."""

    values: tuple[float, ...]
    version: str
    source_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BOObservation:
    """One evaluated candidate and its objective values."""

    candidate_id: str
    objectives: tuple[float | None, ...]
    feature: FeatureVector | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BOPrediction:
    """Surrogate/acquisition readout for one candidate."""

    candidate_id: str
    mean: tuple[float, ...] = ()
    std: tuple[float, ...] = ()
    acquisition_score: float | None = None
    score_direction: str = "maximize"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BOSelectionResult:
    """Selected candidate ids plus the prediction pool used to select them."""

    selected_candidate_ids: tuple[str, ...]
    predictions: tuple[BOPrediction, ...] = ()
    fallback_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FeatureEncoder(Protocol):
    """Protocol for task-owned candidate encoders."""

    def describe(self) -> CandidateSpaceSpec:
        ...

    def encode(self, candidate: Any) -> FeatureVector:
        ...


class AcquisitionSelector(Protocol):
    """Protocol for task adapters that fit surrogates and select candidates.

    Acquisition math belongs to :mod:`ldm_tts.acquisition`; adapters own only
    candidate encoding, surrogate fitting, and selection mechanics.
    """

    def describe(self) -> AcquisitionSpec:
        ...

    def fit(self, history: Sequence[BOObservation]) -> None:
        ...

    def select(self, candidates: Sequence[Any]) -> BOSelectionResult:
        ...
