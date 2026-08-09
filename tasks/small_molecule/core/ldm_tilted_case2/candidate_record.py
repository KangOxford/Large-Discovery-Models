"""Shared candidate and source records for tilted case2 methods."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class SourceRecord:
    source_id: str
    source_type: str
    seed_smiles: Optional[str]
    source_weight: float
    requested_budget: int
    generated_count: int = 0
    valid_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceRecord":
        return cls(
            source_id=str(data["source_id"]),
            source_type=str(data["source_type"]),
            seed_smiles=data.get("seed_smiles"),
            source_weight=float(data["source_weight"]),
            requested_budget=int(data["requested_budget"]),
            generated_count=int(data.get("generated_count", 0)),
            valid_count=int(data.get("valid_count", 0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class CandidateRecord:
    raw_smiles: str
    canonical_smiles: Optional[str]
    method: str
    sources: list[str]
    occurrence_by_source: dict[str, int]
    base_support_level: Optional[str] = None
    base_support_value: float = 0.0
    q0_base_mass: float = 0.0
    mu: Optional[list[float]] = None
    sigma: Optional[list[float]] = None
    ehvi: Optional[float] = None
    ehvi_z: Optional[float] = None
    log_weight: Optional[float] = None
    resampling_probability: Optional[float] = None
    selected: bool = False
    true_scores: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateRecord":
        return cls(
            raw_smiles=str(data["raw_smiles"]),
            canonical_smiles=data.get("canonical_smiles"),
            method=str(data["method"]),
            sources=list(data.get("sources", [])),
            occurrence_by_source={
                str(k): int(v) for k, v in dict(data.get("occurrence_by_source", {})).items()
            },
            base_support_level=data.get("base_support_level"),
            base_support_value=float(data.get("base_support_value", 0.0)),
            q0_base_mass=float(data.get("q0_base_mass", 0.0)),
            mu=_optional_float_list(data.get("mu")),
            sigma=_optional_float_list(data.get("sigma")),
            ehvi=_optional_float(data.get("ehvi")),
            ehvi_z=_optional_float(data.get("ehvi_z")),
            log_weight=_optional_float(data.get("log_weight")),
            resampling_probability=_optional_float(data.get("resampling_probability")),
            selected=bool(data.get("selected", False)),
            true_scores=_optional_float_list(data.get("true_scores")),
            metadata=dict(data.get("metadata", {})),
        )


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _optional_float_list(value: Any) -> Optional[list[float]]:
    if value is None:
        return None
    return [float(item) for item in value]


@dataclass
class ReservoirBuildResult:
    candidates: list[CandidateRecord]
    sources: list[SourceRecord]
    raw_llm_text: str = ""
    parsed_llm_json: dict[str, Any] = field(default_factory=dict)
    llm_attempts: list[dict[str, Any]] = field(default_factory=list)
    drop_counts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
