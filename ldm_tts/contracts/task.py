"""Shared descriptions of LDM task spaces.

These dataclasses are intentionally dependency-light. They distinguish the
scientific candidate domain, finite reservoir expansion, surrogate
representation, LLM response contract, objectives, proposal-search topology,
and acquisition rule without importing task-specific dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


RESERVOIR_ACTION_KINDS = frozenset(
    {
        "emit_candidate",
        "edit_candidate",
        "configure_generator",
        "update_expansion_schema",
    }
)
SURROGATE_KINDS = frozenset({"none", "vector", "kernel"})
SURROGATE_DIMENSION_POLICIES = frozenset({"none", "fixed", "implicit", "evolving"})


@dataclass(frozen=True)
class ObjectiveSpec:
    """One measured objective used by a task."""

    name: str
    direction: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateDomainSpec:
    """Scientific candidate domain visible to LDM-TTS.

    ``dimension`` describes the candidate itself, not the surrogate
    representation used by an optimizer.
    """

    name: str
    kind: str
    dimension: int | None
    representation: str = ""
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ResponseSpaceSpec:
    """LLM output contract for one proposal mode."""

    name: str
    output_kind: str
    schema: dict[str, Any] = field(default_factory=dict)
    parser: str = ""
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReservoirExpansionSpec:
    """One LDM action that expands a reservoir or its expansion schema."""

    name: str
    action_kind: str
    response_space: str
    produces_candidates: bool
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("reservoir expansion name must not be empty")
        if self.action_kind not in RESERVOIR_ACTION_KINDS:
            raise ValueError(
                f"unknown reservoir action kind {self.action_kind!r}; "
                f"expected one of {sorted(RESERVOIR_ACTION_KINDS)}"
            )
        if not self.response_space.strip():
            raise ValueError("reservoir expansion response_space must not be empty")


@dataclass(frozen=True)
class ReservoirSpec:
    """Finite candidate reservoir and the actions that can expand it."""

    name: str
    expansions: tuple[ReservoirExpansionSpec, ...]
    candidate_validator: str
    deduplication_key: str
    max_size: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("reservoir name must not be empty")
        if not self.expansions:
            raise ValueError("reservoir must define at least one expansion action")
        names = [item.name for item in self.expansions]
        if len(names) != len(set(names)):
            raise ValueError("reservoir expansion names must be unique")
        if not any(item.produces_candidates for item in self.expansions):
            raise ValueError("reservoir must have an expansion action that produces candidates")
        if not self.candidate_validator.strip():
            raise ValueError("reservoir candidate_validator must not be empty")
        if not self.deduplication_key.strip():
            raise ValueError("reservoir deduplication_key must not be empty")
        if self.max_size is not None and self.max_size < 1:
            raise ValueError("reservoir max_size must be positive when provided")


@dataclass(frozen=True)
class SurrogateSpaceSpec:
    """Representation of candidates consumed by a surrogate and acquisition."""

    kind: str
    representation: str
    dimension_policy: str
    dimension: int | None = None
    encoder: str = ""
    version: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in SURROGATE_KINDS:
            raise ValueError(
                f"unknown surrogate kind {self.kind!r}; expected one of {sorted(SURROGATE_KINDS)}"
            )
        if self.dimension_policy not in SURROGATE_DIMENSION_POLICIES:
            raise ValueError(
                f"unknown surrogate dimension policy {self.dimension_policy!r}; "
                f"expected one of {sorted(SURROGATE_DIMENSION_POLICIES)}"
            )
        if not self.representation.strip():
            raise ValueError("surrogate representation must not be empty")
        if self.dimension_policy in {"fixed", "evolving"}:
            if self.dimension is None or self.dimension < 1:
                raise ValueError(
                    f"{self.dimension_policy} surrogate representations require a positive dimension"
                )
        elif self.dimension is not None:
            raise ValueError(
                f"{self.dimension_policy} surrogate representations must not declare a dimension"
            )
        if self.kind == "none" and self.dimension_policy != "none":
            raise ValueError("a disabled surrogate must use dimension_policy='none'")
        if self.kind != "none" and self.dimension_policy == "none":
            raise ValueError("an enabled surrogate must declare a non-'none' dimension policy")


@dataclass(frozen=True)
class AcquisitionSpec:
    """BO/acquisition rule that ranks or samples candidates."""

    name: str
    objective_names: tuple[str, ...]
    score_direction: str
    selection_rule: str
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProposalSearchSpec:
    """Topology used to traverse LLM proposal states within one search round."""

    name: str = "single_turn"
    breadth: int = 1
    depth: int = 1
    beam_width: int = 1
    evaluation_policy: str = "terminal"
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LDMTaskSpec:
    """Complete semantic description for one task workflow."""

    task: str
    candidate_domain: CandidateDomainSpec
    objectives: tuple[ObjectiveSpec, ...]
    response_spaces: tuple[ResponseSpaceSpec, ...]
    acquisition: AcquisitionSpec
    reservoir: ReservoirSpec
    surrogate: SurrogateSpaceSpec
    proposal_search: ProposalSearchSpec = field(default_factory=ProposalSearchSpec)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        response_names = [item.name for item in self.response_spaces]
        if len(response_names) != len(set(response_names)):
            raise ValueError("response space names must be unique")
        unknown = sorted(
            {
                expansion.response_space
                for expansion in self.reservoir.expansions
                if expansion.response_space not in response_names
            }
        )
        if unknown:
            raise ValueError(
                "reservoir expansion references unknown response space(s): "
                + ", ".join(unknown)
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


__all__ = [
    "AcquisitionSpec",
    "CandidateDomainSpec",
    "LDMTaskSpec",
    "ObjectiveSpec",
    "ProposalSearchSpec",
    "RESERVOIR_ACTION_KINDS",
    "ReservoirExpansionSpec",
    "ReservoirSpec",
    "ResponseSpaceSpec",
    "SURROGATE_DIMENSION_POLICIES",
    "SURROGATE_KINDS",
    "SurrogateSpaceSpec",
]
