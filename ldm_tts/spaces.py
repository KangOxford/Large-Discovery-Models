"""Shared descriptions of LDM task spaces.

These dataclasses are intentionally lightweight. They describe the shape of a
task's candidate domain, LLM response contract, objectives, proposal-search
topology, and acquisition rule without importing task-specific dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ObjectiveSpec:
    """One measured objective used by a task."""

    name: str
    direction: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateSpaceSpec:
    """Candidate representation visible to LDM-TTS."""

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
    candidate_space: CandidateSpaceSpec
    objectives: tuple[ObjectiveSpec, ...]
    response_spaces: tuple[ResponseSpaceSpec, ...]
    acquisition: AcquisitionSpec
    proposal_search: ProposalSearchSpec = field(default_factory=ProposalSearchSpec)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)
