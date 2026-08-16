"""Deterministic finite reservoir expansion."""

from __future__ import annotations

from itertools import product

from ldm_tts.contracts import RawProposal
from ldm_tts.engine.expansion import ExpansionRequest, ExpansionResult

from tasks.causal_discovery_discrete.core.candidate import canonical_spec_key


THRESHOLDS = tuple(round(0.0025 * index, 4) for index in range(1, 81))
MAX_DEGREES = (2, 3, 4, 5, 6, 8, 10, 12)
SPEC_SPACE = tuple(
    {"min_association": threshold, "max_degree": degree}
    for degree, threshold in product(MAX_DEGREES, THRESHOLDS)
)


def proposal_schema(candidate_count: int) -> dict:
    return {
        "type": "object",
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": candidate_count,
                "maxItems": candidate_count,
                "items": {
                    "type": "object",
                    "required": ["min_association", "max_degree"],
                    "properties": {
                        "min_association": {"type": "number", "minimum": 0, "maximum": 1},
                        "max_degree": {"type": "integer", "minimum": 1, "maximum": 20},
                    },
                    "additionalProperties": False,
                },
            }
        },
        "additionalProperties": False,
    }


class DeterministicCausalExpander:
    def __init__(self, *, collectable: bool = True) -> None:
        self.collectable = bool(collectable)

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        occupied = {item.candidate.canonical_key for item in request.observations}
        start = request.round_idx * request.reservoir_size
        proposals = []
        for offset in range(len(SPEC_SPACE)):
            spec = SPEC_SPACE[(start + offset) % len(SPEC_SPACE)]
            if canonical_spec_key(spec) in occupied:
                continue
            proposals.append(
                RawProposal(
                    dict(spec),
                    "deterministic_catalog",
                    {"collectable": self.collectable, "variant": start + offset},
                )
            )
            if len(proposals) == request.reservoir_size:
                break
        return ExpansionResult(
            proposals=tuple(proposals),
            metadata={"mode": "deterministic", "round": request.round_idx},
        )
