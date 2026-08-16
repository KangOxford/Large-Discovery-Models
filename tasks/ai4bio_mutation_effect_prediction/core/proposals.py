"""Reservoir expansion for bounded predictor architecture specifications."""

from __future__ import annotations

import json
from collections.abc import Callable
from itertools import product
from typing import Any

from ldm_tts.contracts import RawProposal
from ldm_tts.engine.expansion import ExpansionRequest, ExpansionResult
from ldm_tts.transport import ProposalClient, ProposalRequest
from ldm_tts.transport.parsing import load_json_object

from tasks.ai4bio_mutation_effect_prediction.core.candidate import (
    ACTIVATIONS,
    FEATURE_MODES,
    canonical_spec_key,
    normalize_predictor_spec,
)


def _spec_space() -> tuple[dict[str, Any], ...]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    hidden_options = ([], [128], [256], [512], [256, 128], [512, 128])
    for feature_mode, hidden_dims, activation, layer_norm in product(
        FEATURE_MODES,
        hidden_options,
        ACTIVATIONS,
        (False, True),
    ):
        spec = normalize_predictor_spec(
            {
                "feature_mode": feature_mode,
                "hidden_dims": list(hidden_dims),
                "activation": activation,
                "dropout": 0.0 if not hidden_dims else 0.1,
                "layer_norm": bool(layer_norm and hidden_dims),
                "learning_rate": 0.001,
                "weight_decay": 0.05,
            }
        )
        key = canonical_spec_key(spec)
        if key not in seen:
            seen.add(key)
            items.append(spec)
    return tuple(items)


SPEC_SPACE = _spec_space()


def predictor_spec_schema(candidate_count: int) -> dict[str, Any]:
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    item = {
        "type": "object",
        "required": [
            "feature_mode",
            "hidden_dims",
            "activation",
            "dropout",
            "layer_norm",
            "learning_rate",
            "weight_decay",
        ],
        "properties": {
            "feature_mode": {"type": "string", "enum": list(FEATURE_MODES)},
            "hidden_dims": {
                "type": "array",
                "maxItems": 3,
                "items": {"type": "integer", "minimum": 16, "maximum": 1024},
            },
            "activation": {"type": "string", "enum": list(ACTIVATIONS)},
            "dropout": {"type": "number", "minimum": 0.0, "maximum": 0.5},
            "layer_norm": {"type": "boolean"},
            "learning_rate": {"type": "number", "minimum": 1.0e-5, "maximum": 1.0e-2},
            "weight_decay": {"type": "number", "minimum": 0.0, "maximum": 0.2},
        },
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": candidate_count,
                "maxItems": candidate_count,
                "items": item,
            }
        },
        "additionalProperties": False,
    }


def proposal_response_format(candidate_count: int) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "mutation_predictor_specs",
            "strict": True,
            "schema": predictor_spec_schema(candidate_count),
        },
    }


def parse_predictor_specs(text: str, *, expected_count: int) -> list[dict[str, Any]]:
    payload = load_json_object(text)
    if set(payload) != {"candidates"}:
        raise ValueError("proposal response must contain only the candidates field")
    candidates = payload["candidates"]
    if not isinstance(candidates, list) or len(candidates) != expected_count:
        raise ValueError(f"proposal response must contain exactly {expected_count} candidates")
    normalized = [normalize_predictor_spec(item) for item in candidates]
    if len({canonical_spec_key(item) for item in normalized}) != len(normalized):
        raise ValueError("proposal response candidates must be distinct")
    return normalized


class DeterministicPredictorExpander:
    def __init__(self, *, collectable: bool = True) -> None:
        self.collectable = bool(collectable)

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        occupied = {item.candidate.canonical_key for item in request.observations}
        proposals: list[RawProposal] = []
        start = request.round_idx * request.reservoir_size
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


class EndpointPredictorExpander:
    def __init__(
        self,
        client: ProposalClient,
        *,
        before_request: Callable[[], None] | None = None,
    ) -> None:
        self.client = client
        self.before_request = before_request

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        history = [
            {
                "spec": item.candidate.payload.get("spec"),
                "metrics": dict(item.metrics),
            }
            for item in request.observations[-8:]
        ]
        if self.before_request is not None:
            self.before_request()
        response = self.client.propose(
            ProposalRequest(
                messages=(
                    {
                        "role": "system",
                        "content": (
                            "You design parameter-efficient supervised protein mutation "
                            "predictors over frozen ESM-2 embeddings."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Return exactly {request.reservoir_size} distinct JSON predictor "
                            "specs that follow the response schema. Explore useful combinations "
                            "of mutant, delta, or concatenated features while respecting the "
                            "parameter budget. Return no code, markdown, or prose. Observed "
                            f"history: {json.dumps(history, sort_keys=True)}"
                        ),
                    },
                ),
                metadata={"round_idx": request.round_idx},
            )
        )
        specs = parse_predictor_specs(response.text, expected_count=request.reservoir_size)
        proposals = tuple(
            RawProposal(
                spec,
                "openai_predictor_spec",
                {"collectable": True, "round_idx": request.round_idx},
            )
            for spec in specs
        )
        return ExpansionResult(
            proposals=proposals,
            attempts=(response,),
            metadata={"mode": "openai", "round": request.round_idx},
        )
