"""Candidate parsing, validation, materialization, and collection."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ldm_tts.contracts import Candidate, CandidateRejection, RawProposal
from ldm_tts.data import DataCollectionSink, make_complete_design_ir


EMBED_DIM = 1280
PARAMETER_LIMIT = 6_957_956
FEATURE_MODES = ("embedding", "delta", "concat")
ACTIVATIONS = ("relu", "gelu", "silu")
SPEC_KEYS = frozenset(
    {
        "feature_mode",
        "hidden_dims",
        "activation",
        "dropout",
        "layer_norm",
        "learning_rate",
        "weight_decay",
    }
)


def normalize_predictor_spec(payload: Any) -> dict[str, Any]:
    """Return a strict, JSON-canonical predictor architecture specification."""

    if isinstance(payload, Mapping) and set(payload) == {"spec"}:
        payload = payload["spec"]
    if not isinstance(payload, Mapping):
        raise ValueError("candidate must be a predictor specification object")
    unknown = sorted(str(key) for key in payload if key not in SPEC_KEYS)
    missing = sorted(SPEC_KEYS - set(payload))
    if unknown:
        raise ValueError("candidate has unknown field(s): " + ", ".join(unknown))
    if missing:
        raise ValueError("candidate is missing field(s): " + ", ".join(missing))

    feature_mode = payload["feature_mode"]
    activation = payload["activation"]
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"feature_mode must be one of {FEATURE_MODES}")
    if activation not in ACTIVATIONS:
        raise ValueError(f"activation must be one of {ACTIVATIONS}")

    raw_dims = payload["hidden_dims"]
    if not isinstance(raw_dims, (list, tuple)):
        raise ValueError("hidden_dims must be a list")
    if len(raw_dims) > 3:
        raise ValueError("hidden_dims may contain at most three layers")
    hidden_dims: list[int] = []
    for value in raw_dims:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("hidden_dims entries must be integers")
        if value < 16 or value > 1024 or value % 8:
            raise ValueError("hidden_dims entries must be multiples of 8 in [16, 1024]")
        hidden_dims.append(int(value))

    dropout = _finite_number(payload["dropout"], "dropout")
    learning_rate = _finite_number(payload["learning_rate"], "learning_rate")
    weight_decay = _finite_number(payload["weight_decay"], "weight_decay")
    layer_norm = payload["layer_norm"]
    if not isinstance(layer_norm, bool):
        raise ValueError("layer_norm must be a boolean")
    if not 0.0 <= dropout <= 0.5:
        raise ValueError("dropout must be in [0.0, 0.5]")
    if not 1.0e-5 <= learning_rate <= 1.0e-2:
        raise ValueError("learning_rate must be in [1e-5, 1e-2]")
    if not 0.0 <= weight_decay <= 0.2:
        raise ValueError("weight_decay must be in [0.0, 0.2]")

    normalized = {
        "feature_mode": str(feature_mode),
        "hidden_dims": hidden_dims,
        "activation": str(activation),
        "dropout": float(dropout),
        "layer_norm": layer_norm,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
    }
    count = predictor_parameter_count(normalized)
    if count > PARAMETER_LIMIT:
        raise ValueError(
            f"candidate has {count} parameters; official limit is {PARAMETER_LIMIT}"
        )
    return normalized


def predictor_parameter_count(spec: Mapping[str, Any]) -> int:
    """Count trainable parameters in the materialized sequential head."""

    input_dim = EMBED_DIM * (2 if spec["feature_mode"] == "concat" else 1)
    hidden_dims = tuple(int(value) for value in spec["hidden_dims"])
    dimensions = (input_dim, *hidden_dims, 1)
    total = sum(
        in_dim * out_dim + out_dim
        for in_dim, out_dim in zip(dimensions, dimensions[1:])
    )
    if bool(spec["layer_norm"]):
        total += sum(2 * hidden_dim for hidden_dim in hidden_dims)
    return int(total)


def canonical_spec_key(spec: Mapping[str, Any]) -> str:
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def render_predictor_source(spec: Mapping[str, Any]) -> str:
    """Materialize one benchmark-compatible ``MutationPredictor`` class."""

    normalized = normalize_predictor_spec(spec)
    feature_mode = normalized["feature_mode"]
    input_dim = EMBED_DIM * (2 if feature_mode == "concat" else 1)
    modules: list[str] = []
    previous = input_dim
    activation = {
        "relu": "nn.ReLU()",
        "gelu": "nn.GELU()",
        "silu": "nn.SiLU()",
    }[normalized["activation"]]
    for hidden in normalized["hidden_dims"]:
        modules.append(f"nn.Linear({previous}, {hidden})")
        if normalized["layer_norm"]:
            modules.append(f"nn.LayerNorm({hidden})")
        modules.append(activation)
        if normalized["dropout"]:
            modules.append(f"nn.Dropout({normalized['dropout']!r})")
        previous = hidden
    modules.append(f"nn.Linear({previous}, 1)")
    module_lines = ",\n            ".join(modules)
    input_line = {
        "embedding": "x = embedding",
        "delta": "x = delta_embedding",
        "concat": "x = torch.cat([embedding, delta_embedding], dim=-1)",
    }[feature_mode]
    return (
        "class MutationPredictor(nn.Module):\n"
        "    def __init__(self, embed_dim: int = EMBED_DIM):\n"
        "        super().__init__()\n"
        "        if embed_dim != EMBED_DIM:\n"
        "            raise ValueError(f'expected embed_dim={EMBED_DIM}, got {embed_dim}')\n"
        "        self.network = nn.Sequential(\n"
        f"            {module_lines}\n"
        "        )\n\n"
        "    def forward(self, embedding, delta_embedding):\n"
        f"        {input_line}\n"
        "        return self.network(x).squeeze(-1)\n"
    )


@dataclass
class MutationPredictorCandidateDomain:
    """Admit bounded architecture specs and collect only accepted proposals."""

    sink: DataCollectionSink = field(default_factory=DataCollectionSink.disabled)

    def admit(self, proposal: RawProposal) -> Candidate | CandidateRejection:
        try:
            spec = normalize_predictor_spec(proposal.payload)
            key = canonical_spec_key(spec)
            source = render_predictor_source(spec)
            parameter_count = predictor_parameter_count(spec)
        except ValueError as exc:
            return CandidateRejection("invalid_predictor", str(exc), proposal.source)
        candidate = Candidate(
            candidate_id=f"predictor-{key[:12]}",
            payload={
                "spec": spec,
                "code": source,
                "config_overrides": {
                    "learning_rate": spec["learning_rate"],
                    "weight_decay": spec["weight_decay"],
                },
            },
            canonical_key=key,
            source=proposal.source,
            metadata={"parameter_count": parameter_count},
        )
        if bool(proposal.metadata.get("collectable")):
            ir = make_complete_design_ir(
                task_id="ai4bio_mutation_effect_prediction",
                domain="bounded MutationPredictor architecture specification",
                task_description=(
                    "Design a supervised head over mutant and mutant-minus-WT ESM-2 "
                    "embeddings for mutation-effect prediction."
                ),
                objectives=[
                    {
                        "name": "selection_score",
                        "direction": "maximize",
                        "description": "Qualification-only mock selection signal.",
                    }
                ],
                observations=[],
                candidates=[spec],
                design_space_description=(
                    "One to three bounded dense layers, a fixed activation family, "
                    "optional layer normalization, and bounded optimizer settings."
                ),
                request_description="Propose one parameter-budget-valid predictor spec.",
                num_candidates=1,
                reasoning_available=False,
            )
            self.sink.append(
                ir,
                provenance={
                    "candidate_id": candidate.candidate_id,
                    "source": proposal.source,
                },
            )
        return candidate


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
