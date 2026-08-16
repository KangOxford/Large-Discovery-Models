"""Versioned architecture representation used by GP-UCB selection."""

from __future__ import annotations

import math

from ldm_tts.contracts import Candidate, SurrogateSpaceSpec
from ldm_tts.optimization.records import SurrogateVector

from tasks.ai4bio_mutation_effect_prediction.core.candidate import (
    PARAMETER_LIMIT,
    predictor_parameter_count,
)


FEATURE_VERSION = "mutation_predictor_spec_v1"
FEATURE_DIMENSION = 15


class PredictorSpecEncoder:
    def describe(self) -> SurrogateSpaceSpec:
        return SurrogateSpaceSpec(
            kind="vector",
            representation=(
                "15 normalized architecture, optimizer, feature-choice, and parameter-budget "
                "features derived from the admitted predictor specification."
            ),
            dimension_policy="fixed",
            dimension=FEATURE_DIMENSION,
            encoder=(
                "tasks.ai4bio_mutation_effect_prediction.core.surrogate:"
                "PredictorSpecEncoder"
            ),
            version=FEATURE_VERSION,
        )

    def encode(self, candidate: Candidate) -> SurrogateVector:
        spec = candidate.payload["spec"]
        feature_mode = spec["feature_mode"]
        activation = spec["activation"]
        hidden = list(spec["hidden_dims"]) + [0, 0, 0]
        values = (
            float(feature_mode == "embedding"),
            float(feature_mode == "delta"),
            float(feature_mode == "concat"),
            len(spec["hidden_dims"]) / 3.0,
            hidden[0] / 1024.0,
            hidden[1] / 1024.0,
            hidden[2] / 1024.0,
            float(activation == "relu"),
            float(activation == "gelu"),
            float(activation == "silu"),
            float(spec["dropout"]) / 0.5,
            float(spec["layer_norm"]),
            (math.log10(float(spec["learning_rate"])) + 5.0) / 3.0,
            float(spec["weight_decay"]) / 0.2,
            predictor_parameter_count(spec) / PARAMETER_LIMIT,
        )
        return SurrogateVector(
            values=tuple(float(value) for value in values),
            version=FEATURE_VERSION,
            source_id=candidate.candidate_id,
            metadata={"encoder": "normalized_predictor_spec"},
        )
