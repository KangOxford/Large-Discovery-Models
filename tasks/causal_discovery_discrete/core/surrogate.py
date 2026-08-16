"""Fixed surrogate representation for causal algorithm specifications."""

from __future__ import annotations

from ldm_tts.contracts import Candidate, SurrogateSpaceSpec
from ldm_tts.optimization.records import SurrogateVector


FEATURE_VERSION = "causal_mi_skeleton_v1"
FEATURE_DIMENSION = 2


class CausalSpecEncoder:
    def describe(self) -> SurrogateSpaceSpec:
        return SurrogateSpaceSpec(
            kind="vector",
            representation="Normalized mutual-information threshold and degree cap.",
            dimension_policy="fixed",
            dimension=FEATURE_DIMENSION,
            encoder="tasks.causal_discovery_discrete.core.surrogate:CausalSpecEncoder",
            version=FEATURE_VERSION,
        )

    def encode(self, candidate: Candidate) -> SurrogateVector:
        spec = candidate.payload["spec"]
        return SurrogateVector(
            values=(float(spec["min_association"]), float(spec["max_degree"]) / 20.0),
            version=FEATURE_VERSION,
            source_id=candidate.candidate_id,
            metadata={"encoder": "bounded_causal_algorithm_spec"},
        )
