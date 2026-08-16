"""Candidate validation and canonicalization for discrete causal discovery."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ldm_tts.contracts import Candidate, CandidateRejection, RawProposal
from ldm_tts.data import DataCollectionSink, make_complete_design_ir


SPEC_KEYS = frozenset({"min_association", "max_degree"})


def normalize_algorithm_spec(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping) and set(payload) == {"spec"}:
        payload = payload["spec"]
    if not isinstance(payload, Mapping):
        raise ValueError("candidate must be an algorithm specification object")
    unknown = sorted(str(key) for key in payload if key not in SPEC_KEYS)
    missing = sorted(SPEC_KEYS - set(payload))
    if unknown:
        raise ValueError("candidate has unknown field(s): " + ", ".join(unknown))
    if missing:
        raise ValueError("candidate is missing field(s): " + ", ".join(missing))

    threshold = payload["min_association"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("min_association must be numeric")
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("min_association must be finite and in [0, 1]")
    max_degree = payload["max_degree"]
    if isinstance(max_degree, bool) or not isinstance(max_degree, int):
        raise ValueError("max_degree must be an integer")
    if not 1 <= max_degree <= 20:
        raise ValueError("max_degree must be in [1, 20]")
    return {"min_association": threshold, "max_degree": int(max_degree)}


def canonical_spec_key(spec: Mapping[str, Any]) -> str:
    raw = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass
class CausalAlgorithmCandidateDomain:
    sink: DataCollectionSink = field(default_factory=DataCollectionSink.disabled)

    def admit(self, proposal: RawProposal) -> Candidate | CandidateRejection:
        try:
            spec = normalize_algorithm_spec(proposal.payload)
            key = canonical_spec_key(spec)
        except ValueError as exc:
            return CandidateRejection("invalid_algorithm", str(exc), proposal.source)
        candidate = Candidate(
            candidate_id=f"causal-{key[:12]}",
            payload={"spec": spec},
            canonical_key=key,
            source=proposal.source,
        )
        if bool(proposal.metadata.get("collectable")):
            ir = make_complete_design_ir(
                task_id="causal_discovery_discrete",
                domain="bounded discrete causal-discovery algorithm specification",
                task_description=(
                    "Recover a CPDAG from integer-coded observational data using a "
                    "sparse pairwise normalized-mutual-information skeleton."
                ),
                objectives=[{
                    "name": "selection_score",
                    "direction": "maximize",
                    "description": "Qualification-only synthetic selection signal.",
                }],
                observations=[],
                candidates=[spec],
                design_space_description=(
                    "A normalized-mutual-information cutoff and a hard per-node "
                    "degree cap; arbitrary source code is not admitted."
                ),
                request_description="Propose one bounded causal algorithm spec.",
                num_candidates=1,
                reasoning_available=False,
            )
            self.sink.append(
                ir,
                provenance={"candidate_id": candidate.candidate_id, "source": proposal.source},
            )
        return candidate
