#!/usr/bin/env python3
"""Verify materialization and CPU/CUDA tensor contracts for the real reservoir."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.engine.run_store import atomic_json_write
from tasks.ai4bio_mutation_effect_prediction.core.candidate import (
    MutationPredictorCandidateDomain,
    predictor_parameter_count,
)
from tasks.ai4bio_mutation_effect_prediction.core.evaluator import (
    TASK_PATH,
    fixed_template_sha256,
    materialize_harness,
    validate_predictor_contract,
    validate_upstream_contract,
)
from tasks.ai4bio_mutation_effect_prediction.core.proposals import (
    DeterministicPredictorExpander,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    task_dir = args.upstream_root / TASK_PATH
    digests = validate_upstream_contract(task_dir)
    template = (task_dir / "edits" / "custom_template.py").read_text(encoding="utf-8")
    template_fixed_digest = fixed_template_sha256(template)
    proposals = DeterministicPredictorExpander().expand(
        ExpansionRequest(round_idx=0, reservoir_size=4)
    ).proposals
    domain = MutationPredictorCandidateDomain()
    verified = []
    with tempfile.TemporaryDirectory(prefix="mutation-contract-") as raw_dir:
        directory = Path(raw_dir)
        for proposal in proposals:
            candidate = domain.admit(proposal)
            harness = materialize_harness(
                template,
                candidate.payload["code"],
                candidate.payload["config_overrides"],
            )
            if fixed_template_sha256(harness) != template_fixed_digest:
                raise ValueError("materialization changed fixed template regions")
            harness_path = directory / f"{candidate.candidate_id}.py"
            harness_path.write_text(harness, encoding="utf-8")
            contract = validate_predictor_contract(
                harness_path,
                expected_parameters=predictor_parameter_count(candidate.payload["spec"]),
            )
            verified.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "parameter_count": contract["parameter_count"],
                    "devices": contract["devices"],
                    "output_shape": contract["output_shape"],
                    "dtype": contract["dtype"],
                }
            )
    payload = {
        "schema_version": 1,
        "stage": "contract_verified",
        "status": "ok",
        "upstream_files_verified": len(digests),
        "fixed_template_sha256": template_fixed_digest,
        "candidates_verified": verified,
    }
    atomic_json_write(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
