#!/usr/bin/env python3
"""Run the exact ridge seed through the official three-assay evaluator."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ldm_tts.contracts import Candidate
from ldm_tts.engine.run_store import atomic_json_write
from tasks.ai4bio_mutation_effect_prediction.core.evaluator import (
    MLSBenchMutationEvaluator,
    OFFICIAL_COMMIT,
    TASK_PATH,
)


RIDGE_SEED = {
    "feature_mode": "delta",
    "hidden_dims": [],
    "activation": "relu",
    "dropout": 0.0,
    "layer_norm": False,
    "learning_rate": 0.001,
    "weight_decay": 0.05,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cv-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--evaluation-timeout", type=float, default=3540.0)
    args = parser.parse_args()
    ridge_path = args.upstream_root / TASK_PATH / "edits" / "ridge.edit.py"
    module_spec = importlib.util.spec_from_file_location("_official_ridge_seed", ridge_path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"could not load official ridge seed: {ridge_path}")
    ridge_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(ridge_module)
    ridge_source = str(ridge_module._MODEL)
    ridge_digest = hashlib.sha256(ridge_path.read_bytes()).hexdigest()
    candidate = Candidate(
        candidate_id="official-ridge-seed",
        payload={
            "spec": RIDGE_SEED,
            "code": ridge_source,
            "config_overrides": {"learning_rate": 0.001, "weight_decay": 0.05},
        },
        canonical_key=hashlib.sha256(ridge_source.encode("utf-8")).hexdigest(),
        source="pinned_upstream_ridge",
        metadata={"source_path": str(ridge_path), "source_sha256": ridge_digest},
    )
    result = MLSBenchMutationEvaluator(
        upstream_root=args.upstream_root,
        data_dir=args.data_dir,
        cv_dir=args.cv_dir,
        run_dir=args.out_dir,
        timeout_seconds=args.evaluation_timeout,
    ).evaluate(candidate)
    payload = {
        "schema_version": 1,
        "stage": "seed_evaluated",
        "official": True,
        "outside_campaign_budget": True,
        "source_commit": OFFICIAL_COMMIT,
        "source_candidate": "ridge",
        "source_path": str(ridge_path),
        "source_sha256": ridge_digest,
        "candidate_id": candidate.candidate_id,
        "status": result.status,
        "metrics": dict(result.metrics),
        "artifacts": dict(result.artifacts),
        "resource_usage": dict(result.resource_usage),
        "error": result.error,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_write(args.out_dir / "qualification_seed.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 0 if result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
