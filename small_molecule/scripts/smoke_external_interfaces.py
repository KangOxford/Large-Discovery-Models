"""Smoke test for external SDK and JSON acquisition interfaces."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import bo_api  # noqa: E402
from strbo_v1.external_interfaces import evaluate_acquisition  # noqa: E402


def main() -> None:
    request = {
        "history": [
            {"smiles": "CCO", "score": -3.0},
            {"smiles": "CCN", "score": -2.4},
            {"smiles": "CCC", "score": -2.0},
        ],
        "query_smiles": ["CCCO", "CCCN"],
        "acquisition": ["ei", "pi", "ucb"],
        "minimize": True,
        "gp_fit_itersteps": 2,
        "gp_fp_n_bits": 128,
    }
    sdk = evaluate_acquisition(request=request, gp_device="cpu")
    api = json.loads(bo_api.evaluate_acquisition_json(json.dumps(request), gp_device="cpu"))
    assert sdk["ok"] is True
    assert api["ok"] is True
    assert sdk["items"][0]["details"]["mean"] == api["items"][0]["details"]["mean"]
    print(json.dumps(api, indent=2))


if __name__ == "__main__":
    main()
