"""Smoke test for the reusable AcquisitionEvaluator interface."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from strbo_v1.bayesian_analog_search import (
    AcquisitionEvaluator,
    BayesianAnalogSearchConfig,
)
from strbo_v1.gp import GPConfig


def main() -> None:
    history = [
        ("CCO", -3.0),
        ("CCN", -2.4),
        ("CCC", -2.0),
        ("CCCC", -1.7),
    ]
    config = BayesianAnalogSearchConfig(
        acquisition=("ei", "pi", "ucb"),
        minimize=True,
        gp_config=GPConfig(
            device="cpu",
            fit_n_itersteps=2,
            fp_n_bits=128,
        ),
    )
    evaluator = AcquisitionEvaluator(history, config)
    first = evaluator(["CCCO", "CCCN"])
    second = evaluator(["CCCl"])
    expected_keys = {
        "mean",
        "std",
        "variance",
        "acquisition_ei",
        "acquisition_pi",
        "acquisition_ucb",
    }
    assert set(first["CCCO"]) == expected_keys
    assert "CCCl" in second
    print(first)
    print(second)


if __name__ == "__main__":
    main()
