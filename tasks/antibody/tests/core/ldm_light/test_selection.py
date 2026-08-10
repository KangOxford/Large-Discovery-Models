from __future__ import annotations

import numpy as np
import pytest

from tasks.antibody.core.ldm_light.selection import (
    acquisition_probabilities,
    select_by_acquisition,
)


def test_softmax_probabilities_are_stable_and_normalized():
    probabilities = acquisition_probabilities(
        [10_000.0, 10_001.0], reduction="softmax", eta=1.0,
    )

    expected = np.exp([0.0, 1.0]) / np.exp([0.0, 1.0]).sum()
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities == pytest.approx(expected)


def test_infinite_eta_is_explicit_deterministic_max():
    selected, probabilities = select_by_acquisition(
        [0.1, 3.0, 1.0],
        batch_size=1,
        reduction="softmax",
        eta=float("inf"),
        rng=np.random.default_rng(4),
    )

    assert selected == [1]
    assert probabilities == [0.0, 1.0, 0.0]


def test_max_ties_use_first_stable_candidate_in_selection_and_probabilities():
    selected, probabilities = select_by_acquisition(
        [2.0, 2.0, 1.0],
        batch_size=1,
        reduction="max",
        eta=1.0,
        rng=np.random.default_rng(4),
    )

    assert selected == [0]
    assert probabilities == [1.0, 0.0, 0.0]


def test_softmax_selection_is_seeded_and_without_replacement():
    kwargs = {
        "scores": [0.0, 0.5, 1.0],
        "batch_size": 2,
        "reduction": "softmax",
        "eta": 0.7,
    }
    first, _ = select_by_acquisition(rng=np.random.default_rng(9), **kwargs)
    second, _ = select_by_acquisition(rng=np.random.default_rng(9), **kwargs)

    assert first == second
    assert len(set(first)) == 2


@pytest.mark.parametrize("eta", [-1.0, float("nan")])
def test_invalid_eta_is_rejected(eta):
    with pytest.raises(ValueError, match="eta"):
        acquisition_probabilities([1.0], reduction="softmax", eta=eta)
