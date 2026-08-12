from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ldm_tts.optimization.acquisition import (
    MULTI_OBJECTIVE_ACQUISITIONS,
    SINGLE_OBJECTIVE_ACQUISITIONS,
    make_acquisition,
)
from ldm_tts.cli.runner import apply_override, build_plan, load_config


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", sorted(SINGLE_OBJECTIVE_ACQUISITIONS))
def test_single_objective_acquisitions_share_one_interface(name: str) -> None:
    acquisition = make_acquisition(name, minimize=(True,), beta=2.0, xi=0.0)
    kwargs = {"best": 0.0} if name == "ei" else {}

    scores = np.asarray(
        acquisition.score(
            np.array([-1.0, 1.0]),
            np.array([0.2, 0.2]),
            **kwargs,
        )
    )

    assert scores.shape == (2,)
    assert scores[0] > scores[1]


def test_lcb_and_ucb_have_distinct_minimization_semantics() -> None:
    mean = np.array([1.0])
    std = np.array([0.5])

    lcb = make_acquisition("lcb", minimize=(True,), beta=2.0).score(mean, std)
    ucb = make_acquisition("ucb", minimize=(True,), beta=2.0).score(mean, std)

    np.testing.assert_allclose(lcb, np.array([0.0]))
    np.testing.assert_allclose(ucb, np.array([-2.0]))


def test_single_objective_interface_preserves_torch_tensors_when_available() -> None:
    torch = pytest.importorskip("torch")
    mean = torch.tensor([-1.0, 1.0])
    std = torch.tensor([0.2, 0.2])
    acquisition = make_acquisition("ei", minimize=(True,), xi=0.0)

    scores = acquisition.score(mean, std, best=torch.tensor(0.0))

    assert isinstance(scores, torch.Tensor)
    assert scores[0] > scores[1]


def test_weighted_multi_objective_mean_orients_each_objective() -> None:
    acquisition = make_acquisition(
        "mean",
        minimize=(True, False),
        weights=(0.25, 0.75),
    )

    scores = acquisition.score(
        [np.array([-4.0, -2.0]), np.array([0.2, 0.8])],
        [np.zeros(2), np.zeros(2)],
    )

    np.testing.assert_allclose(scores, np.array([1.15, 1.1]))


def test_ehvi_uses_the_same_interface_for_two_objective_posteriors() -> None:
    acquisition = make_acquisition(
        "ehvi",
        minimize=(True, False),
        n_samples=16,
    )

    scores = acquisition.score(
        [np.array([-8.0]), np.array([0.8])],
        [np.array([0.0]), np.array([0.0])],
        pareto_points=[(-7.0, 0.7)],
        ref_point=(0.0, 0.0),
        rng=7,
    )

    assert scores.shape == (1,)
    assert scores[0] > 0


def test_objective_count_rejects_task_incompatible_acquisition() -> None:
    assert MULTI_OBJECTIVE_ACQUISITIONS == {"ehvi", "mean"}
    with pytest.raises(ValueError, match="does not support 2 objectives"):
        make_acquisition("ei", minimize=(True, False))
    with pytest.raises(ValueError, match="does not support 1 objectives"):
        make_acquisition("ehvi", minimize=(True,))


@pytest.mark.parametrize(
    ("config_path", "override", "expected_flag", "expected_value"),
    [
        ("config/nanogpt/mock_best_of_n.yaml", "args.surrogate-mode=ucb", "--surrogate-mode", "ucb"),
        ("config/antibody/mock_ei.yaml", "args.acq=mean", "--acq", "mean"),
        ("config/small_molecule/mock_m1_stratified_oversample.yaml", "args.acq=mean", "--acq", "mean"),
    ],
)
def test_task_configs_expose_the_requested_acquisition_matrix(
    config_path: str,
    override: str,
    expected_flag: str,
    expected_value: str,
) -> None:
    path = REPO_ROOT / config_path
    config = load_config(path)
    apply_override(config, override)

    argv = build_plan(config, path)["argv"]

    flag_index = argv.index(expected_flag)
    assert argv[flag_index + 1] == expected_value
