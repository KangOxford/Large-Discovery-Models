"""Coverage for the acquisition-score aggregation switch (max / mean)."""

import pytest

from ldm_rl.env import ACQUISITION_AGGS, EnvConfig
from ldm_rl.episodes import EpisodeSpec


def test_default_agg_is_max():
    assert EnvConfig(iterations=1, reward="acquisition").acquisition_agg == "max"


@pytest.mark.parametrize("agg", ACQUISITION_AGGS)
def test_valid_aggs_accepted(agg):
    assert EnvConfig(iterations=1, acquisition_agg=agg).acquisition_agg == agg


def test_invalid_agg_rejected():
    with pytest.raises(ValueError):
        EnvConfig(iterations=1, acquisition_agg="median")


def test_agg_threads_through_episode_spec_and_json():
    spec = EpisodeSpec(
        task="small_molecule",
        mode="real",
        reward="acquisition",
        acquisition_agg="mean",
        real={"gp_history_file": "/tmp/gp.jsonl"},
    )
    assert spec.to_env_config().acquisition_agg == "mean"
    assert EpisodeSpec.from_json(spec.to_json()).acquisition_agg == "mean"


def test_aggregation_math():
    scores = [0.2, 0.8, 0.5]
    assert max(scores) == pytest.approx(0.8)
    assert sum(scores) / len(scores) == pytest.approx(0.5)
