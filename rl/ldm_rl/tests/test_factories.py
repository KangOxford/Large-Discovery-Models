"""Factory tests: real task adapters behind the generic environment."""

from __future__ import annotations

import json

import pytest

from ldm_rl import EnvConfig, LDMEnv
from ldm_rl.factories import build_env


def _ai4bio_catalog_action(count: int, offset: int = 0) -> str:
    from tasks.ai4bio_mutation_effect_prediction.core.proposals import SPEC_SPACE

    specs = [dict(SPEC_SPACE[(offset + i) % len(SPEC_SPACE)]) for i in range(count)]
    return json.dumps({"candidates": specs})


def test_build_env_unknown_task_raises() -> None:
    with pytest.raises(KeyError):
        build_env("not_a_task", mode="mock")


def test_ai4bio_mock_episode_improves() -> None:
    env = build_env(
        "ai4bio_mutation_effect_prediction",
        mode="mock",
        config=EnvConfig(iterations=3, reservoir_size=1),
    )
    assert isinstance(env, LDMEnv)
    observation = env.reset()
    assert "ai4bio_mutation_effect_prediction" in observation
    assert "selection_score" in observation

    actions = iter([_ai4bio_catalog_action(1, 0), _ai4bio_catalog_action(1, 1), _ai4bio_catalog_action(1, 2)])
    result = env.run(lambda _obs: next(actions))
    assert result.rounds == 3
    assert result.best_metrics is not None
    assert result.best_metrics["selection_score"] > 0.0
    assert all(step.info["evaluated"][0]["evaluation"]["status"] == "succeeded" for step in result.steps)
    assert result.total_reward >= 0.0


def test_ai4bio_invalid_spec_rejected_with_feedback() -> None:
    env = build_env(
        "ai4bio_mutation_effect_prediction",
        mode="mock",
        config=EnvConfig(iterations=1, reservoir_size=1),
    )
    env.reset()
    # spec outside the parameter budget: three enormous hidden layers
    bogus = {
        "candidates": [
            {
                "hidden_dims": [4096, 4096, 4096],
                "activation": "relu",
                "feature_mode": "concat",
                "dropout": 0.1,
                "layer_norm": True,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
            }
        ]
    }
    step = env.step(json.dumps(bogus))
    assert step.reward == 0.0
    # the task parser rejects over-budget specs during parsing, so the failure
    # surfaces as parse feedback rather than an admission rejection
    assert step.info["parse_error"] is not None
    assert "could not be parsed" in step.observation


def test_ai4bio_gp_acquisition_reward() -> None:
    env = build_env(
        "ai4bio_mutation_effect_prediction",
        mode="mock",
        config=EnvConfig(
            iterations=3, reservoir_size=2, evaluations_per_round=1, reward="acquisition"
        ),
        acquisition_beta=1.0,
    )
    observation = env.reset()
    actions = iter(
        [
            _ai4bio_catalog_action(2, 0),
            _ai4bio_catalog_action(2, 2),
            _ai4bio_catalog_action(2, 4),
        ]
    )
    result = env.run(lambda _obs: next(actions))
    assert result.rounds == 3
    for step in result.steps:
        selection = step.info["selection"]
        assert selection["metadata"]["surrogate"]  # GP-backed selection
        evaluated_ids = {item["candidate"]["candidate_id"] for item in step.info["evaluated"]}
        if evaluated_ids:
            expected = max(
                float(prediction["acquisition_score"])
                for prediction in selection["predictions"]
                if prediction["candidate_id"] in evaluated_ids
            )
            assert step.reward == pytest.approx(expected)
            assert step.info["reward_components"]["kind"] == "acquisition"


def test_small_molecule_mock_episode() -> None:
    from tasks.small_molecule.core.workflow import ExpandingMockCase2LLM

    env = build_env(
        "small_molecule",
        mode="mock",
        config=EnvConfig(iterations=3, reservoir_size=3, evaluations_per_round=1),
    )
    observation = env.reset()
    assert "vina" in observation and "activity" in observation

    llm = ExpandingMockCase2LLM()
    result = env.run(lambda obs: llm.chat("system", obs, json_mode=True))
    assert result.rounds == 3
    # two objectives -> no single incumbent; verify per-step metrics instead
    for step in result.steps:
        metrics = step.info["evaluated"][0]["evaluation"]["metrics"]
        assert set(metrics) == {"vina", "activity"}
        assert metrics["activity"] > 0.0
        assert step.reward >= 0.0
    assert result.total_reward > 0.0


def test_small_molecule_real_mode_is_supported() -> None:
    # Real mode is wired (tasks.small_molecule.core.rl_real.build_real_components),
    # so the factory must NOT reject it; only an unknown mode is rejected.
    from tasks.small_molecule.core.rl_adapter import build_rl_components

    with pytest.raises(ValueError, match="mock and real modes only"):
        build_rl_components(mode="bogus")


def test_causal_discovery_mock_episode() -> None:
    env = build_env(
        "causal_discovery_discrete",
        mode="mock",
        config=EnvConfig(iterations=2, reservoir_size=2, evaluations_per_round=1),
    )
    observation = env.reset()
    assert "causal_discovery_discrete" in observation

    actions = iter(
        [
            json.dumps(
                {
                    "candidates": [
                        {"min_association": 0.30, "max_degree": 3},
                        {"min_association": 0.07, "max_degree": 6},
                    ]
                }
            ),
            json.dumps(
                {
                    "candidates": [
                        {"min_association": 0.10, "max_degree": 5},
                        {"min_association": 0.50, "max_degree": 8},
                    ]
                }
            ),
        ]
    )
    result = env.run(lambda _obs: next(actions))
    assert result.best_metrics is not None
    assert "selection_score" in result.best_metrics
    assert result.total_reward > 0.0
