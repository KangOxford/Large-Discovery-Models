"""Registry-alignment tests for the RL environment dispatcher.

Every registered LDM task must be discoverable through ``build_env``, and the
task-owned ``build_rl_components`` adapter must resolve. Tasks whose candidate
model still needs a dedicated stateful RL design fail fast with a clear
"not wired" error rather than a generic missing-factory ``KeyError``.
"""

from __future__ import annotations

import json

import pytest

from ldm_rl import EnvConfig, LDMEnv
from ldm_rl.factories import build_env

# Tasks whose stateless mock adapter is wired and expected to build an env.
WIRED_MOCK_TASKS = {
    "ai4bio_mutation_effect_prediction",
    "causal_discovery_discrete",
    "small_molecule",
    "llm_kv_adaptive_quantization",
}
# Tasks that resolve to a placeholder adapter and must raise a clear error.
PLACEHOLDER_TASKS = {"antibody", "nanogpt"}


def test_all_registered_tasks_resolve_to_an_adapter() -> None:
    from ldm_tts.registration.registry import discover_task_definitions

    definitions = discover_task_definitions()
    assert set(definitions) == WIRED_MOCK_TASKS | PLACEHOLDER_TASKS
    from ldm_rl.factories import _resolve_adapter

    for task_id in sorted(definitions):
        adapter = _resolve_adapter(task_id)
        assert callable(adapter)


def test_wired_mock_tasks_build_env() -> None:
    for task_id in sorted(WIRED_MOCK_TASKS):
        env = build_env(task_id, mode="mock", config=EnvConfig(iterations=2))
        assert isinstance(env, LDMEnv), task_id
        assert env.task_spec.task == task_id


def test_placeholder_tasks_fail_with_clear_error_not_missing_factory() -> None:
    for task_id in sorted(PLACEHOLDER_TASKS):
        with pytest.raises(NotImplementedError, match="no RL environment adapter"):
            build_env(task_id, mode="mock", config=EnvConfig(iterations=2))


def test_unknown_task_lists_registered_ids() -> None:
    with pytest.raises(KeyError) as excinfo:
        build_env("not_a_task", mode="mock")
    message = str(excinfo.value)
    for task_id in WIRED_MOCK_TASKS | PLACEHOLDER_TASKS:
        assert task_id in message


def test_llm_kv_mock_episode_runs() -> None:
    env = build_env(
        "llm_kv_adaptive_quantization",
        mode="mock",
        config=EnvConfig(iterations=2, reservoir_size=1),
    )
    observation = env.reset()
    assert "llm_kv_adaptive_quantization" in observation

    bit_caps = iter([4, 3])

    def action(_obs: str) -> str:
        return json.dumps(
            {
                "candidates": [
                    {
                        "bit_cap": next(bit_caps),
                        "key_group_size": 32,
                        "value_group_size": 32,
                        "residual_length": 128,
                    }
                ]
            }
        )

    result = env.run(action)
    assert result.rounds == 2
    for step in result.steps:
        assert step.info["evaluated"][0]["evaluation"]["status"] == "succeeded"
        metrics = step.info["evaluated"][0]["evaluation"]["metrics"]
        assert "selection_score" in metrics
