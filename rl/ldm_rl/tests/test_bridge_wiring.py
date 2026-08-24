"""Bridge/adapter wiring tests: real kwargs and seed propagation.

Locks the contract that ``EpisodeSpec`` carries ``seed`` and task-specific
real-evaluation kwargs, and that ``build_env`` forwards both to the task's RL
adapter (so rollout construction honors them instead of dropping them).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from ldm_rl import EnvConfig, EpisodeSpec
from ldm_rl import bridge, factories


def test_episode_spec_real_kwargs_round_trip() -> None:
    spec = EpisodeSpec(
        task="ai4bio_mutation_effect_prediction",
        mode="real",
        iterations=2,
        seed=11,
        real={
            "upstream_root": "/data/upstream",
            "data_dir": "/data/mlsbench",
            "cv_dir": "/data/cv",
            "evaluation_timeout": 60.0,
        },
    )
    restored = EpisodeSpec.from_json(spec.to_json())
    assert restored == spec
    assert restored.real == spec.real
    assert restored.seed == 11


def test_episode_spec_mock_rejects_real_kwargs() -> None:
    with pytest.raises(ValueError, match="must not carry real-evaluation kwargs"):
        EpisodeSpec(task="small_molecule", mode="mock", real={"vina_bin": "/x"})


def test_build_env_forwards_seed_and_real_kwargs(monkeypatch) -> None:
    captured: dict = {}
    real_resolve = factories._resolve_adapter

    def spy_resolve(task_id: str):
        adapter = real_resolve(task_id)

        def wrapped(mode: str = "mock", **kwargs):
            captured.update(kwargs)
            return adapter(mode=mode, **kwargs)

        return wrapped

    monkeypatch.setattr(factories, "_resolve_adapter", spy_resolve)
    factories.build_env(
        "ai4bio_mutation_effect_prediction",
        mode="mock",
        config=EnvConfig(iterations=1, reservoir_size=1),
        seed=42,
        upstream_root="/data/upstream",
        data_dir="/data/mlsbench",
        cv_dir="/data/cv",
    )
    assert captured["seed"] == 42
    assert captured["upstream_root"] == "/data/upstream"
    assert captured["data_dir"] == "/data/mlsbench"
    assert captured["cv_dir"] == "/data/cv"


def test_real_mode_requires_task_kwargs() -> None:
    # ai4bio real mode without upstream/data/cv kwargs fails with a clear error
    # listing what is missing, rather than building a half-wired evaluator.
    with pytest.raises(ValueError, match="requires kwargs"):
        factories.build_env(
            "ai4bio_mutation_effect_prediction",
            mode="real",
            config=EnvConfig(iterations=1, reservoir_size=1),
        )


def test_seed_survives_episode_prompt_data() -> None:
    from ldm_rl.episodes import make_prompt_rows

    rows = make_prompt_rows(
        [
            EpisodeSpec(
                task="small_molecule",
                mode="mock",
                iterations=1,
                seed=100 + index,
            )
            for index in range(3)
        ]
    )
    seeds = [json.loads(row["prompt"])["seed"] for row in rows]
    assert seeds == [100, 101, 102]


def test_bridge_passes_seed_into_build_env(monkeypatch) -> None:
    """End-to-end: EpisodeSpec.seed reaches build_env through bridge.generate."""

    from ldm_rl.tests.test_bridge import FakeSample, _FakeState, _token_ids

    captured: dict = {}
    real_build_env = factories.build_env

    def spy_build_env(task_id: str, **kwargs):
        captured.update(kwargs)
        return real_build_env(task_id, **kwargs)

    monkeypatch.setattr(factories, "build_env", spy_build_env)
    monkeypatch.setattr(bridge, "_load_generate_state", lambda args: _FakeState(args))

    async def fake_post(url, payload):
        from tasks.ai4bio_mutation_effect_prediction.core.proposals import SPEC_SPACE

        spec = dict(SPEC_SPACE[0])
        text = json.dumps({"candidates": [spec]})
        token_ids = _token_ids(text)
        return {
            "text": text,
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.1, tid, 1.0, 1] for tid in token_ids],
            },
        }

    monkeypatch.setattr(bridge, "_load_slime_deps", lambda: (fake_post, FakeSample))

    spec = EpisodeSpec(
        task="ai4bio_mutation_effect_prediction",
        mode="mock",
        iterations=1,
        reservoir_size=1,
        seed=7,
    )
    sample = FakeSample(prompt=spec.to_json())
    asyncio.run(
        bridge.generate(
            _bridge_args(),
            sample,
            {"max_new_tokens": 512},
        )
    )
    assert captured.get("seed") == 7


def _bridge_args():
    from types import SimpleNamespace

    return SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        hf_checkpoint="/unused",
    )
