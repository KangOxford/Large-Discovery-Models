"""CPU-only tests for the rollout loop.

Fakes stand in for the inference engine and the LDM environment, mirroring how
``rl/ldm_rl/tests/test_bridge.py`` tests the slime bridge without slime.  Nothing
here needs a GPU, SkyRL, or the network, so the whole of phase B can be finished
while phase A is still deciding whether the stack installs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from ldm_skyrl._deps import PhaseGateError
from ldm_skyrl.dataset import EpisodeDataset
from ldm_skyrl.generator import (
    CONTEXT_TOKEN,
    POLICY_TOKEN,
    EpisodeTrace,
    LDMAcquisitionGenerator,
    rollout_metrics,
    zero_variance_group_count,
)
from ldm_skyrl.ray_env import collect_forwarded_env
from ldm_skyrl.reward import RewardConcurrency


class FakeTokenizer:
    """One token per character, so token counts are readable in assertions."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 97 + 1000 for c in text]


class FakeEngine:
    """Returns a fixed reply, and records what it was asked."""

    def __init__(self, reply="propose", n_tokens=3, with_logprobs=True):
        self.reply = reply
        self.n_tokens = n_tokens
        self.with_logprobs = with_logprobs
        self.calls: list[list[int]] = []

    async def generate(self, batch):
        self.calls.append(list(batch["prompt_token_ids"][0]))
        ids = list(range(50, 50 + self.n_tokens))
        return {
            "responses": [self.reply],
            "response_ids": [ids],
            "stop_reasons": ["stop"],
            "response_logprobs": [[-0.5] * self.n_tokens] if self.with_logprobs else None,
        }


@dataclass
class FakeStep:
    reward: float
    observation: str
    done: bool


@dataclass
class FakeConfig:
    iterations: int


class FakeEnv:
    def __init__(self, iterations=3, rewards=(0.1, 0.2, 0.3), done_at=None):
        self.config = FakeConfig(iterations=iterations)
        self.rewards = list(rewards)
        self.done_at = done_at
        self.steps = 0

    def reset(self):
        return "OPEN"

    def step(self, action_text):
        reward = self.rewards[self.steps] if self.steps < len(self.rewards) else 0.0
        self.steps += 1
        done = self.done_at is not None and self.steps >= self.done_at
        return FakeStep(reward=reward, observation=f"FEEDBACK{self.steps}", done=done)


def build_generator(engine=None, env=None, max_seq_len=10_000):
    engine = engine or FakeEngine()
    env = env or FakeEnv()
    return LDMAcquisitionGenerator(
        generator_cfg=None,
        ldm_cfg=None,
        inference_engine_client=engine,
        tokenizer=FakeTokenizer(),
        max_seq_len=max_seq_len,
        env_factory=lambda spec: env,
    ), engine, env


# -- the loss mask is the thing most easily got wrong silently ---------------
def test_loss_mask_marks_only_policy_tokens():
    gen, engine, env = build_generator(engine=FakeEngine(n_tokens=3),
                                       env=FakeEnv(iterations=2, done_at=None))
    trace = asyncio.run(gen.run_episode({}))
    # two turns: 3 policy tokens then the feedback, then 3 more policy tokens.
    assert len(trace.response_ids) == len(trace.loss_mask)
    assert sum(trace.loss_mask) == 6, "one mask bit per token the policy sampled"
    assert set(trace.loss_mask) <= {POLICY_TOKEN, CONTEXT_TOKEN}
    assert trace.loss_mask[:3] == [POLICY_TOKEN] * 3


def test_engine_sees_the_growing_transcript_not_a_fresh_prompt():
    gen, engine, _ = build_generator(env=FakeEnv(iterations=2))
    asyncio.run(gen.run_episode({}))
    assert len(engine.calls) == 2
    assert len(engine.calls[1]) > len(engine.calls[0]), (
        "turn two must carry turn one's tokens; a shrinking context would mean "
        "the loop re-prompts from scratch and the mask no longer describes it"
    )
    assert engine.calls[1][:len(engine.calls[0])] == engine.calls[0]


def test_reward_accumulates_over_the_episode():
    gen, _, _ = build_generator(env=FakeEnv(iterations=3, rewards=(0.1, 0.2, 0.3)))
    trace = asyncio.run(gen.run_episode({}))
    assert trace.reward == pytest.approx(0.6)


def test_done_stops_the_episode_early():
    gen, engine, _ = build_generator(env=FakeEnv(iterations=5, rewards=(1.0,) * 5, done_at=2))
    trace = asyncio.run(gen.run_episode({}))
    assert trace.num_turns == 2
    assert len(engine.calls) == 2


def test_length_limit_sets_the_stop_reason():
    gen, _, _ = build_generator(env=FakeEnv(iterations=5), max_seq_len=4)
    trace = asyncio.run(gen.run_episode({}))
    assert trace.stop_reason == "length"


def test_zero_iteration_episode_is_rejected():
    gen, _, _ = build_generator(env=FakeEnv(iterations=0))
    with pytest.raises(ValueError, match="no rounds"):
        asyncio.run(gen.run_episode({}))


def test_mismatched_logprobs_raise_instead_of_shifting_the_ratio():
    trace = EpisodeTrace()
    with pytest.raises(ValueError, match="mismatched"):
        trace.extend([1, 2, 3], POLICY_TOKEN, [-0.1, -0.2])


def test_generate_returns_every_required_key():
    gen, _, _ = build_generator(env=FakeEnv(iterations=2))
    out = asyncio.run(gen.generate({"env_extras": [{}], "sampling_params": None,
                                    "trajectory_ids": None}))
    for key in ("prompt_token_ids", "response_ids", "rewards", "loss_masks",
                "stop_reasons", "rollout_logprobs", "trajectory_generation_times",
                "rollout_metrics"):
        assert key in out, f"GeneratorOutput is missing {key}"
    assert len(out["response_ids"][0]) == len(out["loss_masks"][0])


def test_generate_without_episodes_names_the_phase_item():
    gen, _, _ = build_generator()
    with pytest.raises(PhaseGateError, match="B2"):
        asyncio.run(gen.generate({"env_extras": []}))


# -- the statistic PR #2 measured, reproduced here so the lines compare ------
@pytest.mark.parametrize("rewards,size,expected", [
    ([1.0, 1.0, 1.0, 1.0], 2, 2),
    ([1.0, 2.0, 3.0, 4.0], 2, 0),
    ([1.0, 1.0, 3.0, 4.0], 2, 1),
    ([1.0, 1.0 + 1e-9, 3.0, 4.0], 2, 1),   # inside the tolerance: still flat
])
def test_zero_variance_group_count(rewards, size, expected):
    assert zero_variance_group_count(rewards, size) == expected


def test_zero_variance_rejects_a_partial_group():
    with pytest.raises(ValueError, match="do not divide"):
        zero_variance_group_count([1.0, 2.0, 3.0], 2)


def test_rollout_metrics_reports_the_env_time_share():
    traces = [EpisodeTrace(response_ids=[1, 2], loss_mask=[1, 0], reward=1.0,
                           llm_seconds=1.0, env_seconds=3.0)]
    m = rollout_metrics(traces)
    assert m["env_time_share"] == pytest.approx(0.75)
    assert m["trained_token_share"] == pytest.approx(0.5)


def test_rollout_metrics_on_an_empty_batch_is_empty_not_zero():
    assert rollout_metrics([]) == {}


# -- the smaller pieces ------------------------------------------------------
def test_dataset_row_count_equals_line_count(tmp_path):
    p = tmp_path / "eps.jsonl"
    p.write_text('{"task_id": "a"}\n\n{"task_id": "b"}\n')
    ds = EpisodeDataset(p)
    assert len(ds) == 2
    assert ds[0]["env_extras"]["task_id"] == "a"
    assert ds[0]["env_class"] == EpisodeDataset.ENV_CLASS


def test_dataset_missing_file_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.jsonl"):
        EpisodeDataset(tmp_path / "nope.jsonl")


def test_dataset_bad_json_names_the_line(tmp_path):
    p = tmp_path / "eps.jsonl"
    p.write_text('{"ok": 1}\nnot json\n')
    with pytest.raises(ValueError, match=":2 is not valid JSON"):
        EpisodeDataset(p)


def test_reward_concurrency_never_exceeds_its_ceiling():
    rc = RewardConcurrency(max_concurrency=3)

    async def slow():
        await asyncio.sleep(0.01)
        return 1

    async def drive():
        return await asyncio.gather(*(rc.run(slow) for _ in range(20)))

    assert asyncio.run(drive()) == [1] * 20
    assert rc.peak_in_flight <= 3


def test_reward_concurrency_rejects_a_zero_ceiling():
    with pytest.raises(ValueError, match="at least 1"):
        RewardConcurrency(max_concurrency=0)


def test_forwarded_env_only_reports_what_is_set(monkeypatch):
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setenv("TASK_PYTHON", "/some/python")
    env = collect_forwarded_env()
    assert env["TASK_PYTHON"] == "/some/python"
    assert "CUDA_HOME" not in env
