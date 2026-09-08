"""Concurrent episodes must overlap their (slow) real evaluations.

nanoGPT measures a proposal by training for its full wall-clock budget, so one
environment step costs ~300s and blocks. If ``bridge.generate`` called it
directly, that synchronous call would stall the rollout worker's event loop:
every other episode in the process would queue behind it, and an evaluation GPU
pool would never get more than one device busy no matter how many GPUs it had.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace

import pytest

import ldm_rl.bridge as bridge
import ldm_rl.factories as factories
from ldm_rl import EpisodeSpec

STEP_SECONDS = 0.3
EPISODES = 4


@dataclass
class FakeSample:
    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        FAILED = "failed"

    prompt: str = ""
    tokens: list[int] = field(default_factory=list)
    loss_mask: list[int] | None = None
    response: str = ""
    response_length: int = 0
    reward: float | None = None
    rollout_log_probs: list[float] | None = None
    status: "FakeSample.Status" = Status.PENDING
    metadata: dict = field(default_factory=dict)

    def append_response_tokens(self, args, tokens, log_probs=None, trainable=True,
                              meta_info=None, text=None, update_terminal_info=True):
        self.tokens += [int(t) for t in tokens]
        self.response_length += len(tokens)


class _CharTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [min(ord(ch), 5000) for ch in text]}


class _FakeState:
    def __init__(self, args):
        self.args = args
        self.tokenizer = _CharTokenizer()


class _SlowEnv:
    """Stands in for a real nanoGPT environment: one blocking step."""

    def __init__(self):
        self.calls = 0

    def reset(self):
        return "propose knobs"

    def step(self, action_text):
        self.calls += 1
        time.sleep(STEP_SECONDS)  # a blocking wait, like a training subprocess
        return SimpleNamespace(
            observation="<round 0> measured",
            reward=1.0,
            terminated=True,
            truncated=False,
            done=True,  # EnvStep exposes this as a property
            info={"stop_reason": "iteration_budget", "incumbent": None},
        )


@pytest.fixture
def slow_slime(monkeypatch):
    async def fake_post(url, payload):
        action = json.dumps({"proposals": [{"DEPTH": 8}]})
        token_ids = [min(ord(ch), 5000) for ch in action]
        return {
            "text": action,
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.1, tid, 1.0, 1] for tid in token_ids],
            },
        }

    monkeypatch.setattr(bridge, "_load_slime_deps", lambda: (fake_post, FakeSample))
    monkeypatch.setattr(bridge, "_load_generate_state", lambda args: _FakeState(args))
    monkeypatch.setattr(factories, "build_env", lambda *a, **k: _SlowEnv())


def _args():
    return SimpleNamespace(
        sglang_router_ip="127.0.0.1", sglang_router_port=30000, hf_checkpoint="/unused"
    )


def _episode():
    return EpisodeSpec(
        task="nanogpt", mode="mock", iterations=1, reservoir_size=1, seed=0
    ).to_json()


def test_concurrent_episodes_overlap_their_evaluations(slow_slime):
    async def run_all():
        samples = [FakeSample(prompt=_episode()) for _ in range(EPISODES)]
        return await asyncio.gather(
            *(
                bridge.generate(_args(), sample, {"max_new_tokens": 64})
                for sample in samples
            )
        )

    started = time.monotonic()
    results = asyncio.run(run_all())
    elapsed = time.monotonic() - started

    assert all(sample.reward == 1.0 for sample in results)
    assert all(
        sample.status == FakeSample.Status.COMPLETED for sample in results
    )
    # Serialised, this would take EPISODES * STEP_SECONDS. Allow generous slack
    # for thread startup while still failing loudly if the loop is blocked.
    serial = EPISODES * STEP_SECONDS
    assert elapsed < serial * 0.6, (
        f"{EPISODES} episodes took {elapsed:.2f}s; serialised would be "
        f"{serial:.2f}s, so env.step is blocking the event loop"
    )
