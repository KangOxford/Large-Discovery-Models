"""Bridge tests with a fake Slime backend (no Slime / GPU required)."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace

import pytest

import ldm_rl.bridge as bridge
from ldm_rl import EpisodeSpec


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

    def append_response_tokens(
        self,
        args,
        tokens,
        log_probs=None,
        trainable=True,
        meta_info=None,
        text=None,
        update_terminal_info=True,
    ):
        # Mirrors slime.utils.types.Sample.append_response_tokens so that
        # contract bugs in bridge.generate surface here (e.g. non-trainable
        # env feedback tokens must pad rollout_log_probs with zeros so its
        # length stays aligned with response_length).
        tokens = [int(t) for t in tokens]
        if log_probs is not None and len(log_probs) != len(tokens):
            raise ValueError(
                f"log_probs length {len(log_probs)} != tokens length {len(tokens)}"
            )
        if tokens and trainable and log_probs is None:
            raise ValueError("trainable response tokens require rollout log probabilities.")
        if tokens and not trainable:
            if log_probs is not None:
                raise ValueError(
                    "non-trainable response tokens should not pass rollout log probabilities."
                )
            log_probs = [0.0] * len(tokens)
        if text is not None:
            self.response += text
        previous = self.response_length
        self.tokens += tokens
        self.response_length += len(tokens)
        if self.loss_mask is None:
            self.loss_mask = [1] * previous
        self.loss_mask += [1 if trainable else 0] * len(tokens)
        if log_probs is not None:
            if self.rollout_log_probs is None:
                self.rollout_log_probs = [0.0] * previous
            self.rollout_log_probs += [float(p) for p in log_probs]


class _CharTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [min(ord(ch), 5000) for ch in text]}


class _FakeState:
    def __init__(self, args):
        self.args = args
        self.tokenizer = _CharTokenizer()


def _token_ids(text: str) -> list[int]:
    return [min(ord(ch), 5000) for ch in text]


def _ai4bio_episode(iterations: int = 3) -> str:
    return EpisodeSpec(
        task="ai4bio_mutation_effect_prediction",
        mode="mock",
        iterations=iterations,
        reservoir_size=1,
        seed=0,
    ).to_json()


@pytest.fixture
def fake_slime(monkeypatch):
    def actions(text: str) -> str:
        """A scripted policy: emit successive catalog specs."""

        from tasks.ai4bio_mutation_effect_prediction.core.proposals import SPEC_SPACE

        index = len(re.findall(r"<round", text))
        spec = dict(SPEC_SPACE[index % len(SPEC_SPACE)])
        return json.dumps({"candidates": [spec]})

    async def fake_post(url, payload):
        action = actions(payload["text"])
        token_ids = _token_ids(action)
        return {
            "text": action,
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [
                    [-0.1, tid, 1.0, 1] for tid in token_ids
                ],
            },
        }

    monkeypatch.setattr(bridge, "_load_slime_deps", lambda: (fake_post, FakeSample))
    monkeypatch.setattr(bridge, "_load_generate_state", lambda args: _FakeState(args))
    return fake_post


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        hf_checkpoint="/unused",
    )


def test_generate_runs_full_episode(fake_slime) -> None:
    sample = FakeSample(prompt=_ai4bio_episode(iterations=3))
    out = asyncio.run(bridge.generate(_args(), sample, {"max_new_tokens": 512}))
    assert out.status == FakeSample.Status.COMPLETED
    assert out.reward is not None and out.reward > 0.0
    assert len(out.metadata["env_steps"]) == 3
    assert out.metadata["env_incumbent"] is not None
    # transcript contains rendered feedback between policy turns
    assert "<round" in out.response
    # loss mask aligns with response tokens: 1 for policy, 0 for feedback
    assert len(out.loss_mask) == out.response_length
    assert any(mask == 0 for mask in out.loss_mask)
    assert any(mask == 1 for mask in out.loss_mask)
    assert out.rollout_log_probs is not None


def test_rollout_log_probs_align_with_response(fake_slime) -> None:
    """Regression guard for the multi-turn log-prob contract.

    Each env feedback turn appends non-trainable tokens; their log-probs are
    zeros, so ``len(rollout_log_probs)`` must equal ``response_length`` after a
    full episode. A previous bridge bug dropped the env-turn zeros (only policy
    tokens were recorded), which broke the Megatron actor's
    ``log_prob length == response_length`` assertion during training.
    """

    sample = FakeSample(prompt=_ai4bio_episode(iterations=3))
    out = asyncio.run(bridge.generate(_args(), sample, {"max_new_tokens": 512}))
    assert out.response_length > 0
    assert out.rollout_log_probs is not None
    assert len(out.rollout_log_probs) == out.response_length
    assert len(out.loss_mask) == out.response_length
    # Every non-trainable (loss-mask 0) position has a zero log-prob.
    for mask, log_prob in zip(out.loss_mask, out.rollout_log_probs):
        if mask == 0:
            assert log_prob == 0.0


def test_reward_func_reads_filled_reward(fake_slime) -> None:
    sample = FakeSample(prompt=_ai4bio_episode(iterations=2))
    out = asyncio.run(bridge.generate(_args(), sample, {"max_new_tokens": 512}))
    assert bridge.reward_func(_args(), out) == out.reward


def test_generate_marks_failed_on_bad_episode_spec(fake_slime) -> None:
    sample = FakeSample(prompt='{"task": "not_a_task"}')
    out = asyncio.run(bridge.generate(_args(), sample, {"max_new_tokens": 512}))
    assert out.status == FakeSample.Status.FAILED
    assert out.reward == 0.0
    assert "env_error" in out.metadata
