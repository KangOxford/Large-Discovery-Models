"""``LDMAcquisitionGenerator`` -- the SkyRL replacement for ``ldm_rl/bridge.py``.

One LDM campaign is one RL episode: the policy proposes candidate molecules, the
task evaluates them (docking, then a GP over the observed objectives), and the
episode reward is the acquisition value of what was proposed.  ``ldm_rl.env``
already implements all of that and depends on nothing but the ``ldm_tts``
contracts, so the port is confined to the framework seam.

Why token-in-token-out.  SkyRL's own note on ``InferenceEngineOutput`` is that
``decode(response_ids) == responses`` holds but the reverse does not, because
several token sequences render the same text.  A multi-turn loop that
re-tokenises the transcript after every turn therefore drifts away from what the
engine actually sampled, and the loss mask stops lining up with the tokens the
policy produced.  This generator never re-tokenises: it appends the engine's own
ids, and tokenises only the environment's feedback, which the policy did not
generate and which is masked out anyway.

Testing.  ``engine`` and ``tokenizer`` are constructor arguments rather than
globals, so the whole rollout loop runs on CPU against fakes -- the same
arrangement ``rl/ldm_rl/tests/test_bridge.py`` uses for slime.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from ._deps import gate, generator_base, require_ldm_rl

# Loss-mask convention, fixed here so every reader of this file sees it once.
POLICY_TOKEN = 1  # sampled by the policy: trained on
CONTEXT_TOKEN = 0  # prompt or environment feedback: not trained on


@dataclass
class EpisodeTrace:
    """One episode's token stream, reward, and the metrics worth logging."""

    prompt_ids: list[int] = field(default_factory=list)
    response_ids: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    reward: float = 0.0
    stop_reason: str = "stop"
    num_turns: int = 0
    # Wall-clock split. ``env`` is docking plus the GP, which is the expensive
    # half here: a single GP call was measured at 67 s against roughly 0.2 s of
    # generation, so a step's cost is dominated by whichever of these is not
    # overlapped with training.
    llm_seconds: float = 0.0
    env_seconds: float = 0.0

    def __post_init__(self) -> None:
        if len(self.response_ids) != len(self.loss_mask):
            raise ValueError(
                f"response_ids ({len(self.response_ids)}) and loss_mask "
                f"({len(self.loss_mask)}) must stay the same length"
            )

    def extend(self, ids: Sequence[int], mask_value: int,
               logprobs: Sequence[float] | None = None) -> None:
        self.response_ids.extend(ids)
        self.loss_mask.extend([mask_value] * len(ids))
        if logprobs is None:
            # Position kept so the arrays stay index-aligned; masked out anyway.
            self.logprobs.extend([0.0] * len(ids))
        elif len(logprobs) != len(ids):
            raise ValueError(
                f"got {len(logprobs)} logprobs for {len(ids)} tokens; the engine "
                "returned a mismatched pair, which would silently shift the "
                "importance ratio by one position"
            )
        else:
            self.logprobs.extend(logprobs)


class LDMAcquisitionGenerator(generator_base()):
    """Drives LDM episodes and hands SkyRL a ``GeneratorOutput``."""

    def __init__(
        self,
        *,
        generator_cfg: Any,
        ldm_cfg: Any,
        inference_engine_client: Any,
        tokenizer: Any,
        max_seq_len: int,
        env_factory: Callable[[dict], Any] | None = None,
    ) -> None:
        self.generator_cfg = generator_cfg
        self.ldm_cfg = ldm_cfg
        self.engine = inference_engine_client
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self._env_factory = env_factory
        self._rate_limiter: Any = None

    # -- seams ------------------------------------------------------------
    def _build_env(self, episode_spec: dict) -> Any:
        """Construct one ``LDMEnv``.  Injected in tests, imported in production."""
        if self._env_factory is not None:
            return self._env_factory(episode_spec)
        ldm_rl = require_ldm_rl("B1")
        from ldm_rl.factories import build_env  # noqa: PLC0415 - deferred on purpose

        del ldm_rl
        return build_env(**episode_spec)

    def _encode(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return list(ids)

    async def _sample(self, prompt_ids: list[int], sampling_params: dict | None) -> dict:
        out = await self.engine.generate({
            "prompts": None,
            "prompt_token_ids": [prompt_ids],
            "sampling_params": sampling_params,
            "session_ids": None,
            "mm_features": None,
            "cache_salt": None,
        })
        logprobs = out.get("response_logprobs") or [None]
        return {
            "text": out["responses"][0],
            "ids": list(out["response_ids"][0]),
            "stop_reason": out["stop_reasons"][0],
            "logprobs": logprobs[0],
        }

    # -- one episode ------------------------------------------------------
    async def run_episode(self, episode_spec: dict,
                          sampling_params: dict | None = None) -> EpisodeTrace:
        env = self._build_env(episode_spec)
        trace = EpisodeTrace()

        t0 = time.perf_counter()
        opening = env.reset()
        trace.env_seconds += time.perf_counter() - t0
        trace.prompt_ids = self._encode(opening)

        budget = int(getattr(env.config, "iterations", 0) or 0)
        if budget <= 0:
            raise ValueError(
                f"episode has iterations={budget}; an episode with no rounds "
                "produces a zero-length response and a reward that means nothing"
            )

        for turn in range(budget):
            context = trace.prompt_ids + trace.response_ids
            if len(context) >= self.max_seq_len:
                trace.stop_reason = "length"
                break

            t0 = time.perf_counter()
            sample = await self._sample(context, sampling_params)
            trace.llm_seconds += time.perf_counter() - t0
            trace.extend(sample["ids"], POLICY_TOKEN, sample["logprobs"])
            trace.stop_reason = sample["stop_reason"]
            trace.num_turns = turn + 1

            t0 = time.perf_counter()
            step = env.step(sample["text"])
            trace.env_seconds += time.perf_counter() - t0
            trace.reward += float(step.reward)

            if step.done:
                break
            trace.extend(self._encode(step.observation), CONTEXT_TOKEN)

        return trace

    # -- SkyRL entry point ------------------------------------------------
    async def generate(self, input_batch: Any) -> Any:
        specs = input_batch.get("env_extras")
        if not specs:
            raise gate(
                "B2",
                "input_batch['env_extras'] is empty, so there are no EpisodeSpecs to "
                "run. The dataset adapter is what fills this field.",
            )
        sampling_params = input_batch.get("sampling_params")

        traces = await asyncio.gather(*(
            self.run_episode(spec, sampling_params) for spec in specs
        ))

        return {
            "prompt_token_ids": [t.prompt_ids for t in traces],
            "response_ids": [t.response_ids for t in traces],
            "rewards": [t.reward for t in traces],
            "loss_masks": [t.loss_mask for t in traces],
            "stop_reasons": [t.stop_reason for t in traces],
            "rollout_logprobs": [t.logprobs for t in traces],
            "trajectory_ids": input_batch.get("trajectory_ids"),
            "trajectory_generation_times": [
                t.llm_seconds + t.env_seconds for t in traces
            ],
            "rollout_metrics": rollout_metrics(traces),
        }


def rollout_metrics(traces: Iterable[EpisodeTrace]) -> dict[str, float]:
    """Aggregates worth watching from step one on this line.

    ``env_time_share`` is the one to watch first.  On the slime line the GP
    dominated the step, and the whole argument for the fully async trainer is
    that this share stops being on the critical path.  Logging it makes that
    claim falsifiable rather than assumed.
    """
    traces = list(traces)
    if not traces:
        return {}
    n = len(traces)
    llm = sum(t.llm_seconds for t in traces)
    env = sum(t.env_seconds for t in traces)
    total = llm + env
    return {
        "mean_reward": sum(t.reward for t in traces) / n,
        "mean_turns": sum(t.num_turns for t in traces) / n,
        "mean_response_tokens": sum(len(t.response_ids) for t in traces) / n,
        "trained_token_share": (
            sum(sum(t.loss_mask) for t in traces)
            / max(1, sum(len(t.loss_mask) for t in traces))
        ),
        "llm_seconds": llm,
        "env_seconds": env,
        "env_time_share": env / total if total > 0 else 0.0,
    }


def zero_variance_group_count(rewards: Sequence[float], group_size: int,
                              tol: float = 1e-6) -> int:
    """Groups whose rewards are all equal, i.e. those GRPO gets nothing from.

    This reproduces the statistic PR #2 measured on the slime line (0.87 groups
    per step at n=2, falling to 0.00 at n=8), so the two lines can be compared
    on the same number.  SkyRL computes the same thing internally when
    ``zero_variance_filter`` is on; this copy exists so the number can be logged
    whether or not the filter is enabled.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if len(rewards) % group_size:
        raise ValueError(
            f"{len(rewards)} rewards do not divide into groups of {group_size}; "
            "a partial group would be silently miscounted"
        )
    count = 0
    for start in range(0, len(rewards), group_size):
        group = rewards[start:start + group_size]
        if max(group) - min(group) <= tol:
            count += 1
    return count


__all__ = [
    "CONTEXT_TOKEN",
    "EpisodeTrace",
    "LDMAcquisitionGenerator",
    "POLICY_TOKEN",
    "rollout_metrics",
    "zero_variance_group_count",
]
