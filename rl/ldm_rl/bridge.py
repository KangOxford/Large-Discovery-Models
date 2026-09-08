"""Slime integration for the LDM environment.

Mirrors Slime's search-r1 example: a custom rollout ``generate`` function runs
one multi-turn agent-environment loop per sample, and a custom ``reward_func``
exposes the episode reward. The sample prompt is an :class:`EpisodeSpec` JSON;
the rendered policy prompt comes from ``env.reset()``. Environment feedback is
appended with ``trainable=False`` (loss-mask 0), policy turns with
``trainable=True``.

Usage in a Slime launch script::

    export PYTHONPATH=$REPO_ROOT/rl:$REPO_ROOT
    --custom-generate-function-path ldm_rl.bridge.generate
    --custom-rm-path ldm_rl.bridge.reward_func
    --prompt-data /path/to/ldm_episodes.jsonl
    --input-key prompt --label-key label

Slime imports happen lazily so the environment core stays importable without
Slime installed (unit tests inject fake dependencies instead).
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _load_slime_deps() -> tuple[Any, Any]:
    from slime.utils.http_utils import post
    from slime.utils.types import Sample

    return post, Sample


def _load_generate_state(args: Any) -> Any:
    from slime.rollout.sglang_rollout import GenerateState

    return GenerateState(args)


def _apply_chat_template(state: Any, text: str) -> str:
    """Wrap the rendered prompt in the tokenizer chat template when possible.

    Reasoning-capable checkpoints (e.g. Qwen3.5) emit a ``<think>...</think>``
    block by default, which consumes the rollout response budget and yields
    ``length``-truncated turns before any candidate is produced. Disable
    thinking via the ``enable_thinking`` template kwarg, mirroring
    ``scripts/llm_server.py``; tokenizers without that kwarg fall back to the
    plain template.
    """

    tokenizer = getattr(state, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return text

    def _render(**kwargs: Any) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=False,
            **kwargs,
        )

    try:
        templated = _render(enable_thinking=False)
    except TypeError:
        try:
            templated = _render()
        except Exception:  # noqa: BLE001 - fall back to the untemplated prompt
            return text
    except Exception:  # noqa: BLE001 - fall back to the untemplated prompt
        return text
    return templated if isinstance(templated, str) and templated.strip() else text


def _budget_exhausted(sample, budget: int, turns: int, sample_cls) -> None:
    """Mark a sample that hit the per-episode response-token budget."""
    logger.warning(
        "episode response-token budget %d exhausted after %d turn(s); stopping the "
        "episode early. Raise LDM_RL_EPISODE_TOKEN_BUDGET (or "
        "--rollout-max-response-len) for longer episodes, and raise "
        "--max-tokens-per-gpu with it -- otherwise the sample will not fit a "
        "training micro-batch.",
        budget,
        turns,
    )
    sample.metadata["episode_token_budget_hit"] = True
    sample.status = sample_cls.Status.TRUNCATED


async def generate(args, sample, sampling_params, evaluation: bool = False) -> Any:
    """Slime custom generate function: one LDM campaign episode per sample."""

    post, Sample = _load_slime_deps()
    from ldm_rl.episodes import EpisodeSpec
    from ldm_rl.factories import build_env

    try:
        spec = EpisodeSpec.from_json(str(sample.prompt))
        if spec.mode == "real":
            from ldm_rl.remote_env import RemoteLDMEnv

            task_python = str(
                spec.real.get("task_python")
                or "/mnt/data0/ys/LDM/tasks/small_molecule/.venv/bin/python"
            )
            env = RemoteLDMEnv(spec, task_python)
        else:
            env = build_env(
                spec.task,
                mode=spec.mode,
                config=spec.to_env_config(),
                context=spec.context,
                seed=spec.seed,
                **spec.real,
            )
    except Exception as exc:  # noqa: BLE001 - rollout must not crash the pool
        sample.status = Sample.Status.FAILED
        sample.reward = 0.0
        sample.metadata["env_error"] = f"{type(exc).__name__}: {exc}"
        sample.metadata["stop_reason"] = "episode_setup_failed"
        return sample

    state = _load_generate_state(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # The sample prompt is the raw EpisodeSpec JSON; render the policy prompt
    # here so --apply-chat-template on the data source cannot corrupt the spec.
    prompt_text = env.reset()
    prompt_text = _apply_chat_template(state, prompt_text)
    prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)[
        "input_ids"
    ]
    sample.prompt = prompt_text
    sample.metadata["episode_spec"] = spec.to_dict()
    sample.metadata["env_steps"] = []
    # Written now and overwritten at the end, so a sample that dies mid-episode
    # still carries a reason. slime aggregates this key into
    # `rollout/stop_reason/*`, and that aggregation reports the fraction of
    # samples that carried the key at all -- so every return path here sets it,
    # or the denominator quietly turns "this episode broke" into "this run was
    # not instrumented".
    sample.metadata["stop_reason"] = "aborted_before_first_step"
    sample.tokens = list(prompt_token_ids)
    sample.loss_mask = []

    response = ""
    response_token_ids: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs: list[float] = []
    total_reward = 0.0
    last_step = None
    last_finish = "stop"
    output = None

    # --- per-episode response-token budget -------------------------------------
    #
    # sglang applies ``max_new_tokens`` per request, so an episode of
    # ``spec.iterations`` turns could produce ``iterations * max_new_tokens``
    # trainable tokens plus every environment observation, and nothing
    # downstream bounded that. With iterations=20 and max_new_tokens=8192 the
    # ceiling is ~164k tokens for one training sample.
    #
    # That breaks the training packer: ``slime.utils.dp_schedule.first_fit_pack``
    # cannot split a sample, so a sample longer than ``--max-tokens-per-gpu``
    # lands alone in an over-cap micro-batch, and the forward then materialises a
    # ``[tokens, vocab]`` logits tensor several times over and OOMs. Observed:
    # a 22,784-token micro-batch against a 8,192 cap, dying on a 12.90 GiB fp32
    # allocation.
    #
    # slime's own multi-turn rollout (slime/rollout/sglang_rollout.py) subtracts
    # the tokens already produced from ``max_new_tokens`` on every turn. Do the
    # same, so ``--rollout-max-response-len`` means what it says: the most
    # response tokens one training sample may contribute.
    #
    # Set LDM_RL_EPISODE_TOKEN_BUDGET to allow longer episodes -- and raise
    # --max-tokens-per-gpu with it, or the schedule-time check in
    # build_dp_schedule will stop the run and say so.
    _env_budget = os.environ.get("LDM_RL_EPISODE_TOKEN_BUDGET", "").strip()
    episode_token_budget = (
        int(_env_budget) if _env_budget else int(getattr(args, "rollout_max_response_len", 0) or 0)
    )
    per_turn_max_new_tokens = int(sampling_params.get("max_new_tokens") or 0)

    try:
        for _turn_idx in range(spec.iterations):
            turn_sampling_params = sampling_params
            if episode_token_budget > 0:
                remaining = episode_token_budget - len(response_token_ids)
                if remaining <= 0:
                    _budget_exhausted(sample, episode_token_budget, _turn_idx, Sample)
                    break
                turn_sampling_params = dict(sampling_params)
                turn_sampling_params["max_new_tokens"] = (
                    min(per_turn_max_new_tokens, remaining) if per_turn_max_new_tokens > 0 else remaining
                )
            payload: dict[str, Any] = {
                "text": prompt_text + response,
                "sampling_params": turn_sampling_params,
                "return_logprob": True,
            }
            output = await post(url, payload)
            meta_info = output["meta_info"]
            last_finish = meta_info["finish_reason"]["type"]
            if last_finish == "abort":
                sample.status = Sample.Status.ABORTED
                return sample

            cur_response = output["text"]
            if "output_token_logprobs" not in meta_info:
                raise RuntimeError(
                    "output_token_logprobs missing from sglang meta_info; the "
                    "custom LDM generate function requires return_logprob"
                )
            cur_token_ids = [
                item[1] for item in meta_info["output_token_logprobs"]
            ]
            cur_log_probs = [
                item[0] for item in meta_info["output_token_logprobs"]
            ]

            # Clamp against the budget rather than trusting the server to have
            # honoured max_new_tokens. The packer needs a hard bound: one sample
            # over --max-tokens-per-gpu cannot be split and OOMs the forward.
            budget_hit = False
            if episode_token_budget > 0:
                room = max(episode_token_budget - len(response_token_ids), 0)
                if len(cur_token_ids) >= room:
                    cur_token_ids = cur_token_ids[:room]
                    cur_log_probs = cur_log_probs[:room]
                    budget_hit = True

            response += cur_response
            if cur_token_ids:
                response_token_ids += cur_token_ids
                loss_mask += [1] * len(cur_token_ids)
                rollout_log_probs += cur_log_probs
                sample.append_response_tokens(
                    args,
                    tokens=cur_token_ids,
                    log_probs=cur_log_probs,
                    trainable=True,
                    meta_info=meta_info,
                )
            if budget_hit:
                _budget_exhausted(sample, episode_token_budget, _turn_idx + 1, Sample)
                break

            if last_finish == "length":
                sample.status = Sample.Status.TRUNCATED
                break

            step = env.step(cur_response)
            last_step = step
            total_reward += step.reward
            sample.metadata["env_steps"].append(step.info)

            if step.observation:
                obs_token_ids = state.tokenizer(
                    step.observation, add_special_tokens=False
                )["input_ids"]
                # Observation tokens are part of the training sample, so they
                # count against the same budget. A single environment message can
                # be long enough to blow the cap on its own, which is why
                # bounding max_new_tokens alone is not sufficient.
                budget_hit = False
                if episode_token_budget > 0:
                    room = max(episode_token_budget - len(response_token_ids), 0)
                    if len(obs_token_ids) >= room:
                        obs_token_ids = obs_token_ids[:room]
                        budget_hit = True
                # The transcript keeps the whole observation even when its token
                # stream was cut: the episode ends here, so nothing generates
                # from it again.
                response += step.observation
                if obs_token_ids:
                    response_token_ids += obs_token_ids
                    loss_mask += [0] * len(obs_token_ids)
                    # Environment feedback tokens are non-trainable; pad the
                    # rollout log-probs with zeros so their length stays aligned
                    # with response_length (the actor slices log_probs ==
                    # response_length during training).
                    rollout_log_probs += [0.0] * len(obs_token_ids)
                    sample.append_response_tokens(
                        args, tokens=obs_token_ids, trainable=False
                    )
                if budget_hit:
                    _budget_exhausted(sample, episode_token_budget, _turn_idx + 1, Sample)
                    break

            if step.done:
                break
    except Exception as exc:  # noqa: BLE001 - keep partial credit on env errors
        sample.metadata["env_error"] = f"{type(exc).__name__}: {exc}"
        if sample.status != Sample.Status.ABORTED:
            sample.status = Sample.Status.FAILED
    else:
        if last_step is not None and last_step.done:
            sample.status = Sample.Status.COMPLETED
        else:
            sample.status = Sample.Status.TRUNCATED

    sample.response = response
    sample.response_length = len(response_token_ids)
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = rollout_log_probs
    sample.reward = total_reward
    sample.metadata["env_total_reward"] = total_reward
    if last_step is not None:
        sample.metadata["env_terminated"] = last_step.terminated
        sample.metadata["env_truncated"] = last_step.truncated
        sample.metadata["env_incumbent"] = last_step.info.get("incumbent")
        # Same rule `run_episode` uses, from the same function, so the metric and
        # the environment cannot drift apart. Only the last step matters, which is
        # all `derive_stop_reason` reads.
        from ldm_rl.env import derive_stop_reason

        sample.metadata["stop_reason"] = derive_stop_reason([last_step])
    if sample.metadata.get("episode_token_budget_hit"):
        # The budget cut the episode short before the environment had a reason of
        # its own; that is a distinct outcome and reading it as one of the
        # environment's four would hide a configuration problem as an environment
        # one.
        sample.metadata["stop_reason"] = "episode_token_budget"
    _close = getattr(env, "close", None)
    if _close is not None:
        _close()
    return sample


def reward_func(args, sample, **kwargs) -> float:
    """Slime custom reward function; reads the reward filled during rollout."""

    if sample.reward is not None:
        return float(sample.reward)
    return float(sample.metadata.get("env_total_reward", 0.0))


__all__ = ["generate", "reward_func"]
