"""Smoke-test the real Slime <-> bridge integration (no fake backend).

Unlike ``tests/test_bridge.py`` (which monkeypatches Slime's ``post``/``Sample``),
this script drives ``ldm_rl.bridge.generate`` with a *real* ``slime.utils.types.Sample``
object, a real ``slime.rollout.sglang_rollout.GenerateState`` (which loads the HF
tokenizer), and a live SGLang server on ``sglang_router_ip:port``.

Prerequisites (on the GPU node, in an env with sglang + slime installed):

    1. start SGLang serving the policy checkpoint, e.g.:
         sglang serve --model-path /path/to/Qwen3.5-2B \
             --host 127.0.0.1 --port 30000 --tp 2 --trust-remote-code
    2. run this script with PYTHONPATH=<repo>/rl:<repo>:
         python rl/ldm_rl/examples/slime_bridge_smoke.py

Each run executes one full small-molecule (mock) RL episode through the real
Slime generate path: policy turns come from SGLang, environment feedback is
appended with ``trainable=False``, and the episode reward is filled on the
Sample.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

# Make ``ldm_rl`` (rl/) and the repo root importable when run as a plain script.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_RL_ROOT = Path(__file__).resolve().parents[2]
for _path in (_RL_ROOT, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import ldm_rl.bridge as bridge  # noqa: E402
from ldm_rl.episodes import EpisodeSpec  # noqa: E402

# Overridable via env vars; defaults are placeholders, not real server paths.
HF_CHECKPOINT = os.environ.get("HF_CHECKPOINT", "/path/to/Qwen3.5-2B")
SGLANG_HOST = os.environ.get("SGLANG_HOST", "127.0.0.1")
SGLANG_PORT = int(os.environ.get("SGLANG_PORT", "30000"))


def build_args() -> SimpleNamespace:
    """Minimal args matching what bridge.generate + GenerateState consume."""

    return SimpleNamespace(
        hf_checkpoint=HF_CHECKPOINT,
        sglang_router_ip=SGLANG_HOST,
        sglang_router_port=SGLANG_PORT,
        sglang_server_concurrency=1,
        sglang_dp_size=1,
        rollout_num_gpus=2,
        rollout_num_engines=1,
        n_samples_per_prompt=1,
        rollout_temperature=0.8,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_max_response_len=512,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        sglang_enable_deterministic_inference=False,
        use_distributed_post=False,
    )


def main() -> int:
    from slime.utils.http_utils import init_http_client
    from slime.utils.types import Sample

    args = build_args()
    # bridge.generate's `post` uses Slime's module-level httpx client, which is
    # lazily created here before any rollout POST.
    init_http_client(args)
    spec = EpisodeSpec(
        task="small_molecule",
        mode="mock",
        iterations=4,
        reservoir_size=2,
        seed=7,
    )
    sample = Sample(prompt=spec.to_json())
    sampling_params = {
        "temperature": 0.8,
        "top_p": 1.0,
        "max_new_tokens": 512,
    }

    out = asyncio.run(bridge.generate(args, sample, sampling_params))

    env_steps = out.metadata.get("env_steps", [])
    print(json.dumps({
        "status": str(out.status),
        "reward": out.reward,
        "response_length": out.response_length,
        "loss_mask_len": len(out.loss_mask) if out.loss_mask is not None else 0,
        "n_policy_tokens": sum(1 for m in (out.loss_mask or []) if m == 1),
        "n_env_tokens": sum(1 for m in (out.loss_mask or []) if m == 0),
        "n_env_steps": len(env_steps),
        "env_incumbent": out.metadata.get("env_incumbent"),
        "env_terminated": out.metadata.get("env_terminated"),
        "env_truncated": out.metadata.get("env_truncated"),
        "env_error": out.metadata.get("env_error"),
        "episode_spec": out.metadata.get("episode_spec"),
    }, sort_keys=True, indent=2))

    # Sanity assertions on the real Slime sample object.
    assert out.response_length == len(out.loss_mask)
    if out.metadata.get("env_error") is None:
        assert env_steps, "expected at least one env step on a clean rollout"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
