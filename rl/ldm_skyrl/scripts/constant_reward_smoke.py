"""Phase item A5: the smallest generator that still exercises the whole loop.

Returns a constant reward and a fixed response, so nothing about LDM is on the
path.  What it does exercise is everything that has never run on aarch64 with
driver 565: Ray placement, the vLLM engine, the Megatron policy, the NCCL weight
sync between them, and the checkpoint write.

The reward is constant on purpose, which means every group has zero variance and
GRPO gets no gradient.  That is fine -- and it is also the cheapest possible
check that ``zero_variance_filter`` behaves as documented, because with the
filter off this run is exactly the configuration that produced 0/0 on the slime
line.
"""

from __future__ import annotations

import argparse

from .._deps import gate, generator_base

CONSTANT_REWARD = 1.0


class ConstantRewardGenerator(generator_base()):
    """No environment, no tokenizer round-trip, no LDM import."""

    def __init__(self, inference_engine_client, response_tokens: int = 8):
        self.engine = inference_engine_client
        self.response_tokens = response_tokens

    async def generate(self, input_batch):
        n = len(input_batch.get("env_extras") or input_batch.get("prompts") or [])
        ids = list(range(100, 100 + self.response_tokens))
        return {
            "prompt_token_ids": [[1, 2, 3] for _ in range(n)],
            "response_ids": [list(ids) for _ in range(n)],
            "rewards": [CONSTANT_REWARD] * n,
            "loss_masks": [[1] * self.response_tokens for _ in range(n)],
            "stop_reasons": ["stop"] * n,
            "rollout_logprobs": [[-0.5] * self.response_tokens for _ in range(n)],
            "trajectory_ids": input_batch.get("trajectory_ids"),
            "trajectory_generation_times": [0.0] * n,
            "rollout_metrics": {"mean_reward": CONSTANT_REWARD},
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--export-path", required=True)
    ap.add_argument("--max-training-steps", type=int, default=2)
    args = ap.parse_args()

    raise gate(
        "A1",
        "the trainer construction for this smoke is written against SkyRL 0.3.0 "
        "and is filled in by the same change that lands the resolved uv.lock. "
        f"Arguments parsed fine: model={args.model}, steps={args.max_training_steps}.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
