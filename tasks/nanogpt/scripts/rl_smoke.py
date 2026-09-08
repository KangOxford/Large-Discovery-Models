#!/usr/bin/env python3
"""No-GPU smoke test: run one mock nanoGPT RL episode end to end.

Exercises every step of the real path except the training subprocess itself --
prompt rendering, the strict parser, admission, cross-round de-duplication, GP
selection, the improvement reward and episode termination.

    python -m tasks.nanogpt.scripts.rl_smoke

**The scores printed here are meaningless on purpose.** Mock ``val_bpb`` is a
hash of the candidate's canonical key, so it carries no information about the
config and no policy can learn from it. That is deliberate: this task shipped a
plausible-looking analytic val_bpb once, people trained on it, and its ranking
turned out to be inverted against the real trainer. A mock that cannot be
mistaken for a surrogate is the fix. Use this to check plumbing, never to draw
a conclusion.
"""

from __future__ import annotations

import json
import os
import sys

if __package__ in (None, ""):
    _REPO = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    for _path in (os.path.join(_REPO, "rl"), _REPO):
        if _path not in sys.path:
            sys.path.insert(0, _path)

from ldm_rl import EnvConfig  # noqa: E402
from ldm_rl.factories import build_env  # noqa: E402

from tasks.nanogpt.core.rl_adapter import MockPlumbingEvaluator  # noqa: E402

FREE_KNOBS = ["DEPTH", "MATRIX_LR", "DEVICE_BATCH_SIZE"]
ROUNDS = 5
RESERVOIR = 2

# Top of the mock band, so most mock scores fall below it and the reward path
# is actually visible here. In a real run this is the MEASURED val_bpb of the
# instance's default config -- see tasks.nanogpt.scripts.rl_reference.
REFERENCE = MockPlumbingEvaluator.BAND[1]


def main() -> int:
    env = build_env(
        "nanogpt",
        mode="mock",
        config=EnvConfig(
            iterations=ROUNDS,
            reservoir_size=RESERVOIR,
            evaluations_per_round=1,
            reward="improvement",
            # Oriented space: val_bpb is minimised, so the reference is negated.
            reward_ref_point=(-REFERENCE,),
        ),
        free_knobs=FREE_KNOBS,
        reservoir_size=RESERVOIR,
    )

    prompt = env.reset()
    print("=" * 70)
    print(f"reset prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens)")
    print(f"reference val_bpb (mock): {REFERENCE:.6f}")
    print("=" * 70)

    def knobs(depth, lr, device_batch=128):
        return {"DEPTH": depth, "MATRIX_LR": lr, "DEVICE_BATCH_SIZE": device_batch}

    # Each round exercises a different path on purpose; see the notes printed
    # at the end.
    actions = [
        ("two legal proposals",
         json.dumps({"proposals": [knobs(6, 0.02), knobs(7, 0.024)]})),
        ("two legal proposals",
         json.dumps({"proposals": [knobs(10, 0.04), knobs(11, 0.048)]})),
        ("one illegal (DEVICE_BATCH_SIZE=96 divides no TOTAL_BATCH_SIZE)",
         json.dumps({"proposals": [knobs(12, 0.05, 96), knobs(13, 0.06)]})),
        ("re-proposes round 0, must be de-duplicated",
         json.dumps({"proposals": [knobs(6, 0.02), knobs(7, 0.024)]})),
        ("reasoning around the JSON, must fail the strict parser",
         'Let me think about depth... I propose '
         + json.dumps({"proposals": [knobs(9, 0.03), knobs(8, 0.028)]})),
    ]

    total = 0.0
    for index, (note, action) in enumerate(actions[:ROUNDS]):
        step = env.step(action)
        total += step.reward
        measured = [
            f"{item['evaluation']['metrics']['val_bpb']:.6f}"
            for item in step.info["evaluated"]
            if item["evaluation"]["status"] == "succeeded"
        ]
        rejected = [item["reason"] for item in step.info["rejections"]]
        print(f"round {index}: reward {step.reward:+.6f} | measured {measured or '-'}")
        if rejected:
            print(f"          rejected: {rejected}")
        if step.info.get("parse_error"):
            print(f"          parse_error: {step.info['parse_error']}")
        print(f"          ({note})")
        if step.done:
            print(f"  episode ended: {step.info['stop_reason']}")
            break

    print("-" * 70)
    print(f"episode reward {total:+.6f}")
    print(
        "Expected: round 2 rejects one proposal as rejected_by_trainer; round 3\n"
        "rejects ONE as already_evaluated; round 4 fails to parse and scores 0.\n"
        "\n"
        "Round 3 rejecting only one is correct, and worth knowing: the reservoir\n"
        "de-duplicates against candidates that were EVALUATED, not against every\n"
        "candidate ever proposed. With evaluations_per_round=1 only one of round\n"
        "0's two proposals was measured, so the other is still unmeasured and may\n"
        "legitimately be re-proposed and measured now."
    )
    print(
        "\nThese val_bpb values are hashes, not measurements. Plumbing only.\n"
        "Next: rl/slime_launch/NANOGPT_RL_TRAINING.md"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
