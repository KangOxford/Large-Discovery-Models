#!/bin/bash
# Example Slime GRPO launch for LDM RL episodes.
#
# This is an adaptation of slime/examples/search-r1/run_qwen2.5_3B.sh. It is
# NOT runnable locally: it needs a GPU cluster with Slime's dependencies
# installed, an sglang serving setup, and a base model checkpoint.
#
# Before launching:
#   1. cd <this repo>; ensure rl/slime is initialized (git submodule update --init)
#   2. pip install -e rl/slime --no-deps  (plus slime's own requirements)
#   3. Generate episode prompt data:
#        python rl/ldm_rl/episodes.py --output rl_episodes.jsonl \
#            --task ai4bio_mutation_effect_prediction --mode mock \
#            --count 64 --iterations 8 --reservoir-size 2
#   4. Point CKPT_ARGS at your model (same format as slime's example scripts).
#
# The rollout worker imports ldm_rl and the LDM repo, so both must be on
# PYTHONPATH (repo root for ldm_tts/tasks, rl/ for ldm_rl).

set -ex

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
export PYTHONPATH="${REPO_ROOT}/rl:${REPO_ROOT}:${PYTHONPATH}"

# Source your model configuration, e.g.:
#   source "${REPO_ROOT}/rl/slime/scripts/models/qwen2.5-3B.sh"

CKPT_ARGS=(
   --hf-checkpoint /path/to/Qwen2.5-3B/
   --ref-load /path/to/Qwen2.5-3B_torch_dist/
   # --save /path/to/ldm_rl_slime/
   # --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data "${REPO_ROOT}/rl_episodes.jsonl"
   --input-key prompt
   --label-key label
   # NOTE: do not pass --apply-chat-template here; the sample prompt is an
   # EpisodeSpec JSON and ldm_rl.bridge.generate applies the chat template
   # itself after rendering the real policy prompt.
   --rollout-shuffle
   --num-rollout 512
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 2048
   --rollout-temperature 1
   --global-batch-size 256
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
)

CUSTOM_ARGS=(
   --custom-generate-function-path ldm_rl.bridge.generate
   --custom-rm-path ldm_rl.bridge.reward_func
)

python "${REPO_ROOT}/rl/slime/train.py" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}" \
   "$@"
