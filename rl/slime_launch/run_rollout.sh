#!/bin/bash
# Slime rollout-only launch (validates the full pipeline without training).
#
# Runs ray + rollout manager + SGLang engine + Megatron actor (loads the
# torch_dist reference weights and pushes them to SGLang) + ldm_rl.bridge.generate
# for one small_molecule mock episode.
#
# See rl/SLIME_TRAINING.md for the environment setup and the local-impl patches
# this script depends on.
#
# All paths are injected via environment variables with placeholder defaults so
# no server-specific paths are committed. Override the ones below for your node.
set -ex
export PYTHONUNBUFFERED=1

# --- Overridable environment (defaults are placeholders, not real paths) ---
REPO_ROOT="${LDM_REPO_ROOT:-/path/to/LDM}"
CONDA_BIN="${CONDA_ENV_BIN:-/path/to/conda_env/bin}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/Qwen2.5-1.5B-Instruct}"
TORCH_DIST_CKPT="${TORCH_DIST_CKPT:-$REPO_ROOT/rl/qwen2.5-1.5B_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-$REPO_ROOT/rl_episodes_sm.jsonl}"

SLIME_ROOT=$REPO_ROOT/rl/slime
MEGATRON_ROOT=$REPO_ROOT/rl/megatron-lm

export PATH=$CONDA_BIN:$PATH
export PYTHONPATH=$MEGATRON_ROOT:$REPO_ROOT/rl:$REPO_ROOT:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
export CUDA_DEVICE_MAX_CONNECTIONS=1

cd "$SLIME_ROOT"

MODEL_ARGS=(
   --swiglu --num-layers 28 --hidden-size 1536 --ffn-hidden-size 8960
   --num-attention-heads 12 --use-rotary-position-embeddings --disable-bias-linear
   --add-qkv-bias --normalization "RMSNorm" --norm-epsilon 1e-6 --rotary-base 1000000
   --group-query-attention --num-query-groups 2 --vocab-size 151936
)

CKPT_ARGS=(
   --hf-checkpoint "$HF_CHECKPOINT"
   --ref-load "$TORCH_DIST_CKPT"
)

ROLLOUT_ARGS=(
   --prompt-data "$PROMPT_DATA"
   --input-key prompt --label-key label
   --num-rollout 1 --rollout-batch-size 1 --n-samples-per-prompt 1
   --rollout-max-response-len 512 --rollout-temperature 0.8
   --global-batch-size 1 --debug-rollout-only
)

PERF_ARGS=(
   --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1
   --use-dynamic-batch-size --max-tokens-per-gpu 4096
)

# local (no TE/APEX) implementation flags -- see SLIME_TRAINING.md §1.3
LOCAL_IMPL_ARGS=(
   --no-rope-fusion --no-persist-layer-norm --no-gradient-accumulation-fusion
)

SGLANG_ARGS=(
   --rollout-num-gpus 2 --sglang-mem-fraction-static 0.7
)

CUSTOM_ARGS=(
   --custom-generate-function-path ldm_rl.bridge.generate
   --custom-rm-path ldm_rl.bridge.reward_func
)

ray stop --force 2>/dev/null || true
sleep 3
ray start --head --node-ip-address 127.0.0.1 --num-gpus 2 --disable-usage-stats

RUNTIME_ENV_JSON="{\"env_vars\": {\"PYTHONPATH\": \"$MEGATRON_ROOT:$REPO_ROOT/rl:$REPO_ROOT\", \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"}}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="$RUNTIME_ENV_JSON" \
   -- python3 train.py \
   --actor-num-nodes 1 --actor-num-gpus-per-node 0 --rollout-num-gpus 2 \
   ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ${ROLLOUT_ARGS[@]} \
   ${PERF_ARGS[@]} ${LOCAL_IMPL_ARGS[@]} ${SGLANG_ARGS[@]} ${CUSTOM_ARGS[@]}
