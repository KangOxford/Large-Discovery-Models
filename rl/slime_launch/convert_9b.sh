#!/bin/bash
# Convert a Qwen3.5-9B HF checkpoint -> Megatron torch_dist (ref-load format).
# Usage:
#   MODEL_HF=/mnt/data0/hf_models/models/Qwen3.5-9B \
#   SAVE=/mnt/data0/ys/LDM/rl/qwen3.5-9B_torch_dist  bash convert_9b.sh
set -ex
REPO_ROOT=/mnt/data0/ys/LDM
SLIME_ROOT=$REPO_ROOT/rl/slime
MEGATRON_ROOT=/root/megatron-lm
CONDA_PREFIX=/root/micromamba/envs/slime

MODEL_HF=${MODEL_HF:-/mnt/data0/hf_models/models/Qwen3.5-9B}
SAVE=${SAVE:-$REPO_ROOT/rl/qwen3.5-9B_torch_dist}

export PATH=$CONDA_PREFIX/bin:$PATH
export PYTHONPATH=$MEGATRON_ROOT:$REPO_ROOT/rl:$REPO_ROOT:$PYTHONPATH
export CUDA_HOME=$CONDA_PREFIX
export CUDA_VISIBLE_DEVICES=0

cd "$SLIME_ROOT"
source "$SLIME_ROOT/scripts/models/qwen3.5-9B.sh"   # sets MODEL_ARGS=(...)

python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint "$MODEL_HF" \
   --save "$SAVE"
echo "converted -> $SAVE/"
