#!/bin/bash
# Convert Qwen2.5-7B-Instruct HF -> Megatron torch_dist for the dense control run.
# Usage:
#   MODEL_HF=/path/to/Qwen2.5-7B-Instruct SAVE=/path/to/rl/qwen2.5-7B_torch_dist bash convert_dense7b.sh
set -eux
REPO_ROOT=${REPO_ROOT:-/mnt/data0/ys/LDM}
SLIME_ROOT=${SLIME_ROOT:-$REPO_ROOT/rl/slime}
MEGATRON_ROOT=${MEGATRON_ROOT:-/root/megatron-lm}
CONDA_PREFIX=${CONDA_PREFIX:-/root/micromamba/envs/slime}
MODEL_HF=${MODEL_HF:?set MODEL_HF (Qwen2.5-7B-Instruct dir)}
SAVE=${SAVE:-$REPO_ROOT/rl/qwen2.5-7B_torch_dist}

export PATH=$CONDA_PREFIX/bin:$PATH
export PYTHONPATH=$MEGATRON_ROOT:$REPO_ROOT/rl:$REPO_ROOT:$PYTHONPATH
export CUDA_HOME=$CONDA_PREFIX
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd "$SLIME_ROOT"
source "$SLIME_ROOT/scripts/models/qwen2.5-7B.sh"   # sets MODEL_ARGS=(...)
python tools/convert_hf_to_torch_dist.py ${MODEL_ARGS[@]} --hf-checkpoint "$MODEL_HF" --save "$SAVE"
echo "converted -> $SAVE/"
