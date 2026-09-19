#!/bin/bash
set -ex
export PYTHONUNBUFFERED=1

REPO_ROOT=/mnt/data0/ys/LDM
SLIME_ROOT=$REPO_ROOT/rl/slime
MEGATRON_ROOT=/root/megatron-lm
CONDA_PREFIX=/root/micromamba/envs/slime

# Block the system CUDA 13.0 libcudart.so.13 (visible via ldconfig) so the
# cudnn-frontend check inside transformer_engine only finds libcudart.so.12.
mkdir -p /root/cudart_block
touch /root/cudart_block/libcudart.so.13

export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=/root/cudart_block:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$MEGATRON_ROOT:$REPO_ROOT/rl:$REPO_ROOT:$PYTHONPATH
export CUDA_HOME=$CONDA_PREFIX
export CUDA_VISIBLE_DEVICES=1,2,3
export CUDA_DEVICE_MAX_CONNECTIONS=1

cd "$SLIME_ROOT"

MODEL_ARGS=(
   --swiglu --num-layers 28 --hidden-size 1536 --ffn-hidden-size 8960
   --num-attention-heads 12 --use-rotary-position-embeddings --disable-bias-linear
   --add-qkv-bias --normalization "RMSNorm" --norm-epsilon 1e-6 --rotary-base 1000000
   --group-query-attention --num-query-groups 2 --vocab-size 151936
)

CKPT_ARGS=(
   --hf-checkpoint /mnt/data0/hf_models/models/Qwen2.5-1.5B-Instruct
   --ref-load /mnt/data0/ys/LDM/rl/qwen2.5-1.5B_torch_dist_te
   --save /mnt/data0/ys/LDM/rl/qwen2.5-1.5B_slime_train
   --save-interval 10
)

ROLLOUT_ARGS=(
   --prompt-data /mnt/data0/ys/LDM/rl_episodes_sm.jsonl
   --input-key prompt --label-key label
   --num-rollout 2 --rollout-batch-size 2 --n-samples-per-prompt 1
   --rollout-max-response-len 512 --rollout-temperature 0.8
   --global-batch-size 2 --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1 --context-parallel-size 1
   --use-dynamic-batch-size --max-tokens-per-gpu 4096
)

APEX_ARGS=(--no-gradient-accumulation-fusion)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl
   --eps-clip 0.2 --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam --lr 1e-6 --lr-decay-style constant
   --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.98
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
ray start --head --node-ip-address 127.0.0.1 --num-gpus 3 --disable-usage-stats

RUNTIME_ENV_JSON="{\"env_vars\": {\"PYTHONPATH\": \"$MEGATRON_ROOT:$REPO_ROOT/rl:$REPO_ROOT\", \"LD_LIBRARY_PATH\": \"/root/cudart_block:$CONDA_PREFIX/lib\", \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"}}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="$RUNTIME_ENV_JSON" \
   -- python3 train.py \
   --actor-num-nodes 1 --actor-num-gpus-per-node 1 --rollout-num-gpus 2 \
   ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ${ROLLOUT_ARGS[@]} \
   ${PERF_ARGS[@]} ${GRPO_ARGS[@]} ${OPTIMIZER_ARGS[@]} \
   ${APEX_ARGS[@]} ${SGLANG_ARGS[@]} ${CUSTOM_ARGS[@]}
