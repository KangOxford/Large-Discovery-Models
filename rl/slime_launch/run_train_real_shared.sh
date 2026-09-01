#!/bin/bash
# Real-mode GRPO training with a SHARED (warm-up + continuously updated) GP.
# Reads config_real.json; the prompt data already carries gp_history_file in
# each episode's real kwargs, so the adapter wires the shared GP automatically.
set -ex
export PYTHONUNBUFFERED=1

REPO_ROOT=${REPO_ROOT:-/mnt/data0/ys/LDM}
SLIME_ROOT=$REPO_ROOT/rl/slime
MEGATRON_ROOT=${MEGATRON_ROOT:-/root/megatron-lm}
CONDA_PREFIX=${CONDA_PREFIX:-/root/micromamba/envs/slime}
# 与 run_train_real_9b.sh 一致的可覆盖入口。gen_episodes_runs.sh 现在按 run 产出
# rl_episodes_sm_{warmup,acqmax,acqmean,hv}.jsonl,不再产出单一的 _real.jsonl,
# 所以这里必须能从外面指定用哪个。
EPISODES=${EPISODES:-$REPO_ROOT/rl_episodes_sm_real.jsonl}

CONFIG=${CONFIG:-$REPO_ROOT/rl/slime_launch/config_real.json}

mkdir -p ${CUDART_BLOCK:-/root/cudart_block}
touch ${CUDART_BLOCK:-/root/cudart_block}/libcudart.so.13

export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=${CUDART_BLOCK:-/root/cudart_block}:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$MEGATRON_ROOT:$REPO_ROOT/rl:$REPO_ROOT:$PYTHONPATH
export CUDA_HOME=$CONDA_PREFIX
export CUDA_VISIBLE_DEVICES=1,2,3
export CUDA_DEVICE_MAX_CONNECTIONS=1

NUM_ROLLOUT=$(python3 -c "import json; print(json.load(open('$CONFIG'))['training']['num_rollout'])")
ROLLOUT_BATCH=$(python3 -c "import json; print(json.load(open('$CONFIG'))['training']['rollout_batch_size'])")
N_SAMPLES=$(python3 -c "import json; print(json.load(open('$CONFIG'))['training']['n_samples_per_prompt'])")
GLOBAL_BATCH=$(python3 -c "import json; print(json.load(open('$CONFIG'))['training']['global_batch_size'])")
RESP_LEN=$(python3 -c "import json; print(json.load(open('$CONFIG'))['training']['rollout_max_response_len'])")
MAX_TOKENS=$(python3 -c "import json; print(json.load(open('$CONFIG'))['training']['max_tokens_per_gpu'])")
TEMPERATURE=$(python3 -c "import json; print(json.load(open('$CONFIG'))['training']['rollout_temperature'])")
LR=$(python3 -c "import json; print(json.load(open('$CONFIG'))['training']['lr'])")
SAVE_INTERVAL=$(python3 -c "import json; print(json.load(open('$CONFIG'))['training']['save_interval'])")

cd "$SLIME_ROOT"

MODEL_ARGS=(
   --swiglu --num-layers 28 --hidden-size 1536 --ffn-hidden-size 8960
   --num-attention-heads 12 --use-rotary-position-embeddings --disable-bias-linear
   --add-qkv-bias --normalization "RMSNorm" --norm-epsilon 1e-6 --rotary-base 1000000
   --group-query-attention --num-query-groups 2 --vocab-size 151936
)

CKPT_ARGS=(
   --hf-checkpoint ${HF_MODELS:-/mnt/data0/hf_models/models}/Qwen2.5-1.5B-Instruct
   --ref-load $REPO_ROOT/rl/qwen2.5-1.5B_torch_dist_te
   --save $REPO_ROOT/rl/qwen2.5-1.5B_slime_train_shared
   --save-interval "$SAVE_INTERVAL"
)

ROLLOUT_ARGS=(
   --prompt-data "$EPISODES"
   --input-key prompt --label-key label
   --num-rollout "$NUM_ROLLOUT" --rollout-batch-size "$ROLLOUT_BATCH" --n-samples-per-prompt "$N_SAMPLES"
   --rollout-max-response-len "$RESP_LEN" --rollout-temperature "$TEMPERATURE"
   --global-batch-size "$GLOBAL_BATCH" --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1 --context-parallel-size 1
   --use-dynamic-batch-size --max-tokens-per-gpu "$MAX_TOKENS"
)

APEX_ARGS=(--no-gradient-accumulation-fusion)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl
   --eps-clip 0.2 --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam --lr "$LR" --lr-decay-style constant
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

RUNTIME_ENV_JSON="{\"env_vars\": {\"PYTHONPATH\": \"$MEGATRON_ROOT:$REPO_ROOT/rl:$REPO_ROOT\", \"LD_LIBRARY_PATH\": \"${CUDART_BLOCK:-/root/cudart_block}:$CONDA_PREFIX/lib\", \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"}}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="$RUNTIME_ENV_JSON" \
   -- python3 train.py \
   --actor-num-nodes 1 --actor-num-gpus-per-node 1 --rollout-num-gpus 2 \
   ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ${ROLLOUT_ARGS[@]} \
   ${PERF_ARGS[@]} ${GRPO_ARGS[@]} ${OPTIMIZER_ARGS[@]} \
   ${APEX_ARGS[@]} ${SGLANG_ARGS[@]} ${CUSTOM_ARGS[@]}
