#!/bin/bash
# Real-mode GRPO training for Qwen3.5-9B (hybrid), parameterised for the
# base-vs-SFT x reward-variant run matrix (see RUNS.md).
#
# The reward policy (acquisition-max / acquisition-mean / improvement) is baked
# into the episode prompt-data by gen_episodes_runs.sh, so this launcher only
# needs the model + episode file + save dir. Override via env vars:
#
#   MODEL_HF   HF checkpoint dir              (default: Qwen3.5-9B base)
#   MODEL_REF  Megatron torch_dist ref-load   (default: qwen3.5-9B_torch_dist)
#   EPISODES   prompt-data jsonl              (default: rl_episodes_sm_real.jsonl)
#   SAVE       checkpoint out dir            (default: rl/qwen3.5-9B_slime_train)
#   WANDB_KEY  if set -> enables wandb logging
#   WANDB_PROJECT / WANDB_RUN  wandb names (defaults below)
set -ex
export PYTHONUNBUFFERED=1

REPO_ROOT=${REPO_ROOT:-/mnt/data0/ys/LDM}
SLIME_ROOT=$REPO_ROOT/rl/slime
MEGATRON_ROOT=${MEGATRON_ROOT:-/root/megatron-lm}
CONDA_PREFIX=${CONDA_PREFIX:-/root/micromamba/envs/slime}
CONFIG=${CONFIG:-$REPO_ROOT/rl/slime_launch/config_real.json}

MODEL_HF=${MODEL_HF:-/mnt/data0/hf_models/models/Qwen3.5-9B}
MODEL_REF=${MODEL_REF:-$REPO_ROOT/rl/qwen3.5-9B_torch_dist}
EPISODES=${EPISODES:-$REPO_ROOT/rl_episodes_sm_real.jsonl}
SAVE=${SAVE:-$REPO_ROOT/rl/qwen3.5-9B_slime_train}
WANDB_PROJECT=${WANDB_PROJECT:-ldm-sm-rl}
WANDB_RUN=${WANDB_RUN:-$(basename "$SAVE")}

mkdir -p ${CUDART_BLOCK:-/root/cudart_block}
touch ${CUDART_BLOCK:-/root/cudart_block}/libcudart.so.13

export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=${CUDART_BLOCK:-/root/cudart_block}:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$MEGATRON_ROOT:$REPO_ROOT/rl:$REPO_ROOT:$PYTHONPATH
export CUDA_HOME=$CONDA_PREFIX
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_MAX_CONNECTIONS=1

jq_get() { python3 -c "import json;print(json.load(open('$CONFIG'))['training']['$1'])"; }
NUM_ROLLOUT=$(jq_get num_rollout)
ROLLOUT_BATCH=$(jq_get rollout_batch_size)
N_SAMPLES=$(jq_get n_samples_per_prompt)
GLOBAL_BATCH=$(jq_get global_batch_size)
RESP_LEN=$(jq_get rollout_max_response_len)
MAX_TOKENS=$(jq_get max_tokens_per_gpu)
TEMPERATURE=$(jq_get rollout_temperature)
LR=$(jq_get lr)
SAVE_INTERVAL=$(jq_get save_interval)

cd "$SLIME_ROOT"

# Qwen3.5-9B hybrid (Gated DeltaNet + MTP) architecture args.
source "$SLIME_ROOT/scripts/models/qwen3.5-9B.sh"   # sets MODEL_ARGS=(...)

CKPT_ARGS=(
   --hf-checkpoint "$MODEL_HF"
   --ref-load "$MODEL_REF"
   --save "$SAVE"
   --save-interval "$SAVE_INTERVAL"
)

ROLLOUT_ARGS=(
   --prompt-data "$EPISODES"
   --input-key prompt --label-key label
   --num-rollout "$NUM_ROLLOUT" --rollout-batch-size "$ROLLOUT_BATCH" --n-samples-per-prompt "$N_SAMPLES"
   --rollout-max-response-len "$RESP_LEN" --rollout-temperature "$TEMPERATURE"
   --global-batch-size "$GLOBAL_BATCH" --balance-data
)

# 4 GPUs: TP=2 actor (2 GPUs) + 2 GPUs for the sglang rollout engine.
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --pipeline-model-parallel-size 1 --context-parallel-size 1
   --use-distributed-optimizer
   --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
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

WANDB_ARGS=()
if [[ -n "${WANDB_KEY:-}" ]]; then
   WANDB_ARGS=(--use-wandb --wandb-project "$WANDB_PROJECT" --wandb-key "$WANDB_KEY" --wandb-run-name "$WANDB_RUN")
fi

ray stop --force 2>/dev/null || true
sleep 3
ray start --head --node-ip-address 127.0.0.1 --num-gpus 4 --disable-usage-stats

# `ray start` 返回只表示 raylet 起来了,**不表示 job 提交服务已经能收请求**。
# 2026-09-01 在 GH200 上实测:ray start 07:30:48 报成功,6 秒后 ray job submit
# 拿到 504 Gateway Timeout —— 端口在听,后面的 dashboard agent 还没起。
# 作者机器上启动快,撞不上;换台机器就是必挂。所以显式等它就绪。
_ray_addr=${RAY_DASHBOARD:-http://127.0.0.1:8265}
for _i in $(seq 1 60); do
    if curl -s -o /dev/null -w "%{http_code}" --max-time 3 "$_ray_addr/api/version" 2>/dev/null | grep -q "^200$"; then
        echo "[ray] job 提交服务就绪($_ray_addr,等了 $((_i*2))s)"; break
    fi
    [ "$_i" = 60 ] && { echo "FATAL: 等了 120s,$_ray_addr 仍未就绪"; exit 3; }
    sleep 2
done


RUNTIME_ENV_JSON="{\"env_vars\": {\"PYTHONPATH\": \"$MEGATRON_ROOT:$REPO_ROOT/rl:$REPO_ROOT\", \"LD_LIBRARY_PATH\": \"${CUDART_BLOCK:-/root/cudart_block}:$CONDA_PREFIX/lib\", \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"}}"

# 两条路径,由 SLIME_LAUNCH_MODE 选。默认 jobsubmit(作者原样)。
# 本机上 `ray job submit` 以 504 结束:创建作业的 POST 在 dashboard 的 JobHead
# 子进程模块里超时(实测等 5 分钟),而 raylet/gcs_server 一直健康 —— 集群没问题,
# 只有提交这条路不通。单节点本地集群不需要 job server:同一个进程 ray.init 连上
# 已起的集群即可,runtime_env 里那几个环境变量本来就在当前 shell 里 export 过。
if [ "${SLIME_LAUNCH_MODE:-jobsubmit}" = "direct" ]; then
   echo "[launch] 直接跑 train.py(跳过 ray job submit;单节点本地集群不需要 job server)"
   RAY_ADDRESS=auto python3 train.py \
   --actor-num-nodes 1 --actor-num-gpus-per-node 2 --rollout-num-gpus 2 \
   ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ${ROLLOUT_ARGS[@]} \
   ${PERF_ARGS[@]} ${GRPO_ARGS[@]} ${OPTIMIZER_ARGS[@]} \
   ${APEX_ARGS[@]} ${SGLANG_ARGS[@]} ${CUSTOM_ARGS[@]} ${WANDB_ARGS[@]}
else
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="$RUNTIME_ENV_JSON" \
   -- python3 train.py \
   --actor-num-nodes 1 --actor-num-gpus-per-node 2 --rollout-num-gpus 2 \
   ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ${ROLLOUT_ARGS[@]} \
   ${PERF_ARGS[@]} ${GRPO_ARGS[@]} ${OPTIMIZER_ARGS[@]} \
   ${APEX_ARGS[@]} ${SGLANG_ARGS[@]} ${CUSTOM_ARGS[@]} ${WANDB_ARGS[@]}
fi
