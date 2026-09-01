#!/bin/bash
# Slime real GRPO training over the small_molecule RL environment.
#
# 3 GPUs (CUDA_VISIBLE_DEVICES=0,1,2,3): 1 for the Megatron actor (TP=1),
# 2 for the SGLang rollout engines. Reference weights come from the converted
# torch_dist checkpoint; training checkpoints are saved under --save.
#
# Requires Transformer Engine (see rl/SLIME_TRAINING.md §1.6). TE is compiled
# from source with the pip cuda-toolkit/cuDNN/NCCL packages, whose shared libs
# live under the nvidia/ site-packages subdirs and must be on LD_LIBRARY_PATH.
#
# All paths are injected via environment variables with placeholder defaults so
# no server-specific paths are committed. Override the ones below for your node.
set -ex
export PYTHONUNBUFFERED=1

# --- Overridable environment (defaults are placeholders, not real paths) ---
REPO_ROOT=${REPO_ROOT:-"${LDM_REPO_ROOT:-/path/to/LDM}"}
CONDA_BIN="${CONDA_ENV_BIN:-/path/to/conda_env/bin}"
CONDA_NVIDIA="${CONDA_NVIDIA_LIB:-/path/to/conda_env/lib/python3.12/site-packages/nvidia}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/Qwen2.5-1.5B-Instruct}"
TORCH_DIST_CKPT="${TORCH_DIST_CKPT:-$REPO_ROOT/rl/qwen2.5-1.5B_torch_dist_te}"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/rl/qwen2.5-1.5B_slime_train}"
PROMPT_DATA="${PROMPT_DATA:-$REPO_ROOT/rl_episodes_sm.jsonl}"

SLIME_ROOT=$REPO_ROOT/rl/slime
MEGATRON_ROOT=$REPO_ROOT/rl/megatron-lm
NV=$CONDA_NVIDIA

export PATH=$CONDA_BIN:$PATH
export PYTHONPATH=$MEGATRON_ROOT:$REPO_ROOT/rl:$REPO_ROOT:$PYTHONPATH
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cudnn/lib:$NV/cu13/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
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
   --save "$SAVE_DIR"
   --save-interval 10
)

ROLLOUT_ARGS=(
   --prompt-data "$PROMPT_DATA"
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

# APEX's fused_weight_gradient_mlp CUDA ext is not installed; disable the
# gradient-accumulation fusion it backs (independent of TE).
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


RUNTIME_ENV_JSON="{\"env_vars\": {\"PYTHONPATH\": \"$MEGATRON_ROOT:$REPO_ROOT/rl:$REPO_ROOT\", \"LD_LIBRARY_PATH\": \"$NV/nccl/lib:$NV/cudnn/lib:$NV/cu13/lib\", \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"}}"

# 两条路径,由 SLIME_LAUNCH_MODE 选。默认 jobsubmit(作者原样)。
# 本机上 `ray job submit` 以 504 结束:创建作业的 POST 在 dashboard 的 JobHead
# 子进程模块里超时(实测等 5 分钟),而 raylet/gcs_server 一直健康 —— 集群没问题,
# 只有提交这条路不通。单节点本地集群不需要 job server:同一个进程 ray.init 连上
# 已起的集群即可,runtime_env 里那几个环境变量本来就在当前 shell 里 export 过。
if [ "${SLIME_LAUNCH_MODE:-jobsubmit}" = "direct" ]; then
   echo "[launch] 直接跑 train.py(跳过 ray job submit;单节点本地集群不需要 job server)"
   RAY_ADDRESS=auto python3 train.py \
   --actor-num-nodes 1 --actor-num-gpus-per-node 1 --rollout-num-gpus 2 \
   ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ${ROLLOUT_ARGS[@]} \
   ${PERF_ARGS[@]} ${GRPO_ARGS[@]} ${OPTIMIZER_ARGS[@]} \
   ${APEX_ARGS[@]} ${SGLANG_ARGS[@]} ${CUSTOM_ARGS[@]}
else
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="$RUNTIME_ENV_JSON" \
   -- python3 train.py \
   --actor-num-nodes 1 --actor-num-gpus-per-node 1 --rollout-num-gpus 2 \
   ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ${ROLLOUT_ARGS[@]} \
   ${PERF_ARGS[@]} ${GRPO_ARGS[@]} ${OPTIMIZER_ARGS[@]} \
   ${APEX_ARGS[@]} ${SGLANG_ARGS[@]} ${CUSTOM_ARGS[@]}
fi
