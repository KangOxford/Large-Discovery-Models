#!/bin/bash
# Warm-up phase: rollout-only (inference + evaluate), NO model update.
# Populates the shared GP history file that the training phase then reads.
set -ex
export PYTHONUNBUFFERED=1

REPO_ROOT=${REPO_ROOT:-/mnt/data0/ys/LDM}
SLIME_ROOT=$REPO_ROOT/rl/slime
MEGATRON_ROOT=${MEGATRON_ROOT:-/root/megatron-lm}
CONDA_PREFIX=${CONDA_PREFIX:-/root/micromamba/envs/slime}
CONFIG=${CONFIG:-$REPO_ROOT/rl/slime_launch/config_real.json}

mkdir -p ${CUDART_BLOCK:-/root/cudart_block}
touch ${CUDART_BLOCK:-/root/cudart_block}/libcudart.so.13

export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=${CUDART_BLOCK:-/root/cudart_block}:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$MEGATRON_ROOT:$REPO_ROOT/rl:$REPO_ROOT:$PYTHONPATH
export CUDA_HOME=$CONDA_PREFIX
export CUDA_VISIBLE_DEVICES=2,3
export CUDA_DEVICE_MAX_CONNECTIONS=1

WARMUP_SAMPLES=$(python3 -c "import json; print(json.load(open('$CONFIG'))['warmup']['num_samples'])")

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
)

ROLLOUT_ARGS=(
   --prompt-data $REPO_ROOT/rl_episodes_sm_warmup.jsonl
   --input-key prompt --label-key label
   --num-rollout "$WARMUP_SAMPLES" --rollout-batch-size 1 --n-samples-per-prompt 1
   --rollout-max-response-len 1024 --rollout-temperature 0.8
   --global-batch-size 1 --debug-rollout-only
)

PERF_ARGS=(
   --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1
   --use-dynamic-batch-size --max-tokens-per-gpu 8192
)

APEX_ARGS=(--no-gradient-accumulation-fusion)

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

# `ray start` 返回只表示 raylet 进程起来了,**不表示它已经能接受 worker 注册**。
# 2026-09-01 在 GH200 上量过:ray start 返回之后还要 **约 66 秒**驱动才连得上。
# 落在这个窗口里的驱动**不会超时报错,而是无限期阻塞** —— 实测 P0 与暖机两次
# 都停在 ray._raylet.CoreWorker(...) 构造里,主线程的内核等待点是
# unix_stream_read_generic(在等 raylet 经 Unix socket 的回复),CPU 0%、GPU 全空、
# 日志一行不出。所以它看起来像死锁而不像竞态。
#
# 注意 `ray status` **不能**当就绪判据:实测它在驱动还连不上的时候就已经能通了
# (它只经 GCS)。唯一可靠的判据是**真的用驱动连一次**,也就是下面这个探针。
_ray_wait_ready() {
    local i t0 addr
    t0=$(date +%s)
    for i in $(seq 1 40); do
        if timeout 20 python3 -c "import ray; ray.init(address='auto', log_to_driver=False); ray.shutdown()" \
             >/dev/null 2>&1; then
            echo "[ray] 驱动可连(ray start 返回后 $(( $(date +%s) - t0 ))s)"
            return 0
        fi
    done
    echo "FATAL: ray start 之后 $(( $(date +%s) - t0 ))s 驱动仍连不上"
    return 1
}
_ray_wait_ready || exit 3


RUNTIME_ENV_JSON="{\"env_vars\": {\"PYTHONPATH\": \"$MEGATRON_ROOT:$REPO_ROOT/rl:$REPO_ROOT\", \"LD_LIBRARY_PATH\": \"${CUDART_BLOCK:-/root/cudart_block}:$CONDA_PREFIX/lib\", \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"}}"

# 两条路径,由 SLIME_LAUNCH_MODE 选。默认 jobsubmit(作者原样)。
# 本机上 `ray job submit` 以 504 结束:创建作业的 POST 在 dashboard 的 JobHead
# 子进程模块里超时(实测等 5 分钟),而 raylet/gcs_server 一直健康 —— 集群没问题,
# 只有提交这条路不通。单节点本地集群不需要 job server:同一个进程 ray.init 连上
# 已起的集群即可,runtime_env 里那几个环境变量本来就在当前 shell 里 export 过。
if [ "${SLIME_LAUNCH_MODE:-jobsubmit}" = "direct" ]; then
   echo "[launch] 直接跑 train.py(跳过 ray job submit;单节点本地集群不需要 job server)"
   RAY_ADDRESS=auto python3 train.py \
   --actor-num-nodes 1 --actor-num-gpus-per-node 0 --rollout-num-gpus 2 \
   ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ${ROLLOUT_ARGS[@]} \
   ${PERF_ARGS[@]} ${APEX_ARGS[@]} ${SGLANG_ARGS[@]} ${CUSTOM_ARGS[@]}
else
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="$RUNTIME_ENV_JSON" \
   -- python3 train.py \
   --actor-num-nodes 1 --actor-num-gpus-per-node 0 --rollout-num-gpus 2 \
   ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ${ROLLOUT_ARGS[@]} \
   ${PERF_ARGS[@]} ${APEX_ARGS[@]} ${SGLANG_ARGS[@]} ${CUSTOM_ARGS[@]}
fi
