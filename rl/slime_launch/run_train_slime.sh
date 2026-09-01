#!/bin/bash
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


# Block the system CUDA 13.0 libcudart.so.13 (visible via ldconfig) so the
# cudnn-frontend check inside transformer_engine only finds libcudart.so.12.
mkdir -p ${CUDART_BLOCK:-/root/cudart_block}
touch ${CUDART_BLOCK:-/root/cudart_block}/libcudart.so.13

export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=${CUDART_BLOCK:-/root/cudart_block}:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
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
   --hf-checkpoint ${HF_MODELS:-/mnt/data0/hf_models/models}/Qwen2.5-1.5B-Instruct
   --ref-load $REPO_ROOT/rl/qwen2.5-1.5B_torch_dist_te
   --save $REPO_ROOT/rl/qwen2.5-1.5B_slime_train
   --save-interval 10
)

ROLLOUT_ARGS=(
   --prompt-data $REPO_ROOT/rl_episodes_sm.jsonl
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
