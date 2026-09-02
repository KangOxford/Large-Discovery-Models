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
# 卡的布局。默认是作者的分卡模式(actor 2 张 TP=2 + sglang 2 张)。
#
# 为什么需要 colocate:2026-09-01 在 4×GH200(95GB/卡)上实测,分卡模式的 actor 侧
# **显存不够**。每个 TP rank 持 5,285,884,416 参数,按 Adam 的账
#   bf16 权重 2 + fp32 主权重 4 + exp_avg 4 + exp_avg_sq 4 = 14 字节/参数
#   5.29e9 x 14 ≈ 74 GB,再加激活与碎片就超过 95 GB。
# 实际的 OOM 发生在 TE 的 fused_adam 惰性分配 exp_avg_sq 那一步:
#   torch.OutOfMemoryError: Tried to allocate 1.89 GiB ... 95.00 GiB 中仅剩 490 MiB
# 注意它**很晚才暴露** —— 模型加载、前向、反向(日志里有 run_backward)全过了,
# 卡在第一次参数更新。所以 HANDOFF §7 的"9B + TP2/4 于 96GB/卡宽裕"不成立。
#
# colocate 让 actor 与 sglang 共用同一批卡:actor 用满 4 张(TP=4,每卡参数量减半),
# sglang 也在这 4 张上,由 --offload(colocate 自动打开)在训练/生成之间换出显存。
# 这正是 torch_memory_saver 的用途,也正是 GH200 的强项 —— NVLink-C2C 的
# CPU<->GPU 带宽约 450 GB/s,比 x86 的 PCIe 快约 7 倍,换出代价小得多。
#
# --num-gpus-per-node 必须显式给:它默认 8,而本机每节点 4 张。不给不会报错,
# 但 get_base_gpu_id 里的 % num_gpus_per_node 会算出错误的卡号(静默走错分支)。
if [ "${SLIME_COLOCATE:-0}" = "1" ]; then
   PLACEMENT_ARGS=(
      --colocate
      --num-gpus-per-node ${NUM_GPUS_PER_NODE:-4}
      --actor-num-nodes 1
      --actor-num-gpus-per-node ${ACTOR_GPUS:-4}
      --rollout-num-gpus ${ROLLOUT_GPUS:-4}
   )
   : "${TP_SIZE:=${ACTOR_GPUS:-4}}"
   echo "[layout] colocate:actor ${ACTOR_GPUS:-4} 卡 TP=${TP_SIZE} + sglang ${ROLLOUT_GPUS:-4} 卡(同一批),offload 自动开启"
elif [ "${SLIME_SPLIT_NODES:-0}" = "1" ]; then
   # 双节点分卡:actor 独占节点 A 的 4 张(TP=4),sglang 独占节点 B 的 4 张。
   #
   # 为什么不用上面那个 colocate:colocate 要 sglang 能暂停显存,而那条路径
   # 依赖 torch_memory_saver 的 preload 钩子,在本机连续五次死在
   #   AssertionError: Only hook_mode=preload supports pauseable CUDA Graph
   # 断言发生在 sglang 自己 fork 的 scheduler 子进程里(scheduler.py:4325
   # -> decode_cuda_graph_runner.py:698 -> entrypoint.py:158),而
   # `_hook_mode` 的默认值本来就是 "preload",sglang/slime/tms 三个包里
   # 都找不到把它改掉的赋值点 —— 根因未定。
   #
   # 但 torch_memory_saver **只在共卡时才需要**。actor 与 sglang 各占一个
   # 节点就不必暂停任何东西,那个断言所在的代码路径根本不会被走到。
   #
   # 显存也因此宽裕:TP=4 时每 rank 2.64e9 参数 x 14 字节/参数
   # (bf16 权重 2 + fp32 主权重 4 + exp_avg 4 + exp_avg_sq 4) = 37 GB,
   # 独占 95 GB 的卡。而 TP=2 共卡那次是 5.29e9 x 14 = 74 GB 才 OOM 的。
   #
   # 附带收益:省掉 colocate 每步都要做的 offload/reload 权重搬运。
   # (每步的 update_weights 是 RL 必需品,那个省不掉,与此无关。)
   PLACEMENT_ARGS=(
      --num-gpus-per-node ${NUM_GPUS_PER_NODE:-4}
      --actor-num-nodes 1
      --actor-num-gpus-per-node ${ACTOR_GPUS:-4}
      --rollout-num-gpus ${ROLLOUT_GPUS:-4}
   )
   : "${TP_SIZE:=${ACTOR_GPUS:-4}}"
   echo "[layout] 双节点分卡:actor ${ACTOR_GPUS:-4} 卡 TP=${TP_SIZE}(节点 A) + sglang ${ROLLOUT_GPUS:-4} 卡(节点 B),不共卡、不 offload"
else
   PLACEMENT_ARGS=(
      --actor-num-nodes 1
      --actor-num-gpus-per-node ${ACTOR_GPUS:-2}
      --rollout-num-gpus ${ROLLOUT_GPUS:-2}
   )
   echo "[layout] 分卡:actor ${ACTOR_GPUS:-2} 卡 + sglang ${ROLLOUT_GPUS:-2} 卡"
fi

PERF_ARGS=(
   # TP 要与 actor 的卡数一致:colocate 下 actor 用 4 张,每卡参数量才减半到能放下。
   --tensor-model-parallel-size ${TP_SIZE:-2}
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
   # 注意力后端:sglang 不指定时会自选 fa3,而 **aarch64 版的 sglang-kernel 轮子里
   # 没有编 FA3 内核** —— 运行到建引擎那一步才报
   #   ImportError: Can not import FA3 in sgl_kernel
   # 而 pip install 与 import sgl_kernel 都是成功的,所以这件事只有跑起来才暴露。
   # GH200 上实测可用:triton 3.6.0 与 flashinfer 0.6.12。取 triton —— 它与架构
   # 无关,也是 slime 自己的 Dockerfile 在绕不过 FlashQLA 时用的那个。
   --sglang-attention-backend ${SGLANG_ATTENTION_BACKEND:-triton}
   # --rollout-num-gpus 在这里**又给了一次**,而 SGLANG_ARGS 在命令行上排在
   # PLACEMENT_ARGS 后面(见文件末尾 python3 train.py 那几行),argparse 取后者。
   # 于是 PLACEMENT_ARGS 里精心算出来的卡数会被这个硬写的 2 静默覆盖 ——
   # 不报错、不警告,只是 sglang 少拿两张卡。与本文件里 TP_SIZE 曾被
   # PERF_ARGS 的位置吃掉是同一类缺陷:同一个参数在两处给,后一处赢。
   --rollout-num-gpus ${ROLLOUT_GPUS:-2} --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION:-0.7}
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
if [ "${SLIME_SPLIT_NODES:-0}" = "1" ]; then
   # 双节点:head 必须绑**本机的真实 IP**。127.0.0.1 上的 head 只有本节点连得上,
   # 另一个节点的 ray worker 根本路由不到它 —— 而且不会报错,worker 会一直重试
   # 到超时,看起来像"集群起不来"。
   RAY_HEAD_IP=$(hostname -I | awk '{print $1}')
   ray start --head --node-ip-address "$RAY_HEAD_IP" --port ${RAY_PORT:-6379} \
             --num-gpus ${NUM_GPUS_PER_NODE:-4} --disable-usage-stats
   echo "[ray] head 于 $RAY_HEAD_IP:${RAY_PORT:-6379}($(hostname)),等另一节点的 worker 加入"
   # 把地址交给编排脚本,让它去另一个节点起 worker。写临时文件再改名 ——
   # 读方可能在写到一半时读到,那样它会拿到一个截断的 IP 而不是等下一轮。
   if [ -n "${RAY_HEAD_FILE:-}" ]; then
      mkdir -p "$(dirname "$RAY_HEAD_FILE")"
      echo "$RAY_HEAD_IP:${RAY_PORT:-6379}" > "${RAY_HEAD_FILE}.tmp"
      mv "${RAY_HEAD_FILE}.tmp" "$RAY_HEAD_FILE"
   fi
else
   ray start --head --node-ip-address 127.0.0.1 --num-gpus ${NUM_GPUS_PER_NODE:-4} --disable-usage-stats
fi

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

# 双节点:驱动连得上 != 两个节点的卡都注册好了。placement group 要 8 张卡,
# 而 worker 节点的 raylet 是异步加入的。**不等它就直接建 placement group
# 会无限期挂住**(Ray 不会报"资源不够",它会一直等),这与本文件上面记的
# ray start 竞态是同一类问题:看起来像死锁,实际是竞态。
if [ "${SLIME_SPLIT_NODES:-0}" = "1" ]; then
   _need_gpus=$(( ${ACTOR_GPUS:-4} + ${ROLLOUT_GPUS:-4} ))
   echo "[ray] 等集群凑满 ${_need_gpus} 张卡..."
   _t0=$(date +%s)
   for _i in $(seq 1 ${RAY_WAIT_TRIES:-60}); do   # 高负载下 5 分钟不够,可用 RAY_WAIT_TRIES 延长
      _have=$(RAY_ADDRESS=auto timeout 20 python3 -c "
import ray; ray.init(address='auto', log_to_driver=False)
print(int(ray.cluster_resources().get('GPU', 0))); ray.shutdown()" 2>/dev/null | tail -1)
      if [ "${_have:-0}" -ge "$_need_gpus" ] 2>/dev/null; then
         echo "[ray] 集群 ${_have} 张卡就绪($(( $(date +%s) - _t0 ))s)"
         break
      fi
      sleep 5
   done
   if [ "${_have:-0}" -lt "$_need_gpus" ] 2>/dev/null; then
      echo "FATAL: 等了 $(( $(date +%s) - _t0 ))s,集群只有 ${_have:-0} 张卡,要 ${_need_gpus} 张"
      echo "       第二个节点的 ray worker 没起来 —— 查编排脚本那半边"
      exit 4
   fi
fi


# EXTRA_ARGS：临时诊断用的额外命令行参数(空格分隔的一串)。
# 排在最后,所以它可以覆盖前面数组里的同名参数 —— argparse 取后者。
# 例:EXTRA_ARGS="--no-check-for-nan-in-loss-and-grad" 关掉 Megatron 的 NaN 守卫,
# 用来看 loss 本身是否有限(**只能当诊断**:NaN 梯度会污染参数)。
RUNTIME_ENV_JSON="{\"env_vars\": {\"PYTHONPATH\": \"$MEGATRON_ROOT:$REPO_ROOT/rl:$REPO_ROOT\", \"LD_LIBRARY_PATH\": \"${CUDART_BLOCK:-/root/cudart_block}:$CONDA_PREFIX/lib\", \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"}}"

# 两条路径,由 SLIME_LAUNCH_MODE 选。默认 jobsubmit(作者原样)。
# 本机上 `ray job submit` 以 504 结束:创建作业的 POST 在 dashboard 的 JobHead
# 子进程模块里超时(实测等 5 分钟),而 raylet/gcs_server 一直健康 —— 集群没问题,
# 只有提交这条路不通。单节点本地集群不需要 job server:同一个进程 ray.init 连上
# 已起的集群即可,runtime_env 里那几个环境变量本来就在当前 shell 里 export 过。
if [ "${SLIME_LAUNCH_MODE:-jobsubmit}" = "direct" ]; then
   echo "[launch] 直接跑 train.py(跳过 ray job submit;单节点本地集群不需要 job server)"
   RAY_ADDRESS=auto python3 train.py \
   ${PLACEMENT_ARGS[@]} \
   ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ${ROLLOUT_ARGS[@]} \
   ${PERF_ARGS[@]} ${GRPO_ARGS[@]} ${OPTIMIZER_ARGS[@]} \
   ${APEX_ARGS[@]} ${SGLANG_ARGS[@]} ${CUSTOM_ARGS[@]} ${WANDB_ARGS[@]} ${EXTRA_ARGS:-}
else
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="$RUNTIME_ENV_JSON" \
   -- python3 train.py \
   ${PLACEMENT_ARGS[@]} \
   ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ${ROLLOUT_ARGS[@]} \
   ${PERF_ARGS[@]} ${GRPO_ARGS[@]} ${OPTIMIZER_ARGS[@]} \
   ${APEX_ARGS[@]} ${SGLANG_ARGS[@]} ${CUSTOM_ARGS[@]} ${WANDB_ARGS[@]} ${EXTRA_ARGS:-}
fi
