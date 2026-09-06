#!/bin/bash
# DENSE control run: Qwen2.5-7B-Instruct (standard transformer), same RL pipeline.
#
# Purpose: isolate the 9B NaN-gradient blocker. The Qwen3.5-9B hybrid (Gated
# DeltaNet + full-attention + MTP) has never logged a finite optimizer step (nan
# originating in a full-attention backward), while dense 1.5B trains cleanly. This
# runs a dense 7B — same loop, same reward, same data — to answer one question:
#   does GRPO backward produce FINITE grad_norm on a dense 7-9B-scale model?
# - finite here  -> the blocker is the hybrid architecture's backward, not the
#                   pipeline / reward / data.
# - still nan    -> the problem is general (config / optimizer / env), not hybrid.
#
# This is a control, NOT the paper model. Watch train/grad_norm only.
#
# ── set for your cluster ──────────────────────────────────────────────────────
export REPO_ROOT=${REPO_ROOT:?set REPO_ROOT, e.g. /path/to/LDM}
export SLIME_ROOT=${SLIME_ROOT:-$REPO_ROOT/rl/slime}
export MEGATRON_ROOT=${MEGATRON_ROOT:?set MEGATRON_ROOT}
export CONDA_PREFIX=${CONDA_PREFIX:?set CONDA_PREFIX (torch/TE/slime/sglang env)}
MODEL_HF=${MODEL_HF:?set MODEL_HF (Qwen2.5-7B-Instruct dir)}
MODEL_REF=${MODEL_REF:?set MODEL_REF (its Megatron torch_dist dir; convert with convert_dense7b.sh)}
EPISODES=${EPISODES:-$REPO_ROOT/rl_episodes_sm_R2.jsonl}
SAVE=${SAVE:-$REPO_ROOT/rl/qwen2.5-7B_rl_densecontrol}
WANDB_PROJECT=${WANDB_PROJECT:-ldm-sm-rl}
WANDB_RUN=${WANDB_RUN:-$(basename "$SAVE")}
# ──────────────────────────────────────────────────────────────────────────────
set -eux
CONFIG=${CONFIG:-$REPO_ROOT/rl/slime_launch/config_real.json}

mkdir -p /root/cudart_block 2>/dev/null || true
touch /root/cudart_block/libcudart.so.13 2>/dev/null || true
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=/root/cudart_block:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$MEGATRON_ROOT:$REPO_ROOT/rl:$REPO_ROOT:$PYTHONPATH
export CUDA_HOME=$CONDA_PREFIX
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export CUDA_DEVICE_MAX_CONNECTIONS=1

jq_get() { python3 -c "import json;print(json.load(open('$CONFIG'))['training']['$1'])"; }
NUM_ROLLOUT=$(jq_get num_rollout)
ROLLOUT_BATCH=$(jq_get rollout_batch_size)
N_SAMPLES=${N_SAMPLES:-$(jq_get n_samples_per_prompt)}
UPDATES_PER_ROLLOUT=${UPDATES_PER_ROLLOUT:-1}
GLOBAL_BATCH=$(( ROLLOUT_BATCH * N_SAMPLES / UPDATES_PER_ROLLOUT ))
RESP_LEN=$(jq_get rollout_max_response_len)
MAX_TOKENS=$(jq_get max_tokens_per_gpu)
TEMPERATURE=$(jq_get rollout_temperature)
LR=$(jq_get lr)
SAVE_INTERVAL=$(jq_get save_interval)

cd "$SLIME_ROOT"

# Dense Qwen2.5-7B architecture (standard transformer; no hybrid spec, no MTP).
source "$SLIME_ROOT/scripts/models/qwen2.5-7B.sh"   # sets MODEL_ARGS=(...)

CKPT_ARGS=(--hf-checkpoint "$MODEL_HF" --ref-load "$MODEL_REF" --save "$SAVE" --save-interval "$SAVE_INTERVAL")
ROLLOUT_ARGS=(
   --prompt-data "$EPISODES" --input-key prompt --label-key label
   --num-rollout "$NUM_ROLLOUT" --rollout-batch-size "$ROLLOUT_BATCH" --n-samples-per-prompt "$N_SAMPLES"
   --rollout-max-response-len "$RESP_LEN" --rollout-temperature "$TEMPERATURE"
   --global-batch-size "$GLOBAL_BATCH" --balance-data
)
# Single node, 4 GPUs: TP=2 actor + 2 sglang. Plain fp32 optimizer state (no
# precision-aware / bf16 momenta) so this control isolates the ARCHITECTURE, not
# the optimizer-dtype choice. 7B fp32 Adam state ~= 57.7 GB/rank on TP=2, fits 96GB.
PERF_ARGS=(
   --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 --context-parallel-size 1
   --use-distributed-optimizer
   --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
   --use-dynamic-batch-size --max-tokens-per-gpu "$MAX_TOKENS"
)
APEX_ARGS=(--no-gradient-accumulation-fusion)
GRPO_ARGS=(--advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl --eps-clip 0.2 --eps-clip-high 0.28)
OPTIMIZER_ARGS=(--optimizer adam --lr "$LR" --lr-decay-style constant --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.98 --clip-grad 1.0)
SGLANG_ARGS=(--rollout-num-gpus 2 --sglang-mem-fraction-static 0.7)
CUSTOM_ARGS=(--custom-generate-function-path ldm_rl.bridge.generate --custom-rm-path ldm_rl.bridge.reward_func)
WANDB_ARGS=()
[[ -n "${WANDB_KEY:-}" ]] && WANDB_ARGS=(--use-wandb --wandb-project "$WANDB_PROJECT" --wandb-key "$WANDB_KEY" --wandb-run-name "$WANDB_RUN")

echo "resolved: n_samples=$N_SAMPLES global_batch=$GLOBAL_BATCH (dense Qwen2.5-7B control, TP=2, fp32 optim state)"

ray stop --force 2>/dev/null || true
sleep 3
ray start --head --node-ip-address 127.0.0.1 --num-gpus 4 --disable-usage-stats
RUNTIME_ENV_JSON="{\"env_vars\": {\"PYTHONPATH\": \"$MEGATRON_ROOT:$REPO_ROOT/rl:$REPO_ROOT\", \"LD_LIBRARY_PATH\": \"/root/cudart_block:$CONDA_PREFIX/lib\", \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"}}"
ray job submit --address="http://127.0.0.1:8265" --runtime-env-json="$RUNTIME_ENV_JSON" \
   -- python3 train.py \
   --actor-num-nodes 1 --actor-num-gpus-per-node 2 --rollout-num-gpus 2 \
   ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ${ROLLOUT_ARGS[@]} ${PERF_ARGS[@]} ${GRPO_ARGS[@]} ${OPTIMIZER_ARGS[@]} ${APEX_ARGS[@]} ${SGLANG_ARGS[@]} ${CUSTOM_ARGS[@]} ${WANDB_ARGS[@]}
