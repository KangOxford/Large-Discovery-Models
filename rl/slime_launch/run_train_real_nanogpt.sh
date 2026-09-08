#!/usr/bin/env bash
# Real-mode GRPO on the nanoGPT knob-search task.
#
# The reward is a MEASURED val_bpb: every environment step trains a real
# nanoGPT for its full wall-clock budget on a GPU. That has one structural
# consequence this script exists to handle -- the trainer and the evaluator
# need DISJOINT GPUs. The trainer holds its own for the whole run (Megatron
# actor + sglang rollout), so evaluation cannot borrow them.
#
#   TRAIN_GPUS=0,1,2,3  NANOGPT_EVAL_GPUS=4,5,6,7   # 8-GPU node
#
# Everything else is parameterised; nothing is hardcoded to one cluster.
# Run prepare_nanogpt.sh first (dataset, reward reference, episode data).
set -euo pipefail

# ── required ────────────────────────────────────────────────────────────────
REPO_ROOT=${REPO_ROOT:?set REPO_ROOT to the LDM repo root}
SLIME_ROOT=${SLIME_ROOT:-$REPO_ROOT/rl/slime}
MEGATRON_ROOT=${MEGATRON_ROOT:?set MEGATRON_ROOT}
MODEL_HF=${MODEL_HF:?set MODEL_HF (HF checkpoint dir)}
MODEL_REF=${MODEL_REF:?set MODEL_REF (its Megatron torch_dist dir)}
MODEL_ARGS_SCRIPT=${MODEL_ARGS_SCRIPT:?set MODEL_ARGS_SCRIPT, e.g. \$SLIME_ROOT/scripts/models/qwen3.5-9B.sh}
EPISODES=${EPISODES:?set EPISODES (prompt-data jsonl from prepare_nanogpt.sh)}
WORK_DIR=${WORK_DIR:?set WORK_DIR (must match prepare_nanogpt.sh)}

# ── GPU split ───────────────────────────────────────────────────────────────
TRAIN_GPUS=${TRAIN_GPUS:?set TRAIN_GPUS, e.g. 0,1,2,3}
NANOGPT_EVAL_GPUS=${NANOGPT_EVAL_GPUS:?set NANOGPT_EVAL_GPUS, e.g. 4,5,6,7}
export NANOGPT_EVAL_GPUS

python3 - "$TRAIN_GPUS" "$NANOGPT_EVAL_GPUS" <<'PY'
import sys
train = {x.strip() for x in sys.argv[1].split(",") if x.strip()}
evaluate = {x.strip() for x in sys.argv[2].split(",") if x.strip()}
overlap = sorted(train & evaluate)
if overlap:
    # Sharing a device means a 300s training run competing with the actor's
    # weights and sglang's KV cache; it OOMs one or both.
    sys.exit(
        f"ERROR: TRAIN_GPUS and NANOGPT_EVAL_GPUS overlap on {overlap}. "
        "The evaluator needs whole GPUs the trainer is not holding."
    )
if not evaluate:
    sys.exit("ERROR: NANOGPT_EVAL_GPUS is empty; nothing could be measured.")
print(f"GPU split OK: trainer {sorted(train)} | evaluation {sorted(evaluate)}")
PY

CONFIG=${CONFIG:-$REPO_ROOT/rl/slime_launch/config_nanogpt.json}

for required in "$CONFIG:CONFIG" "$EPISODES:EPISODES" "$MODEL_ARGS_SCRIPT:MODEL_ARGS_SCRIPT" \
                "$SLIME_ROOT/train.py:SLIME_ROOT"; do
  path=${required%:*}; name=${required##*:}
  if [[ ! -e "$path" ]]; then
    echo "ERROR: $name points at a missing path: $path" >&2
    [[ "$name" == "EPISODES" ]] && \
      echo "       run rl/slime_launch/prepare_nanogpt.sh first." >&2
    exit 1
  fi
done

SAVE=${SAVE:-$WORK_DIR/nanogpt_rl_ckpt}
WANDB_PROJECT=${WANDB_PROJECT:-ldm-nanogpt-rl}
WANDB_RUN=${WANDB_RUN:-$(basename "$SAVE")}

export PYTHONUNBUFFERED=1
export PYTHONPATH=$MEGATRON_ROOT:$REPO_ROOT/rl:$REPO_ROOT:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=$TRAIN_GPUS
export CUDA_DEVICE_MAX_CONNECTIONS=1
export AUTORESEARCH_CACHE_DIR=${AUTORESEARCH_CACHE_DIR:-$WORK_DIR/autoresearch_cache}

cfg() { python3 -c "
import json,sys
node=json.load(open('$CONFIG'))
for key in sys.argv[1].split('.'): node=node[key]
print(node)" "$1"; }

NUM_ROLLOUT=$(cfg training.num_rollout)
ROLLOUT_BATCH=$(cfg training.rollout_batch_size)
N_SAMPLES=${N_SAMPLES:-$(cfg training.n_samples_per_prompt)}
UPDATES_PER_ROLLOUT=${UPDATES_PER_ROLLOUT:-$(cfg training.updates_per_rollout)}
RESP_LEN=$(cfg training.rollout_max_response_len)
MAX_TOKENS=$(cfg training.max_tokens_per_gpu)
TEMPERATURE=$(cfg training.rollout_temperature)
LR=$(cfg training.lr)
SAVE_INTERVAL=$(cfg training.save_interval)

# Derive global_batch from n_samples so optimizer-steps-per-rollout stays fixed
# as the group size moves; a hardcoded value silently changes the training
# budget and confounds any n_samples comparison.
if (( (ROLLOUT_BATCH * N_SAMPLES) % UPDATES_PER_ROLLOUT != 0 )); then
  echo "ERROR: rollout_batch($ROLLOUT_BATCH) * n_samples($N_SAMPLES) not divisible by updates_per_rollout($UPDATES_PER_ROLLOUT)" >&2
  exit 1
fi
GLOBAL_BATCH=$(( ROLLOUT_BATCH * N_SAMPLES / UPDATES_PER_ROLLOUT ))

# What this run is going to cost, before it starts costing it.
python3 - "$EPISODES" "$ROLLOUT_BATCH" "$N_SAMPLES" "$NUM_ROLLOUT" "$NANOGPT_EVAL_GPUS" <<'PY'
import json, sys
episodes_file, batch, samples, rollouts, gpus = sys.argv[1:6]
batch, samples, rollouts = int(batch), int(samples), int(rollouts)
n_gpus = len([x for x in gpus.split(",") if x.strip()])
with open(episodes_file) as handle:
    specs = [json.loads(json.loads(line)["prompt"]) for line in handle if line.strip()]
rounds = max(s["iterations"] * s["evaluations_per_round"] for s in specs)
budget = max(float(s["real"].get("time_budget", 300)) for s in specs)
per_step = batch * samples * rounds
seconds = per_step * (budget + 40) / max(n_gpus, 1)
print("--------------------------------------------------------------")
print(f" episodes in file      : {len(specs)}")
print(f" episodes per step     : {batch * samples}")
print(f" real runs per step    : <= {per_step}")
print(f" evaluation GPUs       : {n_gpus}")
print(f" wall clock per step   : ~{seconds/60:.0f} min (before cache hits)")
print(f" total for {rollouts:3d} steps  : ~{seconds*rollouts/3600:.0f} h")
print(" Repeated configs are served from the shared evaluation cache, so the")
print(" real figure falls as the policy converges onto fewer configs.")
print("--------------------------------------------------------------")
PY

cd "$SLIME_ROOT"
# shellcheck source=/dev/null
source "$MODEL_ARGS_SCRIPT"   # sets MODEL_ARGS=(...)

CKPT_ARGS=(
  --hf-checkpoint "$MODEL_HF"
  --ref-load "$MODEL_REF"
  --save "$SAVE"
  --save-interval "$SAVE_INTERVAL"
)

ROLLOUT_ARGS=(
  --prompt-data "$EPISODES"
  --input-key prompt --label-key label
  --num-rollout "$NUM_ROLLOUT"
  --rollout-batch-size "$ROLLOUT_BATCH"
  --n-samples-per-prompt "$N_SAMPLES"
  --rollout-max-response-len "$RESP_LEN"
  --rollout-temperature "$TEMPERATURE"
  --global-batch-size "$GLOBAL_BATCH"
  --balance-data
)

N_TRAIN_GPUS=$(python3 -c "print(len([x for x in '$TRAIN_GPUS'.split(',') if x.strip()]))")
ACTOR_GPUS=${ACTOR_GPUS:-$(( N_TRAIN_GPUS / 2 ))}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-$(( N_TRAIN_GPUS - ACTOR_GPUS ))}
TP_SIZE=${TP_SIZE:-$ACTOR_GPUS}

PERF_ARGS=(
  --tensor-model-parallel-size "$TP_SIZE"
  --pipeline-model-parallel-size 1 --context-parallel-size 1
  --use-distributed-optimizer
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --use-dynamic-batch-size --max-tokens-per-gpu "$MAX_TOKENS"
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl
  --eps-clip 0.2 --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam --lr "$LR" --lr-decay-style constant
  --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.98
)

SGLANG_ARGS=(--rollout-num-gpus "$ROLLOUT_GPUS" --sglang-mem-fraction-static 0.7)

CUSTOM_ARGS=(
  --custom-generate-function-path ldm_rl.bridge.generate
  --custom-rm-path ldm_rl.bridge.reward_func
)

WANDB_ARGS=()
if [[ -n "${WANDB_KEY:-}" ]]; then
  WANDB_ARGS=(--use-wandb --wandb-project "$WANDB_PROJECT"
              --wandb-key "$WANDB_KEY" --wandb-run-name "$WANDB_RUN")
fi

ray stop --force 2>/dev/null || true
sleep 3
ray start --head --node-ip-address 127.0.0.1 --num-gpus "$N_TRAIN_GPUS" --disable-usage-stats

# NANOGPT_EVAL_GPUS must reach the rollout workers: they spawn the environment,
# which claims an evaluation device. (The episode data also carries eval_gpus,
# so this is redundant on purpose -- whichever arrives first works.)
RUNTIME_ENV_JSON=$(python3 - <<PY
import json, os
print(json.dumps({"env_vars": {
    "PYTHONPATH": os.environ["PYTHONPATH"],
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NANOGPT_EVAL_GPUS": os.environ["NANOGPT_EVAL_GPUS"],
    "AUTORESEARCH_CACHE_DIR": os.environ["AUTORESEARCH_CACHE_DIR"],
}}))
PY
)

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="$RUNTIME_ENV_JSON" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "$ACTOR_GPUS" \
  --rollout-num-gpus "$ROLLOUT_GPUS" \
  "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" \
  "${PERF_ARGS[@]}" "${GRPO_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" "${CUSTOM_ARGS[@]}" "${WANDB_ARGS[@]}"
