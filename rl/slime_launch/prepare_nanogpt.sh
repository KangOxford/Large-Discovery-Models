#!/usr/bin/env bash
# One-time preparation for nanoGPT RL: dataset -> reference -> episode data.
#
# Run this on a node that has BOTH a GPU and network access. Step 3 measures
# the reward's reference point by really training the default config, and
# step 1 downloads shards and trains a tokenizer.
#
# Everything is idempotent: the dataset check is a no-op once prepared, and the
# reference measurement is served from the evaluation cache on re-runs.
set -euo pipefail

REPO_ROOT=${REPO_ROOT:?set REPO_ROOT to the LDM repo root}
CONFIG=${CONFIG:-$REPO_ROOT/rl/slime_launch/config_nanogpt.json}
WORK_DIR=${WORK_DIR:?set WORK_DIR, a writable dir for runs/cache/episodes}
NUM_SHARDS=${NUM_SHARDS:-16}
EPISODES_OUT=${EPISODES_OUT:-$WORK_DIR/rl_episodes_nanogpt.jsonl}
REFERENCE_OUT=${REFERENCE_OUT:-$WORK_DIR/nanogpt_reference.json}

export AUTORESEARCH_CACHE_DIR=${AUTORESEARCH_CACHE_DIR:-$WORK_DIR/autoresearch_cache}
export PYTHONPATH=$REPO_ROOT/rl:$REPO_ROOT:${PYTHONPATH:-}

mkdir -p "$WORK_DIR" "$AUTORESEARCH_CACHE_DIR"

cfg() { python3 -c "
import json,sys
node=json.load(open('$CONFIG'))
for key in sys.argv[1].split('.'): node=node[key]
print(node)" "$1"; }

BUDGETS=$(cfg evaluation.budgets)
COUNT=$(cfg episodes.count)
ITERATIONS=$(cfg episodes.iterations)
RESERVOIR=$(cfg episodes.reservoir_size)
EVALS_PER_ROUND=$(cfg episodes.evaluations_per_round)
FREE_MODE=$(cfg episodes.free_knob_mode)
TIMEOUT_SLACK=$(cfg evaluation.timeout_slack)

EVAL_GPUS=${NANOGPT_EVAL_GPUS:-$(cfg evaluation.eval_gpus)}
if [[ -z "$EVAL_GPUS" ]]; then
  echo "ERROR: set NANOGPT_EVAL_GPUS (or evaluation.eval_gpus in the config) to" >&2
  echo "       the physical GPU indices evaluation may use." >&2
  exit 1
fi

echo "=============================================================="
echo " repo         : $REPO_ROOT"
echo " work dir     : $WORK_DIR"
echo " data cache   : $AUTORESEARCH_CACHE_DIR"
echo " eval GPUs    : $EVAL_GPUS"
echo " budgets      : $BUDGETS"
echo "=============================================================="

# --- 1. dataset + tokenizer ------------------------------------------------
# prepare.py resolves its CACHE_DIR from AUTORESEARCH_CACHE_DIR. 16 shards plus
# the 8192-vocab BPE is ~1.5 GB. Use few download workers: with many, prepare.py
# breaks its own stdout pipe when it is not attached to a terminal.
echo "[1/4] dataset + tokenizer (${NUM_SHARDS} shards)"
python3 "$REPO_ROOT/tasks/nanogpt/scripts/prepare.py" \
  --num-shards "$NUM_SHARDS" --download-workers 2

# --- 2. FlashAttention-3 ---------------------------------------------------
# train.py fetches the Hopper FA3 kernel from the HF hub on first use. Do it
# here, while there is network: a compute node without egress cannot.
echo "[2/4] prefetch FlashAttention-3 kernel"
python3 - <<'PY' || echo "  (skipped: prefetch failed; train.py will retry at run time)"
try:
    from kernels import get_kernel
    get_kernel("varunneal/flash-attention-3")
    print("  FA3 cached")
except Exception as exc:
    print(f"  FA3 prefetch unavailable: {type(exc).__name__}: {exc}")
    raise SystemExit(1)
PY

# --- 3. reference measurement ---------------------------------------------
# The improvement reward pays for getting below a FIXED bar. Measure it now:
# left unset, the bar becomes each episode's own first proposal, which pays the
# policy for opening badly.
echo "[3/4] measure the reward reference (one real run per budget)"
python3 -m tasks.nanogpt.scripts.rl_reference \
  --budgets "$BUDGETS" \
  --output "$REFERENCE_OUT" \
  --output-dir "$WORK_DIR" \
  --eval-gpus "$EVAL_GPUS" \
  --repo-root "$REPO_ROOT" \
  --autoresearch-cache-dir "$AUTORESEARCH_CACHE_DIR" \
  --timeout-slack "$TIMEOUT_SLACK"

# --- 4. episode prompt data ------------------------------------------------
# task_python must be set: bridge.generate builds a RemoteLDMEnv for real-mode
# episodes and falls back to a small-molecule venv path that does not exist
# here. nanoGPT's environment only needs numpy + ldm_tts, so the interpreter
# running slime is the right one -- the heavy torch stack lives in the trainer
# subprocess the evaluator spawns, not in the env worker.
echo "[4/4] episode prompt data"
REAL_KWARGS=$(python3 -c "
import json, sys
print(json.dumps({
    'task_python': sys.executable,
    'repo_root': '$REPO_ROOT',
    'output_dir': '$WORK_DIR',
    'eval_gpus': '$EVAL_GPUS',
    'autoresearch_cache_dir': '$AUTORESEARCH_CACHE_DIR',
    'timeout_slack': $TIMEOUT_SLACK,
}))")

python3 -m tasks.nanogpt.scripts.gen_rl_episodes \
  --output "$EPISODES_OUT" \
  --count "$COUNT" \
  --iterations "$ITERATIONS" \
  --reservoir-size "$RESERVOIR" \
  --evaluations-per-round "$EVALS_PER_ROUND" \
  --budgets "$BUDGETS" \
  --references "$REFERENCE_OUT" \
  --free-knob-mode "$FREE_MODE" \
  --seed-offset "${SEED_OFFSET:-0}" \
  --real-kwargs "$REAL_KWARGS"

echo
echo "ready:"
echo "  reference : $REFERENCE_OUT"
echo "  episodes  : $EPISODES_OUT"
echo "  cache     : $WORK_DIR/eval_cache.jsonl"
echo
echo "next: EPISODES=$EPISODES_OUT bash $REPO_ROOT/rl/slime_launch/run_train_real_nanogpt.sh"
