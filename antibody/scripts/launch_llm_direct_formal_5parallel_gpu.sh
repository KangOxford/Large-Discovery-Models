#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/data0/shared/AntBO/HEBO/AntBO"
PYTHON="/home/zsgpu/miniconda3/envs/DGM/bin/python"
GPU_ID="${GPU_ID:-0}"
MAX_JOBS="${MAX_JOBS:-5}"
N_EVALS="${N_EVALS:-200}"
N_INIT="${N_INIT:-20}"
GEN_M="${GEN_M:-5}"
SOFTMAX_ETA="${SOFTMAX_ETA:-1.0}"
TEMPERATURE="${TEMPERATURE:-0.7}"
GP_TRAIN_STEPS="${GP_TRAIN_STEPS:-300}"
OUT_BASE="${OUT_BASE:-outputs/llm_direct_formal}"
ANTIGENS_FILE="${ANTIGENS_FILE:-test_5_antigens.txt}"

METHODS=(LLM_gen LDM_gen_softmax LDM_gen_argmax)
SEEDS=(42 43 44 45 46)

cd "$REPO_ROOT"

mkdir -p "$OUT_BASE/logs"
cp "$ANTIGENS_FILE" "$OUT_BASE/antigens.txt"

mapfile -t ANTIGENS < "$ANTIGENS_FILE"

count_rows() {
  local csv="$1"
  if [[ ! -f "$csv" ]]; then
    echo 0
    return
  fi
  local lines
  lines=$(wc -l < "$csv" || echo 0)
  if [[ "$lines" -le 0 ]]; then
    echo 0
  else
    echo $((lines - 1))
  fi
}

run_one() {
  local method="$1"
  local antigen="$2"
  local seed="$3"
  local method_out="$OUT_BASE/$method"
  local mode="${method}_m${GEN_M}_eta${SOFTMAX_ETA}"
  if [[ "$SOFTMAX_ETA" == "1.0" ]]; then
    mode="${method}_m${GEN_M}_eta1"
  fi
  local run_dir="$method_out/${mode}_antigen_${antigen}_seed_${seed}_n${N_EVALS}_batch1"
  local results_csv="$run_dir/results.csv"
  local log_file="$OUT_BASE/logs/${method}_${antigen}_seed${seed}.log"
  local rows
  rows=$(count_rows "$results_csv")
  if [[ "$rows" -ge "$N_EVALS" ]]; then
    echo "[skip complete] method=$method antigen=$antigen seed=$seed rows=$rows"
    return 0
  fi

  echo "[start] method=$method antigen=$antigen seed=$seed gpu=$GPU_ID log=$log_file"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" scripts/run_llm_direct_absolut.py \
    --method "$method" \
    --config bo/config.yaml \
    --antigens_file "$OUT_BASE/one_${antigen}.txt" \
    --seed "$seed" \
    --n_trials 1 \
    --n_evals "$N_EVALS" \
    --batch_size 1 \
    --out_root "$method_out" \
    --n_init "$N_INIT" \
    --gen_m "$GEN_M" \
    --softmax_eta "$SOFTMAX_ETA" \
    --temperature "$TEMPERATURE" \
    --timeout_s 120 \
    --max_retries 3 \
    --history_top_k 20 \
    --gp_train_steps "$GP_TRAIN_STEPS" \
    --acq_device cuda \
    --include_antigen_context \
    --fallback_random \
    > "$log_file" 2>&1
  echo "[done] method=$method antigen=$antigen seed=$seed"
}

active_jobs() {
  jobs -pr | wc -l
}

for antigen in "${ANTIGENS[@]}"; do
  printf '%s\n' "$antigen" > "$OUT_BASE/one_${antigen}.txt"
done

echo "Launching direct LLM formal experiments"
echo "repo=$REPO_ROOT"
echo "methods=${METHODS[*]}"
echo "antigens=${ANTIGENS[*]}"
echo "seeds=${SEEDS[*]}"
echo "max_jobs=$MAX_JOBS gpu=$GPU_ID n_evals=$N_EVALS n_init=$N_INIT gen_m=$GEN_M eta=$SOFTMAX_ETA temperature=$TEMPERATURE"
echo "logs=$OUT_BASE/logs"

for method in "${METHODS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for antigen in "${ANTIGENS[@]}"; do
      while [[ "$(active_jobs)" -ge "$MAX_JOBS" ]]; do
        wait -n || true
      done
      run_one "$method" "$antigen" "$seed" &
    done
    while [[ "$(active_jobs)" -gt 0 ]]; do
      wait -n || true
    done
  done
done

wait || true
echo "All direct LLM formal experiments completed."
