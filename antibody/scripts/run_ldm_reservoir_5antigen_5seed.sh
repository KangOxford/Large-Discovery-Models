#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data0/shared/AntBO/HEBO/AntBO

ENV_PY="${ENV_PY:-/home/zsgpu/miniconda3/envs/DGM/bin/python}"
ANTIGEN_FILE="${ANTIGEN_FILE:-test_5_antigens.txt}"
OUT_ROOT="${OUT_ROOT:-outputs/ldm_reservoir_nchoices_5antigen_5seed_200eval}"
LOG_DIR="${LOG_DIR:-outputs/ldm_reservoir_logs}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${PYTHONPATH:-.}"

mkdir -p "$OUT_ROOT" "$LOG_DIR"
RUN_LOG="$LOG_DIR/ldm_reservoir_nchoices_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "============================================================"
echo "Running LDM reservoir parallel baseline"
echo "Env python: $ENV_PY"
echo "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Antigens: $ANTIGEN_FILE"
echo "Seeds: 42..46"
echo "Output root: $OUT_ROOT"
echo "Expected LLM API requests: 5 antigens * 5 seeds * (200 - 20) = 4500"
echo "Expected LLM generated strategy completions: 5 antigens * 5 seeds * (200 - 20) * 5 = 22500"
echo "Each reservoir step: 1 LLM request with n=5 choices -> 5 strategies -> parallel/batched pools -> 1 Absolut query"
echo "============================================================"

"$ENV_PY" scripts/run_ldm_reservoir_absolut.py \
  --config bo/config.yaml \
  --antigens_file "$ANTIGEN_FILE" \
  --seed 42 \
  --n_trials 5 \
  --n_evals 200 \
  --n_init 20 \
  --init_mode random \
  --out_root "$OUT_ROOT" \
  --n_strategies 5 \
  --parallel_budget 600 \
  --selection softmax \
  --planner_mode n_choices \
  --softmax_eta 1.0 \
  --temperature 0.25 \
  --timeout_s 120 \
  --max_retries 1 \
  --gp_train_steps 300 \
  --sample_timeout_s 5 \
  --bias_weight 0.05 \
  --include_antigen_context

"$ENV_PY" scripts/plot_all_methods_with_ldm_parallel.py \
  --ldm-parallel-root "$OUT_ROOT"

echo "============================================================"
echo "Done. Results:"
find "$OUT_ROOT" -name results.csv | sort
echo "Count: $(find "$OUT_ROOT" -name results.csv | wc -l)"
echo "============================================================"
