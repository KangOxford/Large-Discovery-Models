#!/usr/bin/env bash
set -e

cd /mnt/data0/shared/AntBO/HEBO/AntBO

# ================================================================
# ONE-TOGGLE: USE_LLM=true  → LDM orchestrator + DSL + LLM calls
#             USE_LLM=false → original AntBO (no LLM, no DSL)
# ================================================================
USE_LLM="${USE_LLM:-true}"
ANTIGEN_FILE="${ANTIGEN_FILE:-test_5_antigens.txt}"
SEEDS=(45 46)
RUN_TAG="$(date +%Y%m%d_%H%M%S)"

# Output root is an internal variable, configurable via env var.
if [ "$USE_LLM" = "true" ]; then
  OUT_ROOT="${OUT_ROOT:-outputs/ldm_ninit20_iter200}"
  RUN_LABEL="LLM AntBO"
else
  OUT_ROOT="${OUT_ROOT:-outputs/antbo_ninit20_iter200}"
  RUN_LABEL="Baseline AntBO (no LLM)"
fi

# When USE_LLM=false, sed-flip both flags in a temp copy of config. The
# original config is never modified.
BASE_CONFIG="bo/config.yaml"
CONFIG_FILE="$BASE_CONFIG"
if [ "$USE_LLM" = "false" ]; then
  CONFIG_FILE="/tmp/antbo_no_llm_${RUN_TAG}.yaml"
  sed -e 's/llm_init_enabled: true/llm_init_enabled: false/' \
      -e 's/llm_loop_enabled: true/llm_loop_enabled: false/' \
      "$BASE_CONFIG" > "$CONFIG_FILE"
  echo "Created temp config (LLM disabled): $CONFIG_FILE"
fi

mkdir -p "$OUT_ROOT"
mkdir -p outputs/backup

echo "============================================================"
echo "Running ${RUN_LABEL}"
echo "Antigen file: $ANTIGEN_FILE"
echo "Seeds: ${SEEDS[@]}"
echo "Output root: $OUT_ROOT"
echo "Config: $CONFIG_FILE"
echo "Run tag: $RUN_TAG"
echo "============================================================"

echo "Current key config:"
grep -n "max_iters\|n_init\|llm_init_enabled\|llm_loop_enabled\|acq_n_candidates\|init_pool_size\|llm_antigen_context\|tabular_search_csv" "$CONFIG_FILE" || true

for SEED in "${SEEDS[@]}"; do
  echo "============================================================"
  echo "Running seed = $SEED"
  echo "============================================================"

  SEED_DIR="$OUT_ROOT/seed_${SEED}"

  # Protect previous run of the same seed, if any.
  if [ -d "$SEED_DIR" ]; then
    OLD_DEST="${SEED_DIR}_old_${RUN_TAG}"
    echo "Existing $SEED_DIR found."
    echo "Moving old seed result to: $OLD_DEST"
    mv "$SEED_DIR" "$OLD_DEST"
  fi
  mkdir -p "$SEED_DIR"

  python bo/main.py \
    --antigens_file "$ANTIGEN_FILE" \
    --save-path "$SEED_DIR" \
    --seed "$SEED" \
    --n_trials 1 \
    --config "$CONFIG_FILE"

  echo "Results for seed $SEED:"
  find "$SEED_DIR" -name "results.csv" | sort
done

# Clean up temp config
if [ "$USE_LLM" = "false" ] && [ -f "$CONFIG_FILE" ]; then
  rm -f "$CONFIG_FILE"
fi

echo "============================================================"
echo "Done."
echo "Mode: ${RUN_LABEL}"
echo "New results are saved under:"
echo "$OUT_ROOT"
echo ""
echo "Expected results.csv count: 25"
echo "Actual results.csv count:"
find "$OUT_ROOT" -name "results.csv" | wc -l
echo ""
echo "Per-(antigen, seed) LLM decision logs (only when LLM enabled):"
find "$OUT_ROOT" -name "llm_decisions.json" | wc -l
echo ""
echo "Result files:"
find "$OUT_ROOT" -name "results.csv" | sort
echo "============================================================"
