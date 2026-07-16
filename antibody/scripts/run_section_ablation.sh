#!/usr/bin/env bash
set -e

cd /mnt/data0/shared/AntBO/HEBO/AntBO

ANTIGEN_FILE="test_5_antigens.txt"
OUT_ROOT="outputs/section_ablation"

mkdir -p "$OUT_ROOT"
mkdir -p outputs/backup

run_one () {
  NAME="$1"
  RANKED_INIT="$2"
  TRUST_REGION="$3"
  ANTIGEN_CONTEXT="$4"

  echo "============================================================"
  echo "Running $NAME"
  echo "llm_ranked_init=$RANKED_INIT"
  echo "llm_trust_region=$TRUST_REGION"
  echo "llm_antigen_context=$ANTIGEN_CONTEXT"
  echo "============================================================"

  python scripts/set_llm_sections.py "$RANKED_INIT" "$TRUST_REGION" "$ANTIGEN_CONTEXT"

  if [ -d outputs/llm_run_outputs/BO_transformed_overlap ]; then
    BACKUP_NAME="outputs/backup/pre_${NAME}_$(date +%Y%m%d_%H%M%S)"
    echo "Moving existing outputs/llm_run_outputs/BO_transformed_overlap to $BACKUP_NAME"
    mv outputs/llm_run_outputs/BO_transformed_overlap "$BACKUP_NAME"
  fi

  python bo/main.py \
    --antigens_file "$ANTIGEN_FILE" \
    --seed 42 \
    --n_trials 1 \
    --config bo/config.yaml

  if [ ! -d outputs/llm_run_outputs/BO_transformed_overlap ]; then
    echo "ERROR: outputs/llm_run_outputs/BO_transformed_overlap was not created."
    exit 1
  fi

  DEST="$OUT_ROOT/$NAME"

  if [ -d "$DEST" ]; then
    OLD_DEST="${DEST}_old_$(date +%Y%m%d_%H%M%S)"
    echo "Existing $DEST found. Moving it to $OLD_DEST"
    mv "$DEST" "$OLD_DEST"
  fi

  mv outputs/llm_run_outputs/BO_transformed_overlap "$DEST"

  echo "Saved $NAME results to $DEST"
}

# Section 1 only: LLM-ranked initialization
run_one "section1_ranked_init_only" "true" "false" "false"

# Section 2 only: LLM trust-region policy
run_one "section2_trust_region_only" "false" "true" "false"

# Section 3 effect: antigen context + trust region
run_one "section3_antigen_context_effect" "false" "true" "true"

# Restore full LLM config after all experiments
python scripts/set_llm_sections.py true true true

echo "============================================================"
echo "Done. Results are under:"
echo "$OUT_ROOT"
echo "============================================================"

find "$OUT_ROOT" -name "results.csv" | sort
