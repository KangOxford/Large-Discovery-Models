#!/usr/bin/env bash
set -e

cd /mnt/data0/shared/AntBO/HEBO/AntBO

# ================================================================
# TEMP rerun script: fills in incomplete / missing runs from
# scripts/run_full_llm_5seeds.sh in outputs/ldm_ninit20_iter200/.
#
# Verified gaps (2026-06-30):
#   seed_44 / 1OB1_C   — rows 193–199 empty (last good = 192)
#   seed_46 / 1H0D_C   — rows 193–199 empty (last good = 192)
#   seed_46 / 1NSN_S   — results.csv missing
#   seed_46 / 1OB1_C   — results.csv missing
#
# Parameters (same as run_full_llm_5seeds.sh, USE_LLM=true default):
#   --config bo/config.yaml
#   --n_trials 1
#   --save-path outputs/ldm_ninit20_iter200/seed_{44,46}
#
# NOTE: resume=false in config, so partial results.csv/optim.pkl
# for seed_44/1OB1_C and seed_46/1H0D_C will be overwritten cleanly.
# ================================================================

CONFIG_FILE="bo/config.yaml"
OUT_ROOT="outputs/ldm_ninit20_iter200"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"

ANTIGEN_FILE_44="/tmp/antbo_rerun_44_${RUN_TAG}.txt"
ANTIGEN_FILE_46="/tmp/antbo_rerun_46_${RUN_TAG}.txt"

echo "============================================================"
echo "Rerun missing/incomplete runs in $OUT_ROOT"
echo "Run tag: $RUN_TAG"
echo "Config:  $CONFIG_FILE"
echo "============================================================"

# --- seed_44: only 1OB1_C needs a fresh run ---
printf "1OB1_C\n" > "$ANTIGEN_FILE_44"

echo "============================================================"
echo "Rerunning seed_44 — antigens: 1OB1_C"
echo "  antigen file: $ANTIGEN_FILE_44"
echo "============================================================"
python bo/main.py \
  --antigens_file "$ANTIGEN_FILE_44" \
  --save-path "$OUT_ROOT/seed_44" \
  --seed 44 \
  --n_trials 1 \
  --config "$CONFIG_FILE"

# --- seed_46: 1H0D_C, 1NSN_S, 1OB1_C ---
printf "1H0D_C\n1NSN_S\n1OB1_C\n" > "$ANTIGEN_FILE_46"

echo "============================================================"
echo "Rerunning seed_46 — antigens: 1H0D_C, 1NSN_S, 1OB1_C"
echo "  antigen file: $ANTIGEN_FILE_46"
echo "============================================================"
python bo/main.py \
  --antigens_file "$ANTIGEN_FILE_46" \
  --save-path "$OUT_ROOT/seed_46" \
  --seed 46 \
  --n_trials 1 \
  --config "$CONFIG_FILE"

# --- Clean up temp antigen files ---
rm -f "$ANTIGEN_FILE_44" "$ANTIGEN_FILE_46"

echo "============================================================"
echo "Done."
echo ""
echo "Verification — row counts for the rerun targets:"
for f in \
  "$OUT_ROOT/seed_44/BO_transformed_overlap/antigen_1OB1_C_kernel_transformed_overlap_search-strat_local_seed_44_cdr_constraint_True_seqlen_11/results.csv" \
  "$OUT_ROOT/seed_46/BO_transformed_overlap/antigen_1H0D_C_kernel_transformed_overlap_search-strat_local_seed_46_cdr_constraint_True_seqlen_11/results.csv" \
  "$OUT_ROOT/seed_46/BO_transformed_overlap/antigen_1NSN_S_kernel_transformed_overlap_search-strat_local_seed_46_cdr_constraint_True_seqlen_11/results.csv" \
  "$OUT_ROOT/seed_46/BO_transformed_overlap/antigen_1OB1_C_kernel_transformed_overlap_search-strat_local_seed_46_cdr_constraint_True_seqlen_11/results.csv" \
  ; do
  if [ -f "$f" ]; then
    rows=$(wc -l < "$f")
    echo "  $f  →  $rows lines ($((rows - 1)) iterations)"
  else
    echo "  MISSING: $f"
  fi
done
echo "============================================================"
