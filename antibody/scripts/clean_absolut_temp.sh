#!/usr/bin/env bash
# Clean up stale Absolut temp files left by crashed AntBO runs.
#
# Filename format produced by task/tools.py:Absolut.energy:
#   {prefix}TempCDR3_{antigen}.txt
#   {prefix}TempBindingsFor{antigen}_t{i}_Part1_of_1.txt
#   {prefix}{antigen}FinalBindings_Process_{i}_Of_{n}.txt
# where prefix = run{pid}_{epoch_ms}_
#
# This script matches the prefix AND the exact suffix structure, so Absolut's
# own data files ({hash}Structures.txt, {hash}.txt, etc.) are never touched.
#
# Usage:
#   bash scripts/clean_absolut_temp.sh                       # default: >30 min old
#   AGE_MINUTES=60 bash scripts/clean_absolut_temp.sh        # custom threshold
#   AGE_MINUTES=0  bash scripts/clean_absolut_temp.sh        # purge ALL (dangerous:
#                                                            # confirm no active run)
#   ABSOLUT_PATH=/other/path bash scripts/clean_absolut_temp.sh

set -e

AGE_MINUTES="${AGE_MINUTES:-30}"
ABSOLUT_PATH="${ABSOLUT_PATH:-/mnt/data0/shared/AntBO/Absolut}"

if [ ! -d "$ABSOLUT_PATH" ]; then
    echo "ERR: ABSOLUT_PATH does not exist: $ABSOLUT_PATH" >&2
    exit 1
fi

# Activate nullglob so unmatched patterns expand to nothing (not literal text).
shopt -s nullglob

cd "$ABSOLUT_PATH"

now_ms=$(($(date +%s) * 1000))
threshold_ms=$((now_ms - AGE_MINUTES * 60 * 1000))

# Three exact-match globs: prefix + full suffix structure.
# Prefix run[0-9]*_[0-9]*_ locks the run{pid}_{epoch_ms}_ header.
# Suffix locks the per-file-type structure produced by Absolut.energy.
patterns=(
    'run[0-9]*_[0-9]*_TempCDR3_*.txt'                                # input
    'run[0-9]*_[0-9]*_TempBindingsFor*_t[0-9]*_Part1_of_1.txt'       # per-thread temp
    'run[0-9]*_[0-9]*_*FinalBindings_Process_[0-9]*_Of_[0-9]*.txt'   # final output
)

removed=0
skipped_active=0
skipped_nomatch=0

for pat in "${patterns[@]}"; do
    for f in $pat; do
        # Extract epoch_ms from the run{pid}_{epoch_ms}_ prefix.
        ts_ms=$(printf '%s' "$f" | sed -nE 's/^run[0-9]+_([0-9]+)_.*/\1/p')
        if [ -z "$ts_ms" ]; then
            skipped_nomatch=$((skipped_nomatch + 1))
            continue
        fi
        if [ "$ts_ms" -le "$threshold_ms" ]; then
            rm -f "$f"
            removed=$((removed + 1))
        else
            skipped_active=$((skipped_active + 1))
        fi
    done
done

echo "Cleaned $removed stale file(s); skipped $skipped_active active (< ${AGE_MINUTES}min), $skipped_nomatch unparseable in $ABSOLUT_PATH"
