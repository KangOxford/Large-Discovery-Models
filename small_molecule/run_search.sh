#!/usr/bin/env bash
# Run search trajectories across methods and seeds. Each (method, seed)
# writes one JSON under output/bo/. Edit variables at the top + inline
# args on each python invocation.
#
# Default: multi-objective vina+nn (n_obj=2, EHVI acquisition for the
# BO methods; Chebyshev-ParEGO expansion for the random-best method).
# Per-backend minimise direction is hard-coded (vina min, nn max); the
# JSON config.echo carries the resulting tuple. To swap objectives,
# change OBJECTIVE below (e.g. OBJECTIVE="vina", OBJECTIVE="nn",
# OBJECTIVE="vina+nn+mock", ...).
#
# After running, aggregate + plot via:
#     python plot_search_results.py --input-dir output/bo_vina_nn
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
    # Sanity-check that the activated Python has the LLM advisor's
    # dependencies. If the venv is stale (e.g. missing dotenv), fall
    # back to the system Python so the script still runs.
    if ! python -c "import dotenv" 2>/dev/null; then
        echo "WARN: .venv lacks python-dotenv; deactivating and using system Python." >&2
        # shellcheck disable=SC1091
        source "$(which deactivate 2>/dev/null || echo /dev/null)" 2>/dev/null || true
    fi
fi

# ============================================================================
# Shared cross-algorithm variables (used by all methods).
# ============================================================================

# SEED_SMILES="./output/smiles.txt"
SEED_SMILES="CCO,CCN,CCC"
NUM_EVALUATIONS=80
BATCH_SIZE=5
INIT_SIZE=10

# Acquisition (used by the single-objective BO methods;
# Multi-objective uses EHVI / Chebyshev ParEGO regardless of this field).
ACQUISITION="ei"            # ei | ucb | pi
XI=0.01
KAPPA=2.0

# Multi-objective (used by all methods when OBJECTIVE is '+'-joined).
OBJECTIVE="vina+nn"           # 'vina', 'nn', 'mock', or '+'-joined
                              #  e.g. 'vina+nn', 'vina+nn+mock'.
                              # Default is 'mock' so the script runs
                              # end-to-end with zero external deps
                              # (no Vina binary, no lightgbm model).
                              # For real docking runs, set
                              # OBJECTIVE="vina+nn" and ensure the
                              # Vina binary + activity_modeling model
                              # are present.
REF_POINT=""                  # "" → use DEFAULT_REF (per-objective defaults);
                              # "X,Y,..." → override the registry.
EHVI_N_SAMPLES=128            # Monte-Carlo samples per candidate in 2-obj EHVI.
CHE_ALPHA=1.0                 # Beta(α,1) concentration for Chebyshev-ParEGO
                              # simplex sampling (n_obj >= 3 only; ignored
                              # for n_obj < 3). α=1 → uniform on simplex.

# GP (shared across all four methods)
# Note: --gp-device and --reasyn-devices are derived from a single
# DEVICE variable below (see device_flags() helper).
GP_FIT_ITERSTEPS=100
GP_LEARNING_RATE=0.05
GP_MIN_JITTER=1e-6
GP_MAX_JITTER=1e-1
GP_STANDARDIZE_Y=1          # 1 = --gp-standardize-y; 0 = --no-gp-standardize-y
GP_FP_RADIUS=2
GP_FP_N_BITS=2048

# SMILES length cap (shared across all four methods). Drives:
#   - the search-loop pool filter (--smiles-max-len, default 50)
#   - the GP string kernel's int64 tensor padding (GPConfig.smiles_maxlen)
# Set to "" to disable (--smiles-max-len None).
SMILES_MAX_LEN=100

# Device (single source-of-truth for both the GP and ReaSyn).
# Set to a GPU index (e.g. "0", "1", "7"), a comma-separated list of
# indices for multi-GPU ReaSyn (e.g. "0,1"), or "cpu" for CPU-only runs.
# The device_flags() helper below derives --gp-device (formatted as
# "cuda:<index>" for non-cpu) and --reasyn-devices (the index list) from
# this single variable. For CPU, --reasyn-devices is omitted (ReaSyn
# does not support CPU).
DEVICE="1"

# Vina (used by vina + nn paths; passed when --objective is not mock)
VINA_BIN="../bin/vina"
VINA_CACHE_DIR="output/bo/vina_cache"
VINA_PDB_ID="8UN5"
VINA_CHAIN_ID="A"
VINA_LIGAND_RESNAME=""        # leave empty for default
VINA_EXHAUSTIVENESS=4
VINA_N_POSES=3
VINA_SEED=42
VINA_MAX_WORKERS=4
VINA_ALLOW_DEBUG_RECEPTOR=0   # 1 = --vina-allow-debug-receptor
VINA_NO_CACHE=0               # 1 = --vina-no-cache

# ReaSyn (used by vina + nn paths; passed when --objective is not mock)
REASYN_REPO="../ReaSyn"
REASYN_MODEL_PATH="data/trained_model/nv-reasyn-ar-166m-v2.ckpt,data/trained_model/nv-reasyn-eb-174m-v2.ckpt"
# REASYN_DEVICES is derived from DEVICE below (see device_flags() helper).
REASYN_PYTHON_BIN="../ReaSyn/.venv/bin/python"
REASYN_SEARCH_WIDTH=5
REASYN_EXHAUSTIVENESS=8
REASYN_NUM_CYCLES=3
REASYN_NUM_EDITFLOW_SAMPLES=10
REASYN_NUM_EDITFLOW_STEPS=30
REASYN_TIME_LIMIT=20
REASYN_NUM_WORKERS_PER_GPU=1
REASYN_FILTER_SIM=0.8
REASYN_NO_CANONICALIZE=0       # 1 = --reasyn-no-canonicalize

# Output + logging
OUTPUT_DIR="output/bo_vina_nn"
LOG_LEVEL="WARNING"          # DEBUG | INFO | WARNING | ERROR
VERBOSE=1                    # 1 = --verbose

# Variance / seeds
BASE_SEED=0
N_SEEDS=5

# ============================================================================
# Method-specific knobs (inlined near each python invocation).
# ============================================================================

# Random methods
RANDOM_POOL_MIN_SIZE=9       # set to "" to disable refill; Python default is 1
RANDOM_POOL_MAX_SIZE=18       # set to "" for unbounded; Python default is None

# BO methods (acq_budget is per-implementation because GP inference cost differs:
# Tanimoto is cheap on GPU so we can afford a large pool, while the string
# kernel is much more expensive and benefits from a tighter budget).
BO_ACQ_BUDGET_TAN=1024        # bo-tanimoto: pool subsample for GP+acquisition; "" = no subsampling
BO_ACQ_BUDGET_STR=128         # bo-strkernel: pool subsample for GP+acquisition; "" = no subsampling
BO_MAX_POOL_SIZE=1024           # BO pool FIFO cap; empty = unbounded (Python None)

# ============================================================================
# LLM advisor (bo-*-ldm methods).
# Credentials are read from .env (LLM_API_KEY, LLM_BASE_URL) by the
# Python process at startup. .env is the single source of truth.
# There is no LLM_MODEL env var; the model is hardcoded to
# DeepSeek-V4-Flash in the Python code.
# ============================================================================
LLM_MODEL="DeepSeek-V4-Flash"            # local var; default only
LLM_POOL_MIN_SIZE=10                    # "" = auto-set to --batch-size for LDM methods
LLM_TRAJECTORY_DIR=""                   # path; "" = no sidecar (trajectory still embedded in main JSON)

# LDM_SYS_PROMPT is a free-form supplement appended to all three LLM
# system prompts (Stage A1 actions, A2 review-analogs, B review-
# suggestions). The Python side treats this as a path-or-text: if the
# value is a path to an existing file, the file's contents are read
# and used; otherwise the value is used as inline text.
#
# Default: pass the bundled ``ldm_system_prompt.txt`` (in the repo
# root). To override, set this to a different path or to an inline
# string. Set to "" to disable the supplement entirely.
LLM_SYS_PROMPT="ldm_system_prompt.txt"

# ============================================================================
# Experiment selection — edit THIS list to choose which methods to run.
# To run only LDM methods, change to:
#     METHODS=("bo-tanimoto-ldm" "bo-strkernel-ldm")
# ============================================================================
# METHODS=("bo-tanimoto" "bo-strkernel" "bo-tanimoto-ldm" "bo-strkernel-ldm" "random" "random-best")
METHODS=("bo-tanimoto-ldm" "bo-strkernel-ldm")

# ============================================================================
# Run
# ============================================================================

mkdir -p "$OUTPUT_DIR"

# Per-method GPU assignment is just a marker here; the python child inherits
# CUDA_VISIBLE_DEVICES from the environment. Uncomment / edit as needed:
# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# Helper: emit verbose / gp-standardize flags based on shared variables.
gp_standardize_flag() { [[ "$GP_STANDARDIZE_Y" == "1" ]] && printf -- "--gp-standardize-y\n" || printf -- "--no-gp-standardize-y\n"; return 0; }
verbose_flag() { [[ "$VERBOSE" == "1" ]] && printf -- "--verbose\n" || true; return 0; }

# Helper: emit --gp-device and --reasyn-devices derived from DEVICE.
# For non-cpu, formats --gp-device as "cuda:<index>" and emits
# --reasyn-devices as the raw index list. For cpu, emits --gp-device cpu
# and skips --reasyn-devices (ReaSyn has no CPU backend).
device_flags() {
    if [[ "$DEVICE" == "cpu" || -z "$DEVICE" ]]; then
        echo "--gp-device cpu"
    else
        echo "--gp-device cuda:${DEVICE}"
        echo "--reasyn-devices ${DEVICE}"
    fi
}

# Helper: emit multi-objective flags (always; the CLI silently ignores
# --ref-point for n_obj==1). --ehvi-n-samples / --che-alpha are
# harmless for n_obj==1.
mo_flags() {
    echo "--objective $OBJECTIVE"
    [[ -n "$REF_POINT" ]] && echo "--ref-point $REF_POINT"
    echo "--ehvi-n-samples $EHVI_N_SAMPLES"
    echo "--che-alpha $CHE_ALPHA"
    return 0
}

# Helper: emit vina flags (only when not empty).
vina_flags() {
    echo "--vina-bin $VINA_BIN"
    echo "--vina-cache-dir $VINA_CACHE_DIR"
    echo "--vina-pdb-id $VINA_PDB_ID"
    echo "--vina-chain-id $VINA_CHAIN_ID"
    [[ -n "$VINA_LIGAND_RESNAME" ]] && echo "--vina-ligand-resname $VINA_LIGAND_RESNAME"
    echo "--vina-exhaustiveness $VINA_EXHAUSTIVENESS"
    echo "--vina-n-poses $VINA_N_POSES"
    echo "--vina-seed $VINA_SEED"
    echo "--vina-max-workers $VINA_MAX_WORKERS"
    [[ "$VINA_ALLOW_DEBUG_RECEPTOR" == "1" ]] && echo "--vina-allow-debug-receptor"
    [[ "$VINA_NO_CACHE" == "1" ]] && echo "--vina-no-cache"
    return 0
}

reasyn_flags() {
    # --reasyn-devices is emitted by device_flags() (derived from DEVICE).
    echo "--reasyn-repo $REASYN_REPO"
    echo "--reasyn-model-path $REASYN_MODEL_PATH"
    [[ -n "$REASYN_PYTHON_BIN" ]] && echo "--reasyn-python-bin $REASYN_PYTHON_BIN"
    echo "--reasyn-search-width $REASYN_SEARCH_WIDTH"
    echo "--reasyn-exhaustiveness $REASYN_EXHAUSTIVENESS"
    echo "--reasyn-num-cycles $REASYN_NUM_CYCLES"
    echo "--reasyn-num-editflow-samples $REASYN_NUM_EDITFLOW_SAMPLES"
    echo "--reasyn-num-editflow-steps $REASYN_NUM_EDITFLOW_STEPS"
    echo "--reasyn-time-limit $REASYN_TIME_LIMIT"
    echo "--reasyn-num-workers-per-gpu $REASYN_NUM_WORKERS_PER_GPU"
    echo "--reasyn-filter-sim $REASYN_FILTER_SIM"
    [[ "$REASYN_NO_CANONICALIZE" == "1" ]] && echo "--reasyn-no-canonicalize"
    return 0
}

gp_flags() {
    # --gp-device is emitted by device_flags() (derived from DEVICE).
    echo "--gp-fit-itersteps $GP_FIT_ITERSTEPS"
    echo "--gp-learning-rate $GP_LEARNING_RATE"
    echo "--gp-min-jitter $GP_MIN_JITTER"
    echo "--gp-max-jitter $GP_MAX_JITTER"
    gp_standardize_flag
    echo "--gp-fp-radius $GP_FP_RADIUS"
    echo "--gp-fp-n-bits $GP_FP_N_BITS"
    [[ -n "$SMILES_MAX_LEN" ]] && echo "--smiles-max-len $SMILES_MAX_LEN"
    return 0
}

# LLM advisor flags (only for bo-*-ldm methods).
llm_flags() {
    echo "--llm-model $LLM_MODEL"
    [[ -n "$LLM_POOL_MIN_SIZE" ]] && echo "--pool-min-size $LLM_POOL_MIN_SIZE"
    [[ -n "$LLM_TRAJECTORY_DIR" ]] && echo "--llm-trajectory-dir $LLM_TRAJECTORY_DIR"
    [[ -n "$LLM_SYS_PROMPT" ]] && printf -- "--ldm-sys-prompt %s\n" "$LLM_SYS_PROMPT"
    return 0
}

# ---- MAIN LOOP ----
echo "Writing outputs to $OUTPUT_DIR/ (ALG_seed=SEED.json)"
echo
echo "Per-seed method order: ${METHODS[*]}"
echo

# Per-seed main loop. Each method is dispatched via `case` so the
# user can edit METHODS=() to control which experiments run.
for SEED in $(seq "$BASE_SEED" "$((BASE_SEED + N_SEEDS - 1))"); do
    echo "=== seed=$SEED ==="
    for METHOD in "${METHODS[@]}"; do
        CMD=(python -u run_search.py
             --method "$METHOD"
             --seed "$SEED"
             --seed-smiles "$SEED_SMILES"
             --num-evaluations "$NUM_EVALUATIONS"
             --batch-size "$BATCH_SIZE"
             --output "$OUTPUT_DIR"
             --log-level "$LOG_LEVEL"
             $(verbose_flag)
        )

        case "$METHOD" in
            bo-tanimoto|bo-tanimoto-ldm|bo-strkernel|bo-strkernel-ldm)
                CMD+=(--init-size "$INIT_SIZE"
                      --acquisition "$ACQUISITION"
                      --xi "$XI"
                      --kappa "$KAPPA")
                [[ -n "$BO_MAX_POOL_SIZE" ]] && CMD+=(--max-pool-size "$BO_MAX_POOL_SIZE")
                case "$METHOD" in
                    bo-tanimoto|bo-tanimoto-ldm)
                        [[ -n "$BO_ACQ_BUDGET_TAN" ]] && CMD+=(--acq-budget "$BO_ACQ_BUDGET_TAN") ;;
                    bo-strkernel|bo-strkernel-ldm)
                        [[ -n "$BO_ACQ_BUDGET_STR" ]] && CMD+=(--acq-budget "$BO_ACQ_BUDGET_STR") ;;
                esac
                ;;
            random|random-best) ;;
            *) echo "Unknown method: $METHOD"; exit 1 ;;
        esac

        # Objective, device, GP, vina, reasyn, LLM flags (only the
        # relevant ones are emitted for each method).
        CMD+=($(mo_flags) $(device_flags) $(gp_flags) $(vina_flags) $(reasyn_flags))
        case "$METHOD" in
            *-ldm) CMD+=($(llm_flags)) ;;
        esac

        printf '  [%s] ' "$METHOD"; printf ' %q' "${CMD[@]}"; echo
        # Continue past a single (method, seed) failure so a transient
        # LLM API error or Vina hiccup doesn't kill the whole batch.
        # set +e/-e brackets so the outer `set -e` still applies
        # elsewhere.
        set +e
        "${CMD[@]}"
        rc=$?
        set -e
        if [[ $rc -ne 0 ]]; then
            echo "  [WARN] $METHOD seed=$SEED exited with code $rc; continuing." >&2
        fi
    done
    echo
done

echo "All trajectories done. JSONs in $OUTPUT_DIR/"
echo "Plot with: python plot_search_results.py --input-dir $OUTPUT_DIR"
