#!/usr/bin/env bash
# Phase item A5: does the SkyRL loop close at all on one GH200?
#
# Deliberately contains NO LDM code. The generator returns a constant reward, so
# a failure here is a failure of the stack -- Ray, vLLM, Megatron, the NCCL
# weight sync -- and cannot be confused with a defect in the LDM wiring that
# phase B builds. Running the two together is how a week gets spent on the wrong
# half.
#
# Pass criterion (written before the run): two training steps complete and a
# checkpoint is written.
#
# Usage, on a node with at least one free GPU:
#   bash rl/ldm_skyrl/scripts/run_1gpu_smoke.sh 2>&1 | tee A5_smoke.log
set -euo pipefail

# --- the pins this cluster forces -------------------------------------------
# Driver 565 means CUDA 12.7. SkyRL main wants CUDA 13.0 and driver r580, so the
# pin is the 0.3.0 release commit, whose wheels are cu128/cu129. A cu130 build
# installs cleanly and then reports torch.cuda.is_available() == False.
SKYRL_COMMIT="${SKYRL_COMMIT:-b8a5caaa}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-0.8B}"

# TileLang JIT-compiles the gated-delta-net kernels and needs a full CUDA
# toolkit; without it the GDN backward dies on a missing cuda/atomic header.
if [ -z "${CUDA_HOME:-}" ]; then
  echo "FATAL: CUDA_HOME is unset. The GDN backward needs the full toolkit," >&2
  echo "       and the failure it produces otherwise looks like a model bug." >&2
  exit 2
fi
export NVTE_FUSED_ATTN=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
# NCCL_P2P_DISABLE=1 is never correct here: it forces NVLink traffic through
# shared memory and multi-node training becomes about 25x slower.

# --- node-local scratch, never Lustre ---------------------------------------
# Checkpoints and logs go to node-local storage during the run and are copied
# back afterwards. Writing them straight to Lustre is one of the documented
# metadata anti-patterns on this machine.
RUN_NAME="${RUN_NAME:-ldm_skyrl_a5_smoke}"
SCRATCH="${SCRATCH:-/local/user/$(id -u)/${RUN_NAME}}"
mkdir -p "$SCRATCH/ckpt" "$SCRATCH/export"

echo "=== A5 smoke ==="
echo "skyrl pin      : ${SKYRL_COMMIT} (release 0.3.0)"
echo "model          : ${MODEL_NAME}"
echo "scratch        : ${SCRATCH}"
nvidia-smi --query-gpu=index,name,memory.used,driver_version --format=csv || true

# A6 rides along here: with no flashinfer wheel for aarch64, vLLM picks some
# other attention backend. Which one is a fact worth having in the log rather
# than a thing to rediscover later.
python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available())"

python -m ldm_skyrl.scripts.constant_reward_smoke \
  --model "$MODEL_NAME" \
  --ckpt-path "$SCRATCH/ckpt" \
  --export-path "$SCRATCH/export" \
  --max-training-steps 2 \
  "$@"

echo "=== checkpoint written? ==="
ls -1 "$SCRATCH/ckpt" | head
