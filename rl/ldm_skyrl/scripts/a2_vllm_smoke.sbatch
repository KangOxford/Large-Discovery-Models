#!/bin/bash
#SBATCH --job-name=ldm-skyrl-a2-vllm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --partition=workq
#SBATCH --output=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_skyrl_nb/A2_vllm_%j.log
#
# Phase item A2: install the vLLM 0.23.0+cu129 aarch64 wheel and emit one token
# on a GH200 running driver 565.
#
# Only two wheels are installed, both prebuilt for aarch64. Nothing compiles:
# --no-build turns "a dependency needs a source build" into an immediate, named
# failure instead of a silent multi-hour nvcc run inside a 30-minute job.
#
# torch comes from the cu128 index and vllm from the cu129 index. That pairing
# is deliberate and is what the A1 lock resolved: vllm's PyPI wheel links
# libcudart.so.13, which a 565 driver cannot load, while its cu129 wheel links
# libcudart.so.12, which torch+cu128 supplies.
set -uo pipefail

D=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_skyrl_nb
VENV=/home/u6gb/kangli.u6gb/envs/ldm-skyrl-a2
UV=/projects/public/u6gb/.local/bin/uv

# Caches on node-local disk: the Lustre project inode quota is at 99.2 percent,
# and an install is tens of thousands of small files.
export UV_CACHE_DIR=/local/user/$(id -u)/uv-cache-a2
export UV_PYTHON_INSTALL_DIR=/home/u6gb/kangli.u6gb/.uv-python
export UV_NO_PROGRESS=1
mkdir -p "$UV_CACHE_DIR"

echo "[a2] job=$SLURM_JOB_ID node=$(hostname) utc=$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,memory.used,driver_version --format=csv

echo "[a2] --- creating venv ---"
# --allow-existing, not --clear: a previous attempt leaves a valid, empty py3.12
# venv behind, and reusing it is both idempotent and free. Deleting things I
# created a minute ago is exactly the case where "this is obviously safe to
# remove" turns out not to be. If an env is ever genuinely broken, the escape is
# a new timestamped name, not a delete.
"$UV" venv --python 3.12 --allow-existing "$VENV" || exit 2

echo "[a2] --- installing torch 2.11.0+cu128 (aarch64 wheel) ---"
# --index-strategy unsafe-best-match is required, not optional. uv's default is
# first-match: it stops at the first index that carries a package name at all.
# torch exists on PyPI, so uv never looked at the pytorch index and reported
# "no version of torch==2.11.0+cu128" while the wheel was sitting on the very
# index named one line above. The name being present is not the version being
# present.
VIRTUAL_ENV="$VENV" "$UV" pip install --no-build --index-strategy unsafe-best-match \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple \
    "torch==2.11.0+cu128" || exit 3

echo "[a2] --- installing vllm 0.23.0+cu129 (aarch64 wheel) ---"
VIRTUAL_ENV="$VENV" "$UV" pip install --no-build --index-strategy unsafe-best-match \
    --index-url https://wheels.vllm.ai/0.23.0/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple \
    "vllm==0.23.0+cu129" "torch==2.11.0+cu128" || exit 4

echo "[a2] --- installed versions ---"
"$VENV/bin/python" -c "import importlib.metadata as m; print('torch', m.version('torch')); print('vllm', m.version('vllm'))"
VIRTUAL_ENV="$VENV" "$UV" pip list 2>/dev/null | grep -iE '^(torch|vllm|triton|nvidia|flashinfer|xformers|transformers)' | head -30

echo "[a2] --- probe ---"
export A2_MODEL=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/models/Qwen2.5-1.5B-Instruct
export VLLM_LOGGING_LEVEL=INFO
export CUDA_VISIBLE_DEVICES=0

# flashinfer JIT-compiles its sampling kernel on first use and needs a real CUDA
# toolkit to do it. Two separate problems follow, and both are handled here.
#
# The sampler itself is not what A2 is asking about, so it is switched off: the
# criterion is whether the aarch64 wheel emits a token, not whether a kernel can
# be compiled at run time. vLLM falls back to its native PyTorch sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0
#
# The toolkit still matters for anything else that JITs, and the default here is
# actively wrong: the only nvcc inside this venv is
# site-packages/nvidia/cu13/bin/nvcc, a CUDA 13 compiler pulled in as a
# dependency. This cluster's driver is 565, which is CUDA 12.7, and the LDM
# CUDA red line exists precisely because a cu13 build installs cleanly and then
# fails at run time. The recipe records the same trap for tilelang's GDN kernel.
# So point CUDA_HOME at the real 12.9 toolkit the slime line already built
# against, and say out loud which nvcc is in front.
export CUDA_HOME=/home/u6gb/kangli.u6gb/envs/ldm-rl-train
export PATH="$CUDA_HOME/bin:$PATH"
echo "[a2] CUDA_HOME=$CUDA_HOME"
echo "[a2] nvcc in front: $(command -v nvcc) -> $(nvcc --version 2>/dev/null | tail -1)"
echo "[a2] cu13 nvcc also present in venv: $(ls "$VENV"/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc 2>/dev/null || echo none)"
"$VENV/bin/python" "$D/a2_vllm_probe.py"
RC=$?
echo "[a2] probe exit=$RC"
exit $RC
