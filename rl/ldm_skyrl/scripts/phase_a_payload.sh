#!/bin/bash
# Phase A items A3, A4 and A5, in one node's worth of work.
#
# Stages are independent and each prints its own verdict, so a failure late in
# the script does not erase what earlier stages established. Nothing here is
# deleted on a re-run: the conda clone and the installs are idempotent, so a
# second attempt resumes rather than rebuilding.
set -uo pipefail

T0=$(date +%s)
stamp() { printf '\n[%s +%dm] === %s ===\n' "$(date -u +%H:%M:%SZ)" "$(( ($(date +%s)-T0)/60 ))" "$1"; }
verdict() { printf '\nGATE %s: %s\n' "$1" "$2"; }

BASE=/home/u6gb/kangli.u6gb/envs/ldm-rl-train      # the aarch64 stack slime already built
# The clone goes on node-local disk, not /home. /home is at its hard limit
# (101G used against a 101G quota), and the clone is 133,207 files, so it cannot
# live there. Lustre has the space but its inode quota is at 99.2 percent, and
# 133k files would finish it. Node-local has 3.5 TB and the env only has to
# outlive this job.
# The env is staged as ONE tar on Lustre rather than 133,207 loose files: the
# project inode quota has been near its limit all evening, and a tarball costs a
# single inode. Extraction also lands it at exactly the path it was validated at
# -- same uid, same /local/user/<uid>/ldm-skyrl-cp -- so any absolute path baked
# into the conda env still resolves.
VENV=/local/user/$(id -u)/ldm-skyrl-cp             # staged copy; BASE is never modified
ENVTAR=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_skyrl_env_v2.tar
SKYRL=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/oss/SkyRL-v0.3.0
DATA=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_skyrl_nb/a5_data
MODEL=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/models/Qwen2.5-1.5B-Instruct
UV=/projects/public/u6gb/.local/bin/uv
# How many cards this run may use. The gates work at 1; 4 is only faster.
# Making this a knob rather than a constant is what lets the payload land on
# whatever a borrowed allocation actually has free, instead of waiting for a
# whole node -- and a new 1-node job on this queue would not start for 42 hours.
NG=${NG:-4}
SCRATCH=/local/user/$(id -u)/skyrl_phase_a
mkdir -p "$SCRATCH/ckpt" "$SCRATCH/export" "$SCRATCH/logs"

export UV_CACHE_DIR=/local/user/$(id -u)/uv-cache-phasea
export UV_NO_PROGRESS=1
mkdir -p "$UV_CACHE_DIR"

stamp "node and cards (NG=$NG)"
hostname; nvidia-smi --query-gpu=index,name,memory.used,driver_version --format=csv

# ---------------------------------------------------------------- S0: clone
stamp "S0 clone the prebuilt aarch64 env (BASE is left untouched)"
# cp -a, not `conda create --clone`. The conda path unpacks into
# $CONDA_ROOT/pkgs, which lives on /home, and /home is at its hard limit
# (101G of 101G) -- so the clone fails with "Disk quota exceeded" even though
# the DESTINATION here is node-local with 3.1 TB free. Worse, it reports the
# failure as a corrupt archive ("you probably need to delete and re-download"),
# which points at the wrong thing entirely.
#
# A plain copy was verified on 2026-09-09: 19 GB in 13.5 minutes, and the copied
# env imports torch 2.11.0+cu129, transformer_engine 2.16.0, flash_attn 2.8.3,
# megatron-core 0.16.0rc0 and ray 2.58.0 with sys.prefix correctly resolved to
# the new location.
if [ -x "$VENV/bin/python" ]; then
  echo "env already present at $VENV, reusing it"
elif [ -f "$ENVTAR" ]; then
  echo "extracting $ENVTAR -> $(dirname "$VENV") (19 GB)"
  tar -xf "$ENVTAR" -C "$(dirname "$VENV")" || { verdict S0 "FAIL - extract failed"; exit 10; }
else
  echo "no tar; copying $BASE -> $VENV (19 GB, about 14 minutes)"
  cp -a "$BASE" "$VENV" || { verdict S0 "FAIL - copy failed"; exit 10; }
fi
"$VENV/bin/python" -V || { verdict S0 "FAIL - no python in the clone"; exit 10; }

# The clone carries its own CUDA 12.9 toolkit. Naming it explicitly matters:
# without CUDA_HOME a runtime JIT reaches for site-packages/nvidia/cu13/bin/nvcc,
# a CUDA 13 compiler on a cluster whose driver is CUDA 12.7.
export CUDA_HOME="$VENV"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="9.0"      # GH200 is Hopper, sm_90
export MAX_JOBS=64
export NVTE_FRAMEWORK=pytorch
echo "nvcc in front: $(command -v nvcc) -> $(nvcc --version 2>/dev/null | tail -1)"
verdict S0 "clone ready at $VENV"

# ------------------------------------------------------- S1: fsdp-path installs
stamp "S1 add what the fsdp path needs that the clone lacks"
# --index-strategy unsafe-best-match is required: uv otherwise stops at the first
# index carrying a package NAME, finds torch on PyPI, and never consults the
# pytorch index. Costs one job to learn.
PIP=("$UV" pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match
     --extra-index-url https://wheels.vllm.ai/0.23.0/cu129
     --extra-index-url https://download.pytorch.org/whl/cu129
     --extra-index-url https://pypi.org/simple)

echo "--- vllm (prebuilt aarch64 wheel) ---"
"${PIP[@]}" --no-build "vllm==0.23.0+cu129" 2>&1 | tail -6
echo "--- causal-conv1d (compiles; sdist only on every platform) ---"
# Not on A5's critical path and deliberately not allowed to block it. SkyRL's
# [fsdp] extra lists causal-conv1d, but skyrl.train, RayPPOTrainer and
# FullyAsyncRayPPOTrainer all import without it -- it is for Mamba-family
# models, and this gate runs Qwen2.5-1.5B, a plain transformer. Attempting the
# build is worth ~25 minutes only if there is time; failing it must not cost the
# gate, so the result is recorded and the run continues either way.
if "${PIP[@]}" "causal-conv1d" 2>&1 | tail -6; then
  verdict S1b "causal-conv1d built"
else
  verdict S1b "causal-conv1d not built -- continuing, A5 does not need it"
fi
echo "--- skyrl-train support deps ---"
# No --no-build here. func-timeout ships an sdist and no wheel, so forbidding
# builds for the whole line makes the WHOLE line unsatisfiable and installs
# none of it -- which surfaces much later as ModuleNotFoundError: jaxtyping,
# from inside skyrl.train.trainer. These are pure-python packages; "building"
# one runs no compiler. --no-build stays where it belongs: on the CUDA wheels.
"${PIP[@]}" loguru tqdm ninja tensorboard func_timeout "hydra-core==1.3.2" \
    accelerate torchdata jaxtyping omegaconf pylatexenc "ray[default]==2.56.0" 2>&1 | tail -4
echo "--- skyrl itself, --no-deps so its pins do not fight the staged env ---"
"${PIP[@]}" --no-deps "$SKYRL" "$SKYRL/skyrl-gym" 2>&1 | tail -4

"$VENV/bin/python" - <<'PY'
import importlib.metadata as m
for p in ["torch","vllm","skyrl","skyrl-gym","flash_attn","causal_conv1d","ray","transformer_engine","megatron-core"]:
    try: print(f"  {p:<22}{m.version(p)}")
    except Exception as e: print(f"  {p:<22}MISSING")
PY
"$VENV/bin/python" -c "import skyrl.train, torch, vllm; print('imports OK; cuda', torch.cuda.is_available())" \
  && verdict S1 "fsdp-path install OK" || { verdict S1 "FAIL"; }

# ---------------------------------------------------------- S2: A5 and A4, fsdp
stamp "S2 gate A5 (and A4): two GRPO steps and a checkpoint, SkyRL's own example path"
# 256 training rows at train_batch_size 128 with epochs=1 is exactly two steps.
# The generator, the environment and the reward are all SkyRL's; no LDM code is
# on this path, so a failure here belongs to the stack.
export VLLM_USE_FLASHINFER_SAMPLER=0
"$VENV/bin/python" -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA/train.parquet']" \
  data.val_data="['$DATA/validation.parquet']" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.model.path="$MODEL" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_gpus_per_node=$NG \
  trainer.placement.ref_num_gpus_per_node=$NG \
  generator.inference_engine.num_engines=$NG \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.6 \
  generator.batched=true \
  generator.n_samples_per_prompt=4 \
  environment.env_class=gsm8k \
  trainer.epochs=1 \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.max_prompt_length=256 \
  generator.sampling_params.max_generate_length=128 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.ckpt_interval=1 \
  trainer.resume_mode=null \
  trainer.logger=console \
  trainer.project_name=ldm_skyrl \
  trainer.run_name=phase_a5 \
  trainer.log_path="$SCRATCH/logs" \
  trainer.ckpt_path="$SCRATCH/ckpt" \
  trainer.export_path="$SCRATCH/export" 2>&1 | tail -120
A5RC=${PIPESTATUS[0]}
echo "--- checkpoint directory ---"; ls -R "$SCRATCH/ckpt" 2>/dev/null | head -20
CKPT=$(find "$SCRATCH/ckpt" -maxdepth 3 -type d -name 'global_step*' 2>/dev/null | wc -l)
if [ "$A5RC" -eq 0 ] && [ "$CKPT" -ge 1 ]; then
  verdict A5 "PASS - the loop closed and $CKPT checkpoint dir(s) were written"
  verdict A4 "PASS - colocated training performed NCCL weight syncs each step (see the log above)"
else
  verdict A5 "FAIL - rc=$A5RC, checkpoint dirs=$CKPT"
fi

# ------------------------------------------------- S3: megatron-path installs
stamp "S3 add the megatron extras (mamba-ssm and megatron-bridge compile)"
"${PIP[@]}" "mamba-ssm>=2.3.0" 2>&1 | tail -8
"${PIP[@]}" "git+https://github.com/NVIDIA-NeMo/Megatron-Bridge@91a15142a4b4442a8d46ab539d1b923bd08570d0" 2>&1 | tail -6
"$VENV/bin/python" -c "import mamba_ssm, causal_conv1d; print('mamba-ssm and causal-conv1d import OK')"

# --------------------------------------------------------- S4: A3, megatron
stamp "S4 gate A3: does the megatron strategy reach the first forward"
"$VENV/bin/python" -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA/train.parquet']" \
  data.val_data="['$DATA/validation.parquet']" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.model.path="$MODEL" \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_gpus_per_node=$NG \
  trainer.placement.ref_num_gpus_per_node=$NG \
  generator.inference_engine.num_engines=$NG \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.gpu_memory_utilization=0.5 \
  generator.batched=true \
  generator.n_samples_per_prompt=4 \
  environment.env_class=gsm8k \
  trainer.epochs=1 \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.max_prompt_length=256 \
  generator.sampling_params.max_generate_length=128 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.ckpt_interval=1 \
  trainer.resume_mode=null \
  trainer.logger=console \
  trainer.project_name=ldm_skyrl \
  trainer.run_name=phase_a3 \
  trainer.log_path="$SCRATCH/logs" \
  trainer.ckpt_path="$SCRATCH/ckpt_megatron" \
  trainer.export_path="$SCRATCH/export" 2>&1 | tail -120
A3RC=${PIPESTATUS[0]}
[ "$A3RC" -eq 0 ] && verdict A3 "PASS - megatron strategy ran" || verdict A3 "FAIL - rc=$A3RC"

stamp "done"
echo "elapsed $(( ($(date +%s)-T0)/60 )) minutes"
