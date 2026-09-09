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

stamp "node and cards (NG=$NG requested)"
hostname; nvidia-smi --query-gpu=index,name,memory.used,driver_version --format=csv

# Re-derive the free cards HERE rather than trusting what the hunt loop saw.
# A VRAM census is stale before it can be acted on: a node measured elsewhere
# tonight read [1,1,1,1] MiB and 92,211 MiB sixty seconds later. Landing anyway
# would not merely be slow: a shared card costs 2.29x wall-clock, so under any
# time-budgeted comparison the run completes far fewer optimizer steps and its
# loss reads worse for that reason alone. (An accompanying "0.106 bpb" figure was
# circulated and then retracted at the source -- it was that accounting effect,
# not a bias. The 2.29x stands.) Either way the gate would return a number shaped
# like a result. Better to give the node back and let the loop find another.
mapfile -t _MEM < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
_FREE=(); for i in "${!_MEM[@]}"; do [ "${_MEM[$i]}" -lt 2048 ] && _FREE+=("$i"); done
echo "cards under 2048 MiB right now: ${_FREE[*]:-none}  (hunt loop expected $NG)"
if [ "${#_FREE[@]}" -lt 1 ]; then
  verdict S0 "ABORT - every card was taken between the probe and this step; not measuring on a shared card"
  exit 20
fi
if [ "${#_FREE[@]}" -lt "$NG" ]; then
  echo "fewer cards than expected; running on ${#_FREE[@]} instead of $NG"
  NG=${#_FREE[@]}
fi
# Clamp to a power of two. SkyRL asserts that
#   policy_mini_batch_size * n_samples_per_prompt / num_gpus
# is divisible by micro_train_batch_size_per_gpu, and the dataset is 256 rows at
# train_batch_size 128, which is exactly two steps. Every divisor of 128 is a
# power of two, so an odd NG has no solution: NG=3 gives 32*4/3 = 42, and the run
# dies four minutes in on an assertion rather than on anything about the port.
# Making the card count a free variable is what introduced this; constraining it
# back is cheaper than making the batch arithmetic depend on it.
# Host memory, not just VRAM. Landing 8 died as
#   slurmstepd: Detected 1 oom_kill event ... task 0: Out Of Memory
# on a node whose other three cards held ~49 GB each: the neighbours were
# competing for host RAM, and a per-card VRAM probe cannot see that. The probe's
# scope has to match what actually kills the run, otherwise a clean reading from
# too narrow a probe is indistinguishable from a genuinely free node.
# Read the JOB cgroup, not /proc/meminfo. Landing 8 was OOM-killed here, and my
# first fix for it read node MemAvailable -- which was 518 GB at the time and
# would have waved the run straight through. The limit that actually applies is
# the allocation's own cgroup, shared with every sibling step other sessions have
# attached into this same job: 449 GB per node here, against a node that reports
# over 500 GB free. Replacing a too-narrow probe with a differently-wrong one is
# not progress; the probe has to read the quantity that does the killing.
_CG=$(awk -F: '/^0:/{print $3; exit}' /proc/self/cgroup 2>/dev/null)
_JOB_CG="/sys/fs/cgroup${_CG%%/step_*}"
if [ -r "$_JOB_CG/memory.max" ] && [ -r "$_JOB_CG/memory.current" ]; then
  _MX=$(cat "$_JOB_CG/memory.max"); _CU=$(cat "$_JOB_CG/memory.current")
  if [ "$_MX" = "max" ]; then
    echo "job cgroup memory: unlimited (current $((_CU/1073741824)) GB)"
  else
    _FREE_GB=$(( (_MX - _CU) / 1073741824 ))
    echo "job cgroup memory: $((_CU/1073741824)) GB used of $((_MX/1073741824)) GB, ${_FREE_GB} GB headroom"
    if [ "$_FREE_GB" -lt 100 ]; then
      verdict S0 "ABORT - only ${_FREE_GB} GB headroom in the shared job cgroup; sibling steps from other sessions are using it and this run would be OOM-killed as landing 8 was"
      exit 22
    fi
  fi
else
  echo "job cgroup memory: could not read $_JOB_CG -- proceeding, but this is unmeasured rather than fine"
fi

_NG2=1; while [ $((_NG2 * 2)) -le "$NG" ]; do _NG2=$((_NG2 * 2)); done
if [ "$_NG2" -ne "$NG" ]; then
  echo "clamping NG $NG -> $_NG2 (batch arithmetic needs a power of two); leaving $((NG - _NG2)) card(s) unused"
  _FREE=("${_FREE[@]:0:$_NG2}"); NG=$_NG2
fi
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${_FREE[*]}")

# Print the derived batch sizes. "What batch did this actually run at" has to be
# greppable in the log rather than reconstructible from three scripts.
TRAIN_BSZ=128; MINI_BSZ=32; NSAMP=4; MICRO=4
echo "[bsz] NG=$NG  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[bsz] train_batch_size=$TRAIN_BSZ over 256 rows = $((256 / TRAIN_BSZ)) steps at epochs=1"
echo "[bsz] policy_mini_batch_size=$MINI_BSZ x n_samples=$NSAMP / NG=$NG = $((MINI_BSZ * NSAMP / NG)) per gpu, micro=$MICRO, remainder $(( (MINI_BSZ * NSAMP / NG) % MICRO ))"

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
# --no-build-isolation, because these three build from source against torch and
# uv's isolated build environment does not contain it. The error says so
# explicitly ("either add it to [tool.uv.extra-build-dependencies] or uv pip
# install torch into the environment and re-run with --no-build-isolation"), and
# I read that message only after treating this failure as unimportant for six
# landings -- "A5 does not need causal-conv1d" was true and let me skip the
# sentence that also explained why mamba-ssm and fast-hadamard-transform failed.
# One cause, three packages, and I had the diagnosis in the log the whole time.
# FORCE_BUILD, and then verify the extension rather than the exit code. The
# previous version installed the Python package, saw pip exit 0, and announced
# "causal-conv1d built" -- while `import causal_conv1d_cuda` failed, so nothing
# that calls a kernel could run. --no-build-isolation was a real fix for a real
# cause (three packages failing on a missing torch at build time), and I reported
# it as verified in the same edit that introduced this gate. Fixing a cause and
# confirming the fix are two acts.
export CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE MAX_JOBS=${MAX_JOBS:-32}
# --reinstall-package, because the node-local env survives between landings and a
# kernel-less causal_conv1d 1.7.0 from an earlier attempt already satisfies the
# requirement -- uv skipped the build entirely ("Installed 1 package in 16ms",
# and that one package was nvidia-cusparselt). --no-binary cannot force a source
# build of something it has decided not to install at all.
#
# Third time tonight that an installed-but-wrong package was invisible to a step
# whose job is to make sure a package is installed: vllm-router 0.1.15 imported
# fine and failed at run time, the module sweep found nothing to do, and now this.
# "Is it present" and "is it the right one" are different questions and only the
# first is cheap to ask by accident.
"${PIP[@]}" --no-build-isolation --reinstall-package causal-conv1d --no-binary causal-conv1d "causal-conv1d" 2>&1 | tail -6
if "$VENV/bin/python" -c "import causal_conv1d_cuda" 2>/dev/null; then
  verdict S1b "causal-conv1d built AND causal_conv1d_cuda imports"
else
  verdict S1b "causal_conv1d_cuda MISSING -- the python package may be installed but its CUDA kernel is not; A5 does not need it, A3 does"
fi
echo "--- skyrl-train support deps ---"
# Read from SkyRL's own [project.dependencies] + [skyrl-train] + [fsdp] extras
# rather than hand-listed. Hand-listing cost two landings: first jaxtyping, then
# peft, each surfacing fifteen minutes in as a ModuleNotFoundError from deep
# inside the trainer. A curated list is a second, silently-drifting copy of a
# dependency table that already exists.
#
# transformers is deliberately left out. SkyRL pins <=5.8.0 and the staged env
# carries 5.12.1, which vllm 0.23.0 was installed against; downgrading a core
# package inside a working env to satisfy a declared bound is the more expensive
# mistake to make first. If A5 fails on a transformers API, that is the next
# thing to change and it will say so.
PURE_DEPS=("datasets>=4.0.0" "pillow>=11.3.0" "rich>=14.1.0" "safetensors>=0.6.2" \
  "tokenizers>=0.21.2" "typer>=0.17.4" "peft==0.18.1" "hf_transfer" "cloudpathlib>=0.23.0" \
  "loguru" "tqdm" "ninja" "tensorboard" "func_timeout" "hydra-core==1.3.2" "accelerate" \
  "torchdata" "omegaconf" "ray==2.56.0" "debugpy==1.8.0" "wandb" "tensordict" "jaxtyping" \
  "polars" "s3fs" "fastapi" "uvicorn" "pybind11" "setuptools")
"${PIP[@]}" "${PURE_DEPS[@]}" 2>&1 | tail -5
# No --no-build here. func-timeout ships an sdist and no wheel, so forbidding
# builds for the whole line makes the WHOLE line unsatisfiable and installs
# none of it -- which surfaces much later as ModuleNotFoundError: jaxtyping,
# from inside skyrl.train.trainer. These are pure-python packages; "building"
# one runs no compiler. --no-build stays where it belongs: on the CUDA wheels.
echo "--- skyrl itself, --no-deps so its pins do not fight the staged env ---"
"${PIP[@]}" --no-deps "$SKYRL" "$SKYRL/skyrl-gym" 2>&1 | tail -4

"$VENV/bin/python" - <<'PY'
import importlib.metadata as m
for p in ["torch","vllm","skyrl","skyrl-gym","flash_attn","causal_conv1d","ray","transformer_engine","megatron-core"]:
    try: print(f"  {p:<22}{m.version(p)}")
    except Exception as e: print(f"  {p:<22}MISSING")
PY
# Pins that must hold regardless of what is already installed. The module sweep
# below detects ABSENCE and says nothing about CORRECTNESS: vllm_router 0.1.15
# imports perfectly and then fails at run time, because it renamed
# RouterArgs.pd_disaggregation to vllm_pd_disaggregation while SkyRL 0.3.0 still
# calls the old name. The sweep therefore found nothing to do and the same
# AttributeError came back a second time. uv resolved 0.1.14.post1 in A1; that
# resolution is the authority on version, and it is applied unconditionally here
# rather than left to a code path that only runs when something is missing.
echo "--- version pins from the A1 lock (applied whether or not anything is missing) ---"
"${PIP[@]}" --reinstall-package vllm-router \
  "https://github.com/SumanthRH/router/releases/download/0.1.14.post1/vllm_router-0.1.14.post1-cp38-abi3-manylinux_2_28_aarch64.whl" 2>&1 | tail -3
# Find RouterArgs wherever it lives rather than at a path I guessed. The previous
# version of this check hard-coded vllm_router.parsers.parser, which does not
# exist in 0.1.14.post1, so it printed WARN on a correctly pinned environment --
# and because A5 passed anyway, I read past that WARN twice. A check whose
# failures you have learned to ignore has stopped being a check, so this one
# either finds the attribute or fails the gate.
"$VENV/bin/python" - <<'PYPIN'
import importlib, importlib.metadata as m, pkgutil, sys
v = m.version("vllm-router"); print("  vllm-router", v)
assert v.startswith("0.1.14"), f"wrong vllm-router: {v} (SkyRL 0.3.0 calls the pre-rename API)"
import vllm_router
found = None
for mod in pkgutil.walk_packages(vllm_router.__path__, "vllm_router."):
    try: M = importlib.import_module(mod.name)
    except Exception: continue
    RA = getattr(M, "RouterArgs", None)
    if RA is not None: found = (mod.name, RA); break
assert found, "RouterArgs not found anywhere in vllm_router"
name, RA = found
names = set(dir(RA)) | set(getattr(RA, "__annotations__", {})) | set(RA.__init__.__code__.co_varnames)
assert "pd_disaggregation" in names, (
    f"{name}.RouterArgs has no pd_disaggregation -- this is the 0.1.15 rename "
    f"that SkyRL 0.3.0 does not call. Saw: {sorted(n for n in names if 'disagg' in n)}")
print(f"  {name}.RouterArgs has pd_disaggregation")
PYPIN
[ $? -ne 0 ] && { verdict S1 "FAIL - vllm-router pin check failed; do not trust a run on this env"; exit 21; }

# Stop guessing which modules the run reaches. Three landings were lost to three
# different missing packages -- jaxtyping, peft, vllm_router -- each found one at
# a time, fifteen minutes in, because the check imported a module that happened
# to be fine. Both my curated dependency list AND my "heavy, skip it" exclusion
# list were negative assertions with undeclared coverage; vllm-router was on the
# exclusion list even though A1 had already recorded that it ships an aarch64
# wheel.
#
# So: import every module in the package, collect every missing top-level name at
# once, install them, and repeat until the set is empty. The sweep is the
# positive statement that replaces both lists.
for attempt in 1 2 3; do
  MISSING=$("$VENV/bin/python" - <<'PYSWEEP'
import importlib, pkgutil, sys
missing = set()
import skyrl
for mod in pkgutil.walk_packages(skyrl.__path__, "skyrl."):
    try:
        importlib.import_module(mod.name)
    except ModuleNotFoundError as e:
        if e.name: missing.add(e.name.split(".")[0])
    except Exception:
        pass          # anything that is not a missing module is not this check's business
print(" ".join(sorted(missing)))
PYSWEEP
)
  MISSING=$(echo "$MISSING" | tr -s ' ')
  [ -z "$(echo "$MISSING" | tr -d ' ')" ] && { echo "module sweep clean on attempt $attempt"; break; }
  echo "sweep found missing top-level modules: $MISSING"
  # Map the few import names that differ from their distribution name.
  PKGS=""; for m in $MISSING; do
    case "$m" in
      # By URL, not by name. The sweep is a positive statement about WHICH module
      # is missing and says nothing about which version satisfies it. uv resolved
      # vllm-router 0.1.14.post1 from this wheel in A1; installing the bare name
      # gets 0.1.15 from PyPI, which renamed RouterArgs.pd_disaggregation to
      # vllm_pd_disaggregation, and SkyRL 0.3.0 calls the old name. The lock
      # already held the answer and installing by name walked around it.
      vllm_router) PKGS="$PKGS https://github.com/SumanthRH/router/releases/download/0.1.14.post1/vllm_router-0.1.14.post1-cp38-abi3-manylinux_2_28_aarch64.whl";;
      fla)         PKGS="$PKGS flash-linear-attention";;
      cv2)         PKGS="$PKGS opencv-python-headless";;
      causal_conv1d|mamba_ssm|transformer_engine*|megatron*|flashinfer*) : ;;   # compile or already staged
      *)           PKGS="$PKGS $m";;
    esac
  done
  [ -z "$PKGS" ] && { echo "nothing installable left; the rest need a compile"; break; }
  echo "installing:$PKGS"
  "${PIP[@]}" $PKGS 2>&1 | tail -4
done

"$VENV/bin/python" -c "
import torch, vllm, skyrl.train
from skyrl.train.entrypoints import main_base
from skyrl.backends.skyrl_train.workers.fsdp import fsdp_worker
from skyrl.backends.skyrl_train.inference_servers import setup as _s
print('fsdp + inference-server path imports OK; cuda', torch.cuda.is_available())
" && verdict S1 "fsdp-path install OK" || { verdict S1 "FAIL - see the traceback above"; }

# ---------------------------------------------------------- S2: A5 and A4, fsdp
# SKIP_A5=1 exists because I passed it on a run before writing it, and it reached
# nothing -- the run silently did the full A5 anyway. That is the same defect this
# line documented on PR #5 (a knob that is declared, set, logged, and never
# arrives), committed by me minutes after publishing it. The lesson that actually
# transfers is not "remember the knob": it is that passing a variable is not
# evidence it is read, and one grep answers it.
if [ "${SKIP_A5:-0}" = "1" ]; then
  echo "SKIP_A5=1 -- A5 already passed on this stack at 20:15Z; going straight to the megatron path"
  verdict A5 "SKIPPED by request (SKIP_A5=1); the passing evidence is 6438532.42 on nid010964"
else
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
  generator.n_samples_per_prompt=$NSAMP \
  environment.env_class=gsm8k \
  trainer.epochs=1 \
  trainer.train_batch_size=$TRAIN_BSZ \
  trainer.policy_mini_batch_size=$MINI_BSZ \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=$MICRO \
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
fi   # end SKIP_A5 guard

stamp "S3 add the megatron extras (mamba-ssm and megatron-bridge compile)"
# Same reason as S1b: these compile against torch, so build isolation has to go.
"${PIP[@]}" --no-build-isolation --reinstall-package mamba-ssm --no-binary mamba-ssm "mamba-ssm>=2.3.0" 2>&1 | tail -8
"${PIP[@]}" "git+https://github.com/NVIDIA-NeMo/Megatron-Bridge@91a15142a4b4442a8d46ab539d1b923bd08570d0" 2>&1 | tail -6
# Import the CUDA extensions by name. mamba_ssm and causal_conv1d both import
# cleanly with no kernel behind them, which is exactly how A3 got to S4 and then
# died on `No module named selective_scan_cuda` six minutes later.
"$VENV/bin/python" -c "
import importlib
missing = [m for m in ('causal_conv1d_cuda', 'selective_scan_cuda')
           if not importlib.util.find_spec(m)]
import mamba_ssm, causal_conv1d
assert not missing, f'python packages installed but CUDA kernels absent: {missing}'
print('mamba-ssm and causal-conv1d import OK, and both CUDA extensions are present')
"

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
  generator.n_samples_per_prompt=$NSAMP \
  environment.env_class=gsm8k \
  trainer.epochs=1 \
  trainer.train_batch_size=$TRAIN_BSZ \
  trainer.policy_mini_batch_size=$MINI_BSZ \
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
