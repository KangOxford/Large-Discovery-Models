"""Environment variables that have to reach Ray workers, and why each one does.

SkyRL's ``initialize_ray`` has no hook for extra variables, so the recipe wraps
it.  The list below is the LDM equivalent.  Two entries are load-bearing:

``CUDA_HOME``
    The tilelang JIT that compiles the gated-delta-net kernels needs a full CUDA
    toolkit.  The recipe records that without it the GDN backward dies on a
    missing ``cuda/atomic`` header.  The 9B failure on the slime line is in the
    same place -- ``linear_attn.A_log`` and ``dt_bias`` gradients are all
    non-finite -- which is a lead, not a diagnosis.  Phase item C5 is what turns
    it into one.

``TASK_PYTHON``
    The training stack and the evaluation stack are separate interpreters, a
    boundary ``ldm_rl/remote_env.py`` already enforces by stripping
    ``LD_LIBRARY_PATH`` / ``CUDA_HOME`` / ``CUDA_VISIBLE_DEVICES`` from the
    subprocess it starts.  The worker has to know where the other interpreter is.
"""

from __future__ import annotations

import os

FORWARDED_ENV_VARS = (
    "CUDA_HOME",       # tilelang GDN JIT; see module docstring
    "CUDA_PATH",
    "TASK_PYTHON",     # evaluation-stack interpreter
    "VINA_BIN",
    "NN_MODEL",
    "HF_TOKEN",
    "WANDB_API_KEY",
)


def collect_forwarded_env() -> dict[str, str]:
    """The subset of ``FORWARDED_ENV_VARS`` that is actually set."""
    return {k: os.environ[k] for k in FORWARDED_ENV_VARS if os.environ.get(k)}


def missing_required(required: tuple[str, ...] = ("CUDA_HOME", "TASK_PYTHON")) -> list[str]:
    """Required variables that are absent.  Callers report; nothing is guessed."""
    return [k for k in required if not os.environ.get(k)]


__all__ = ["FORWARDED_ENV_VARS", "collect_forwarded_env", "missing_required"]
