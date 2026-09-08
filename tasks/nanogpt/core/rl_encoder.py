"""Fixed-length feature encoding of a nanoGPT config, for the GP surrogate.

``RBFGPSurrogate`` uses a **single scalar lengthscale** over the whole vector
(``ldm_tts/optimization/gp.py::_rbf_kernel``), so every feature has to arrive
pre-normalised onto a comparable scale or one wide-ranged coordinate silently
dominates the kernel. Everything below is mapped into ``[0, 1]`` using the
schema's own bounds, log-scaled wherever the schema declares
``"scale": "log"``.

Two deliberate representation choices, both of which cost nothing to revert
(bump ``FEATURE_VERSION``) but matter a lot with few observations:

**1. Derived geometry is included, not just raw knobs.** The failure that
motivated this whole rewrite was a surrogate in which ``TOTAL_BATCH_SIZE`` was
exactly inert, while changing it alone cost 0.023 bpb on the real trainer (~17
sigma). The reason is that ``TOTAL_BATCH_SIZE`` does not act on its own: it
sets ``grad_accum_steps = TOTAL_BATCH_SIZE / (DEVICE_BATCH_SIZE * 2048)`` and
therefore how many optimizer steps fit in the wall-clock budget. In raw-knob
coordinates that interaction has to be learned; in derived coordinates it is
handed to the GP. Same argument for ``total_params`` (a nonlinear function of
DEPTH x ASPECT_RATIO x HEAD_DIM) and ``flops_per_token``.

**2. ``WINDOW_PATTERN`` becomes one continuous ``long_frac``, not a 6-way
one-hot.** The trainer only consumes the pattern through per-layer window
sizes, and its quality effect is monotone-with-saturation in the fraction of
full-context layers. One informative coordinate beats five mostly-zero ones
when the GP has tens of observations rather than thousands.

Normalisation bounds for the derived features are computed from the schema's
own corners at import time (see :func:`_derived_bounds`) rather than hardcoded,
so editing the schema cannot silently push features outside ``[0, 1]``.
"""

from __future__ import annotations

import math
from typing import Any

from ldm_tts.contracts import Candidate, SurrogateSpaceSpec
from ldm_tts.optimization.records import SurrogateVector

from tasks.nanogpt.core import rl_knobs as knobs_mod

#: Bump on ANY change to the feature layout or normalisation. A GP fitted on
#: one version's vectors must never be fed another's.
FEATURE_VERSION = "nanogpt_knob_geometry_v1"

FEATURE_NAMES: tuple[str, ...] = (
    "depth",
    "aspect_ratio",
    "head_dim",
    "long_frac",
    "total_batch_size",
    "device_batch_size",
    "embedding_lr",
    "unembedding_lr",
    "matrix_lr",
    "scalar_lr",
    "weight_decay",
    "warmup_ratio",
    "warmdown_ratio",
    "final_lr_frac",
    "log_total_params",
    "log_flops_per_token",
    "log_grad_accum_steps",
)
FEATURE_DIM = len(FEATURE_NAMES)

_LINEAR_KNOBS = ("WEIGHT_DECAY", "WARMUP_RATIO", "WARMDOWN_RATIO", "FINAL_LR_FRAC")
_LOG_KNOBS = ("EMBEDDING_LR", "UNEMBEDDING_LR", "MATRIX_LR", "SCALAR_LR")


def _unit(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return min(1.0, max(0.0, (float(value) - low) / (high - low)))


def _log_unit(value: float, low: float, high: float) -> float:
    lo = math.log(max(low, 1e-12))
    hi = math.log(max(high, 1e-12))
    return _unit(math.log(max(float(value), 1e-12)), lo, hi)


def _derived_bounds(schema: dict | None = None) -> dict[str, tuple[float, float]]:
    """Min/max of the derived features over the schema's corners.

    ``total_params`` and ``flops_per_token`` are monotone in DEPTH and
    ASPECT_RATIO, so their extremes sit at corner combinations; HEAD_DIM and
    WINDOW_PATTERN are enumerated in full because they are small choice sets.
    ``grad_accum_steps`` is enumerated over the legal (TOTAL, DEVICE) pairs
    only -- illegal pairs never reach the encoder.
    """

    schema = schema if schema is not None else knobs_mod.load_schema()
    depths = (schema["DEPTH"]["min"], schema["DEPTH"]["max"])
    ratios = (schema["ASPECT_RATIO"]["min"], schema["ASPECT_RATIO"]["max"])
    head_dims = schema["HEAD_DIM"]["choices"]
    patterns = schema["WINDOW_PATTERN"]["choices"]

    params: list[float] = []
    flops: list[float] = []
    for depth in depths:
        for ratio in ratios:
            for head_dim in head_dims:
                model_dim, _ = knobs_mod.derive_shape(depth, ratio, head_dim)
                total, _ = knobs_mod.count_params(depth, model_dim)
                params.append(float(total))
                for pattern in patterns:
                    flops.append(
                        knobs_mod.flops_per_token(depth, model_dim, pattern)
                    )

    accums: list[float] = []
    for total_batch in schema["TOTAL_BATCH_SIZE"]["choices"]:
        for device_batch in schema["DEVICE_BATCH_SIZE"]["choices"]:
            tokens = device_batch * knobs_mod.MAX_SEQ_LEN
            if total_batch % tokens == 0:
                accums.append(total_batch / tokens)

    return {
        "log_total_params": (math.log(min(params)), math.log(max(params))),
        "log_flops_per_token": (math.log(min(flops)), math.log(max(flops))),
        "log_grad_accum_steps": (math.log(min(accums)), math.log(max(accums))),
    }


_DERIVED_BOUNDS = _derived_bounds()


def encode_config(knobs: dict[str, Any], schema: dict | None = None) -> tuple[float, ...]:
    """Map one validated config onto :data:`FEATURE_DIM` features in ``[0, 1]``."""

    schema = schema if schema is not None else knobs_mod.load_schema()
    geo = knobs_mod.geometry(knobs)

    values: list[float] = [
        _unit(knobs["DEPTH"], schema["DEPTH"]["min"], schema["DEPTH"]["max"]),
        _unit(
            knobs["ASPECT_RATIO"],
            schema["ASPECT_RATIO"]["min"],
            schema["ASPECT_RATIO"]["max"],
        ),
        _unit(
            schema["HEAD_DIM"]["choices"].index(knobs["HEAD_DIM"]),
            0,
            len(schema["HEAD_DIM"]["choices"]) - 1,
        ),
        float(geo["long_frac"]),
        _log_unit(
            knobs["TOTAL_BATCH_SIZE"],
            min(schema["TOTAL_BATCH_SIZE"]["choices"]),
            max(schema["TOTAL_BATCH_SIZE"]["choices"]),
        ),
        _log_unit(
            knobs["DEVICE_BATCH_SIZE"],
            min(schema["DEVICE_BATCH_SIZE"]["choices"]),
            max(schema["DEVICE_BATCH_SIZE"]["choices"]),
        ),
    ]
    for name in _LOG_KNOBS:
        values.append(_log_unit(knobs[name], schema[name]["min"], schema[name]["max"]))
    for name in _LINEAR_KNOBS:
        values.append(_unit(knobs[name], schema[name]["min"], schema[name]["max"]))

    values.append(
        _unit(math.log(geo["total_params"]), *_DERIVED_BOUNDS["log_total_params"])
    )
    values.append(
        _unit(
            math.log(geo["flops_per_token"]), *_DERIVED_BOUNDS["log_flops_per_token"]
        )
    )
    values.append(
        _unit(
            math.log(max(geo["grad_accum_steps"], 1e-12)),
            *_DERIVED_BOUNDS["log_grad_accum_steps"],
        )
    )

    if len(values) != FEATURE_DIM:  # pragma: no cover - guards a refactor slip
        raise AssertionError(
            f"encoder produced {len(values)} features, expected {FEATURE_DIM}"
        )
    return tuple(float(v) for v in values)


def surrogate_spec() -> SurrogateSpaceSpec:
    """The task-spec declaration this encoder must agree with.

    ``LDMEnv._validate_optimizer_config`` compares ``kind``,
    ``dimension_policy``, ``dimension`` and ``version`` between the task spec
    and the encoder's ``describe()``, and refuses to build if they differ --
    which is what stops a GP being fed vectors of a shape it was not fitted on.
    """

    return SurrogateSpaceSpec(
        kind="vector",
        representation=(
            "normalised nanoGPT knobs plus derived geometry "
            "(log params / flops-per-token / grad-accum steps)"
        ),
        dimension_policy="fixed",
        dimension=FEATURE_DIM,
        encoder="tasks.nanogpt.core.rl_encoder.NanogptSurrogateEncoder",
        version=FEATURE_VERSION,
        metadata={"feature_names": list(FEATURE_NAMES)},
    )


class NanogptSurrogateEncoder:
    """``SurrogateEncoder`` over knob-dict candidates."""

    def describe(self) -> SurrogateSpaceSpec:
        return surrogate_spec()

    def encode(self, candidate: Candidate) -> SurrogateVector:
        config = candidate.payload["config"]
        return SurrogateVector(
            encode_config(config),
            FEATURE_VERSION,
            source_id=candidate.candidate_id,
            metadata={"canonical_key": candidate.canonical_key},
        )


__all__ = [
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "FEATURE_VERSION",
    "NanogptSurrogateEncoder",
    "encode_config",
    "surrogate_spec",
]
