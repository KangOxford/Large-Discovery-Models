"""The nanoGPT RL action space: 14 hyperparameter knobs of ``scripts/train.py``.

This module is the single source of truth for *what a proposal is* and *whether
the real trainer would accept it*. The RL candidate domain, the surrogate
encoder and the prompt renderer all import from here, so the action space
cannot drift apart between them.

Everything here mirrors ``tasks/nanogpt/scripts/train.py`` (equivalently
``resources/train/real_train.py``):

* the knob names/ranges/choices come from
  ``resources/schemas/real_operations.json`` (version ``real_train_knobs_v1``);
* :data:`DEFAULTS` are the values committed at ``train.py`` lines 433-451, i.e.
  the baseline a proposal has to beat;
* :func:`derive_shape`, :func:`count_params`, :func:`window_sizes` and
  :func:`flops_per_token` reproduce ``Model``'s geometry so that a proposal can
  be described (and encoded for a GP) without running the trainer.

There is deliberately **no analytic val_bpb surrogate here**. An earlier
version of this task shipped one; its ranking inverted against the real trainer
(Spearman -0.60 over 7 policy proposals, 6 of them genuine 17-50 sigma
regressions), so the only score this task recognises is a measured one. See
``rl_eval.py``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable

_HERE = os.path.dirname(os.path.abspath(__file__))
_TASK_ROOT = os.path.dirname(_HERE)

SCHEMA_PATH = os.environ.get(
    "NANOGPT_KNOB_SCHEMA",
    os.path.join(_TASK_ROOT, "resources", "schemas", "real_operations.json"),
)

# --- mirrored from scripts/prepare.py -------------------------------------
MAX_SEQ_LEN = 2048
VOCAB_SIZE = 8192

# --- mirrored from scripts/train.py ---------------------------------------
DEFAULT_TIME_BUDGET = 300          # train.py TIME_BUDGET
H100_BF16_PEAK_FLOPS = 989.5e12
MLP_EXPANSION = 4
VRAM_BUDGET_MB = 80 * 1024

#: train.py's committed top-level assignments (lines 433-451). "Baseline"
#: means these values, and it means the same thing on both sides: the schema
#: and these defaults are identical to what the trainer ships.
DEFAULTS: dict[str, Any] = {
    "ASPECT_RATIO": 64,
    "HEAD_DIM": 128,
    "WINDOW_PATTERN": "SSSL",
    "TOTAL_BATCH_SIZE": 2 ** 19,
    "EMBEDDING_LR": 0.6,
    "UNEMBEDDING_LR": 0.004,
    "MATRIX_LR": 0.04,
    "SCALAR_LR": 0.5,
    "WEIGHT_DECAY": 0.2,
    "WARMUP_RATIO": 0.0,
    "WARMDOWN_RATIO": 0.5,
    "FINAL_LR_FRAC": 0.0,
    "DEPTH": 8,
    "DEVICE_BATCH_SIZE": 128,
}

#: Knob order used by every fixed-length encoding. Frozen: changing it
#: invalidates every GP fitted on a previously written feature vector.
KNOB_ORDER: tuple[str, ...] = (
    "DEPTH",
    "ASPECT_RATIO",
    "HEAD_DIM",
    "WINDOW_PATTERN",
    "TOTAL_BATCH_SIZE",
    "DEVICE_BATCH_SIZE",
    "EMBEDDING_LR",
    "UNEMBEDDING_LR",
    "MATRIX_LR",
    "SCALAR_LR",
    "WEIGHT_DECAY",
    "WARMUP_RATIO",
    "WARMDOWN_RATIO",
    "FINAL_LR_FRAC",
)


class ProposalError(ValueError):
    """A well-formed proposal that the real trainer would reject."""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_CACHE: dict[str, dict] = {}


def load_schema(path: str = SCHEMA_PATH) -> dict:
    """Return the ``parameters`` block of the real knob schema."""

    if path not in _SCHEMA_CACHE:
        with open(path) as handle:
            _SCHEMA_CACHE[path] = json.load(handle)["parameters"]
    return _SCHEMA_CACHE[path]


def schema_version(path: str = SCHEMA_PATH) -> str:
    with open(path) as handle:
        return str(json.load(handle).get("version", ""))


def knob_names(path: str = SCHEMA_PATH) -> list[str]:
    return list(load_schema(path).keys())


# ---------------------------------------------------------------------------
# Geometry, mirroring train.py's Model
# ---------------------------------------------------------------------------


def derive_shape(depth: int, aspect_ratio: int, head_dim: int) -> tuple[int, int]:
    """Return ``(model_dim, num_heads)``.

    ``model_dim`` is ``depth * aspect_ratio`` rounded UP to a multiple of
    ``head_dim`` so that ``n_embd % n_head == 0`` (train.py:68).
    """

    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    return model_dim, model_dim // head_dim


def count_params(depth: int, model_dim: int) -> tuple[int, int]:
    """Return ``(total_params, flops_counted_params)``.

    ``flops_counted`` excludes the token embedding, matching
    ``Model.estimate_flops``, which subtracts ``wte``.
    """

    attn = 4 * model_dim * model_dim
    mlp = 2 * MLP_EXPANSION * model_dim * model_dim
    body = (attn + mlp) * depth
    wte = VOCAB_SIZE * model_dim
    lm_head = VOCAB_SIZE * model_dim
    return body + wte + lm_head, body + lm_head


def window_sizes(depth: int, pattern: str) -> list[int]:
    """Per-layer attention window, mirroring ``Model._compute_window_sizes``.

    ``S`` is half context, ``L`` is full context, the pattern cycles, and the
    last layer is always forced to full context.
    """

    out = [
        MAX_SEQ_LEN if pattern[i % len(pattern)].upper() == "L" else MAX_SEQ_LEN // 2
        for i in range(depth)
    ]
    out[-1] = MAX_SEQ_LEN
    return out


def flops_per_token(depth: int, model_dim: int, pattern: str) -> float:
    _, counted = count_params(depth, model_dim)
    # num_heads * head_dim == model_dim, so train.py's 12*h*q*seq term collapses.
    attn_flops = sum(12 * model_dim * eff for eff in window_sizes(depth, pattern))
    return 6 * counted + attn_flops


def geometry(knobs: dict) -> dict[str, float]:
    """Everything derivable from a config without running the trainer."""

    model_dim, num_heads = derive_shape(
        knobs["DEPTH"], knobs["ASPECT_RATIO"], knobs["HEAD_DIM"]
    )
    total_params, counted = count_params(knobs["DEPTH"], model_dim)
    fpt = flops_per_token(knobs["DEPTH"], model_dim, knobs["WINDOW_PATTERN"])
    windows = window_sizes(knobs["DEPTH"], knobs["WINDOW_PATTERN"])
    return {
        "model_dim": float(model_dim),
        "num_heads": float(num_heads),
        "total_params": float(total_params),
        "flops_counted_params": float(counted),
        "flops_per_token": float(fpt),
        "long_frac": sum(1 for w in windows if w == MAX_SEQ_LEN) / len(windows),
        "grad_accum_steps": float(
            knobs["TOTAL_BATCH_SIZE"] / (knobs["DEVICE_BATCH_SIZE"] * MAX_SEQ_LEN)
        ),
    }


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def divisibility_ok(knobs: dict) -> bool:
    """train.py:496 ``assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0``.

    Note this makes ``DEVICE_BATCH_SIZE=96`` unsatisfiable for every
    ``TOTAL_BATCH_SIZE`` in the schema, because all three choices are powers of
    two. That removes 1/4 of the raw choice grid; it is a property of the real
    trainer, not a modelling decision.
    """

    tokens_per_fwdbwd = knobs["DEVICE_BATCH_SIZE"] * MAX_SEQ_LEN
    return knobs["TOTAL_BATCH_SIZE"] % tokens_per_fwdbwd == 0


def check_hard_constraints(knobs: dict) -> None:
    """Raise :class:`ProposalError` for configs the trainer would reject.

    Only rules that are literally asserted by ``train.py`` are enforced here.

    **There is intentionally no VRAM pre-filter.** The previous analytic VRAM
    estimate under-predicted measured ``peak_vram_mb`` by 5-31x (its activation
    term was ~500x too small in slope), so as a gate it admitted configs that
    OOM while claiming to bound the feasible set. A wrong gate permanently
    distorts the action space, whereas a real OOM is cheap (~20s) and is
    reported as an evaluation failure with ``failure_kind="oom"``. Let reality
    decide, and read the OOM rate off the results instead of trusting a model.
    """

    if not divisibility_ok(knobs):
        tokens_per_fwdbwd = knobs["DEVICE_BATCH_SIZE"] * MAX_SEQ_LEN
        raise ProposalError(
            f"TOTAL_BATCH_SIZE ({knobs['TOTAL_BATCH_SIZE']}) is not divisible by "
            f"DEVICE_BATCH_SIZE*{MAX_SEQ_LEN} ({tokens_per_fwdbwd}); "
            "train.py asserts this at startup"
        )


# ---------------------------------------------------------------------------
# Validation and canonicalisation
# ---------------------------------------------------------------------------


def _same_choice(choice: Any, raw: Any) -> bool:
    if isinstance(choice, str):
        return isinstance(raw, str) and choice.upper() == raw.strip().upper()
    try:
        return abs(float(choice) - float(raw)) < 1e-9
    except (TypeError, ValueError):
        return False


def validate(knobs: dict, schema: dict | None = None) -> dict:
    """Range/type/choice check against the real schema; returns a clean dict.

    Does *not* check :func:`check_hard_constraints` -- callers that need the
    trainer's own asserts must call it too. :func:`admit` does both.
    """

    schema = schema if schema is not None else load_schema()
    clean: dict[str, Any] = {}
    for name, spec in schema.items():
        if name not in knobs:
            raise ProposalError(f"missing knob {name}")
        raw = knobs[name]
        kind = spec["type"]
        if kind == "choice":
            match = next((c for c in spec["choices"] if _same_choice(c, raw)), None)
            if match is None:
                raise ProposalError(f"{name}={raw!r} not in {spec['choices']}")
            clean[name] = match
        elif kind in ("int", "integer"):
            try:
                value: Any = int(round(float(raw)))
            except (TypeError, ValueError):
                raise ProposalError(f"{name}={raw!r} is not an int") from None
            if not spec["min"] <= value <= spec["max"]:
                raise ProposalError(
                    f"{name}={value} outside [{spec['min']}, {spec['max']}]"
                )
            clean[name] = value
        else:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                raise ProposalError(f"{name}={raw!r} is not a float") from None
            if not spec["min"] <= value <= spec["max"]:
                raise ProposalError(
                    f"{name}={value} outside [{spec['min']}, {spec['max']}]"
                )
            clean[name] = value
    return clean


def canonical_key(knobs: dict, time_budget: float = DEFAULT_TIME_BUDGET) -> str:
    """Stable identity of a (config, budget) pair.

    Used for two different jobs, both of which need the *same* notion of
    "already seen":

    * reservoir de-duplication inside and across RL rounds
      (``ReservoirSpec.deduplication_key``);
    * the evaluation cache -- re-running a seen config costs a full time
      budget and buys nothing, since the measured within-config std is
      ~0.0013 bpb.

    Floats are rendered with ``%.6g``, far finer than any behavioural
    difference but coarse enough to be repr-independent.
    """

    parts = []
    for name in KNOB_ORDER:
        value = knobs[name]
        if isinstance(value, str):
            parts.append(f"{name}={value.strip().upper()}")
        elif isinstance(value, bool):  # not expected, but keep it unambiguous
            parts.append(f"{name}={int(value)}")
        elif float(value).is_integer() and abs(float(value)) < 2 ** 53:
            parts.append(f"{name}={int(value)}")
        else:
            parts.append(f"{name}={float(value):.6g}")
    parts.append(f"TIME_BUDGET={float(time_budget):.6g}")
    return "|".join(parts)


def admit(knobs: dict, schema: dict | None = None) -> dict:
    """Validate and constraint-check in one step. Returns the clean config."""

    clean = validate(knobs, schema)
    check_hard_constraints(clean)
    return clean


def with_defaults(free: Iterable[str], proposed: dict, pinned: dict | None = None) -> dict:
    """Assemble a full config: defaults <- pinned <- proposed(free knobs only)."""

    knobs = dict(DEFAULTS)
    if pinned:
        knobs.update(pinned)
    free = list(free)
    knobs.update({name: proposed[name] for name in free if name in proposed})
    return knobs


def describe_config(knobs: dict) -> str:
    """One-line human/LLM-readable rendering, used in env feedback."""

    geo = geometry(knobs)
    return (
        f"DEPTH={knobs['DEPTH']} ASPECT_RATIO={knobs['ASPECT_RATIO']} "
        f"HEAD_DIM={knobs['HEAD_DIM']} WINDOW_PATTERN={knobs['WINDOW_PATTERN']} "
        f"-> model_dim={int(geo['model_dim'])} params={geo['total_params']/1e6:.1f}M; "
        f"TOTAL_BATCH_SIZE={knobs['TOTAL_BATCH_SIZE']} "
        f"DEVICE_BATCH_SIZE={knobs['DEVICE_BATCH_SIZE']} "
        f"(grad_accum={int(geo['grad_accum_steps'])}); "
        f"lr(emb/unemb/matrix/scalar)="
        f"{knobs['EMBEDDING_LR']:.4g}/{knobs['UNEMBEDDING_LR']:.4g}/"
        f"{knobs['MATRIX_LR']:.4g}/{knobs['SCALAR_LR']:.4g}; "
        f"wd={knobs['WEIGHT_DECAY']:.4g} warmup={knobs['WARMUP_RATIO']:.4g} "
        f"warmdown={knobs['WARMDOWN_RATIO']:.4g} final_lr_frac={knobs['FINAL_LR_FRAC']:.4g}"
    )


__all__ = [
    "DEFAULTS",
    "DEFAULT_TIME_BUDGET",
    "KNOB_ORDER",
    "MAX_SEQ_LEN",
    "ProposalError",
    "SCHEMA_PATH",
    "VOCAB_SIZE",
    "admit",
    "canonical_key",
    "check_hard_constraints",
    "count_params",
    "derive_shape",
    "describe_config",
    "divisibility_ok",
    "flops_per_token",
    "geometry",
    "knob_names",
    "load_schema",
    "schema_version",
    "validate",
    "window_sizes",
    "with_defaults",
]
