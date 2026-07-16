"""ReaSyn configuration pool.

Maps the LLM's ``generator_hint`` (a coarse three-way classification)
to a concrete :class:`strbo_v1.analog.ReasynConfig`. Lets the
orchestrator run multiple ReaSyn presets in parallel and pick one
per LLM ``analog`` block based on the hint.

Typical usage::

    pool = ReasynConfigPool.from_env()
    cfg = pick_reasyn_config(pool, "conservative", reasyn_config_override=None)
    analogues = generate_analogs(seeds, cfg)

If the LLM provides a ``reasyn_config_override`` block with a single
field (e.g. ``{"search_width": 12}``), the picked config is copied
and that field is replaced. Bounds are enforced; out-of-range values
fall back to the preset default (logged as a warning).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from strbo_v1.analog import ReasynConfig

from strbo_v1.llm_advisor.blocks import GeneratorHint

LOGGER = logging.getLogger(__name__)


# ReaSyn parameter bounds (kept in sync with the JSON schema in
# ``schema.py``). Used to clamp override values.
_OVERRIDE_BOUNDS: Dict[str, tuple] = {
    "search_width": (1, 64),
    "num_cycles": (1, 32),
    "num_editflow_samples": (1, 500),
    "time_limit": (10, 3600),
}


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------


@dataclass
class ReasynConfigPool:
    """Mapping of preset names to ReaSyn configs.

    Three named presets by default — ``conservative``,
    ``aggressive``, ``scaffold_hop`` — matching the LLM's
    ``generator_hint`` vocabulary. The ``default`` slot is the
    fallback when the hint is unknown.
    """

    conservative: ReasynConfig
    aggressive: ReasynConfig
    scaffold_hop: ReasynConfig
    default: ReasynConfig

    def all_presets(self) -> Dict[str, ReasynConfig]:
        return {
            "conservative": self.conservative,
            "aggressive": self.aggressive,
            "scaffold_hop": self.scaffold_hop,
            "default": self.default,
        }

    @classmethod
    def from_env(cls) -> "ReasynConfigPool":
        """Build a pool with three presets.

        The presets share the same model_path/reasyn_repo but differ
        in their sampling parameters. Callers that need different
        presets should construct the pool by hand.
        """
        # We do not know model_path / reasyn_repo at the pool level;
        # they must be supplied at generate_analogs time. The pool
        # here only sets the *sampling* parameters. We use
        # placeholder model_path; callers that need a real path
        # should override individual configs before calling
        # generate_analogs.
        shared: Dict[str, Any] = dict(
            model_path=("ar.ckpt", "eb.ckpt"),  # caller must override
            num_editflow_steps=100,
            num_workers_per_gpu=1,
        )
        conservative = ReasynConfig(
            **shared, search_width=4, num_cycles=2, num_editflow_samples=10,
        )
        aggressive = ReasynConfig(
            **shared, search_width=10, num_cycles=8,
            num_editflow_samples=40, time_limit=300,
        )
        scaffold_hop = ReasynConfig(
            **shared, search_width=8, num_cycles=6, num_editflow_samples=30,
        )
        default = ReasynConfig(
            **shared, search_width=6, num_cycles=4,
            num_editflow_samples=20, time_limit=120,
        )
        return cls(
            conservative=conservative,
            aggressive=aggressive,
            scaffold_hop=scaffold_hop,
            default=default,
        )


# ---------------------------------------------------------------------------
# Picker
# ---------------------------------------------------------------------------


def _apply_override(
    base: ReasynConfig, override: Optional[Dict[str, Any]],
) -> ReasynConfig:
    """Return a copy of ``base`` with ``override`` applied (clamped)."""
    if not override:
        return base
    cfg = copy.deepcopy(base)
    for k, v in override.items():
        if k not in _OVERRIDE_BOUNDS:
            LOGGER.warning("reasyn_config_override: unknown key %r; ignored", k)
            continue
        lo, hi = _OVERRIDE_BOUNDS[k]
        try:
            iv = int(v)
        except (TypeError, ValueError):
            LOGGER.warning("reasyn_config_override[%r]: not an int (%r); ignored", k, v)
            continue
        if iv < lo or iv > hi:
            LOGGER.warning(
                "reasyn_config_override[%r]=%d out of range [%d, %d]; clamped to %d",
                k, iv, lo, hi, max(lo, min(hi, iv)),
            )
            iv = max(lo, min(hi, iv))
        setattr(cfg, k, iv)
    return cfg


def pick_reasyn_config(
    pool: ReasynConfigPool,
    hint: Optional[GeneratorHint],
    override: Optional[Dict[str, Any]] = None,
) -> ReasynConfig:
    """Pick a :class:`ReasynConfig` from ``pool`` and apply ``override``.

    Unknown hints fall back to ``pool.default``. The override (if any)
    is clamped to the documented bounds and applied last.
    """
    presets = pool.all_presets()
    if hint is None:
        base = presets["default"]
    else:
        base = presets.get(hint, presets["default"])
    return _apply_override(base, override)


__all__ = ["ReasynConfigPool", "pick_reasyn_config"]
