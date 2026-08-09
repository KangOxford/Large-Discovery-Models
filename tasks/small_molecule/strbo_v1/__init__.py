"""Public re-exports for the LDM-TTS small-molecule adapter.

Exports are resolved lazily so lightweight imports, such as
``strbo_v1.llm_advisor`` or ``strbo_v1.rng``, do not immediately require the
full GP / Torch / docking stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_REF": ("strbo_v1.scorer", "DEFAULT_REF"),
    "GPConfig": ("strbo_v1.gp", "GPConfig"),
    "GPSurrogate": ("strbo_v1.gp", "GPSurrogate"),
    "NNScorer": ("strbo_v1.objective_nn", "NNScorer"),
    "NNScorerConfig": ("strbo_v1.objective_nn", "NNScorerConfig"),
    "RNG": ("strbo_v1.rng", "RNG"),
    "ReasynConfig": ("strbo_v1.analog", "ReasynConfig"),
    "Scorer": ("strbo_v1.scorer", "Scorer"),
    "Scorers": ("strbo_v1.scorer", "Scorers"),
    "VinaScorer": ("strbo_v1.objective_vina", "VinaScorer"),
    "VinaScorerConfig": ("strbo_v1.objective_vina", "VinaScorerConfig"),
    "as_rng": ("strbo_v1.rng", "as_rng"),
    "as_scorer_tuple": ("strbo_v1.scorer", "as_scorer_tuple"),
    "chebyshev_scalarize": ("strbo_v1.acquisition", "chebyshev_scalarize"),
    "confidence_bound": ("strbo_v1.acquisition", "confidence_bound"),
    "dominates": ("strbo_v1.acquisition", "dominates"),
    "expected_hypervolume_improvement": (
        "strbo_v1.acquisition",
        "expected_hypervolume_improvement",
    ),
    "expected_improvement": ("strbo_v1.acquisition", "expected_improvement"),
    "generate_analogs": ("strbo_v1.analog", "generate_analogs"),
    "hypervolume": ("strbo_v1.acquisition", "hypervolume"),
    "pareto_front": ("strbo_v1.acquisition", "pareto_front"),
    "probability_of_improvement": ("strbo_v1.acquisition", "probability_of_improvement"),
    "register_ref": ("strbo_v1.scorer", "register_ref"),
    "resolve_ref_point": ("strbo_v1.scorer", "resolve_ref_point"),
    "sample_simplex_weights": ("strbo_v1.acquisition", "sample_simplex_weights"),
}

__all__ = [
    "DEFAULT_REF",
    "GPConfig",
    "GPSurrogate",
    "NNScorer",
    "NNScorerConfig",
    "RNG",
    "ReasynConfig",
    "Scorer",
    "Scorers",
    "VinaScorer",
    "VinaScorerConfig",
    "as_rng",
    "as_scorer_tuple",
    "chebyshev_scalarize",
    "confidence_bound",
    "dominates",
    "expected_hypervolume_improvement",
    "expected_improvement",
    "generate_analogs",
    "hypervolume",
    "pareto_front",
    "probability_of_improvement",
    "register_ref",
    "resolve_ref_point",
    "sample_simplex_weights",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
