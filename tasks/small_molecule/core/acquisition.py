"""Compatibility facade for the shared LDM-TTS acquisition module.

Acquisition behavior is implemented in :mod:`ldm_tts.optimization.acquisition`; this module
keeps the historical ``tasks.small_molecule.core.acquisition`` import path for callers and saved
experiments.
"""

from ldm_tts.optimization.acquisition import (
    AcquisitionConfig,
    AcquisitionFunction,
    PosteriorAcquisition,
    chebyshev_scalarize,
    confidence_bound,
    dominates,
    expected_hypervolume_improvement,
    expected_improvement,
    hypervolume,
    make_acquisition,
    pareto_front,
    posterior_mean_score,
    probability_of_improvement,
    sample_simplex_weights,
)

__all__ = [
    "AcquisitionConfig",
    "AcquisitionFunction",
    "PosteriorAcquisition",
    "chebyshev_scalarize",
    "confidence_bound",
    "dominates",
    "expected_hypervolume_improvement",
    "expected_improvement",
    "hypervolume",
    "make_acquisition",
    "pareto_front",
    "posterior_mean_score",
    "probability_of_improvement",
    "sample_simplex_weights",
]
