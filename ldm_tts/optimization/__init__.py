"""Surrogate representation and acquisition-selection interface."""

from ldm_tts.optimization.records import (
    AcquisitionSelector,
    BOObservation,
    BOPrediction,
    BOSelectionResult,
    FeatureEncoder,
    FeatureVector,
    SurrogateEncoder,
    SurrogateVector,
)

__all__ = [
    "AcquisitionSelector",
    "BOObservation",
    "BOPrediction",
    "BOSelectionResult",
    "FeatureEncoder",
    "FeatureVector",
    "SurrogateEncoder",
    "SurrogateVector",
]
