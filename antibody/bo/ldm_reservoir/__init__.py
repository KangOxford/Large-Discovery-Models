"""Reservoir-style LDM prototype for AntBO.

This package is intentionally separate from ``bo.ldm`` so the existing
implementation remains untouched. It reuses the original DSL atoms and
acquisition executors, but implements a new selection loop:

LLM context -> K search strategies z -> one best candidate per strategy ->
softmax/argmax selection by acquisition.
"""
from .config import ReservoirLDMConfig
from .planner import ReservoirPlan, ReservoirPlanner
from .session import ReservoirAcquisitionSession

__all__ = [
    "ReservoirLDMConfig",
    "ReservoirPlan",
    "ReservoirPlanner",
    "ReservoirAcquisitionSession",
]
