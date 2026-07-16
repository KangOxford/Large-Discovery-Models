"""Configuration for the standalone reservoir LDM prototype."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ReservoirLDMConfig:
    """Parameters for the new LLM-reservoir acquisition loop.

    The defaults are deliberately small enough for smoke tests. A real BO run
    can increase ``per_strategy_budget`` while keeping the number of LLM
    strategies fixed at five.
    """

    n_strategies: int = 5
    per_strategy_budget: int = 200
    sample_timeout_s: float = 5.0
    bias_weight: float = 0.05

    # Inside each strategy pool, choose that strategy's representative x.
    # Supported: "combined" = acquisition + bias_weight*bias, "acq", "bias".
    pool_score: str = "combined"

    # Across the K representatives, choose the final BO query point.
    # Supported: "softmax" or "argmax".
    selection_mode: str = "softmax"
    selection_score: str = "acq"
    softmax_eta: float = 1.0
    rng_seed: Optional[int] = None

    # Fallback if the LLM provides fewer than n_strategies valid strategies.
    fallback_radius_start: int = 1
    fallback_mut_pr: float = 0.5

    # LLM planner settings.
    llm_temperature: float = 0.25
    llm_call_timeout_s: int = 30
    max_retries: int = 3
    history_max_in_prompt: int = 80

    atoms_whitelist: tuple[str, ...] = (
        "LatinHyperCubeSampling", "NeighborSampling", "LocalSearch", "Or",
        "MaxCysteine", "MaxHydrophobicRun", "MaxAromatic",
        "NetChargeRange", "NoNGlycosylation", "BiasSum",
    )

    @classmethod
    def from_dict(cls, values: dict) -> "ReservoirLDMConfig":
        valid = set(cls.__dataclass_fields__)
        unknown = set(values) - valid
        if unknown:
            raise ValueError(f"Unknown ReservoirLDMConfig keys: {sorted(unknown)}")
        return cls(**values)
