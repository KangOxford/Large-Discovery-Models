"""core/ldm/config.py — single source of truth for all LDM parameters."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DSLConfig:
    """All LDM / Orchestrator / DSL parameters in one dataclass.

    Construct via :meth:`from_yaml` from a dict (typically the ``llm:``
    section of ``resources/default_config.yaml``).
    """

    # LLM phase control (replaces the old single llm_orchestrator_enabled)
    llm_init_enabled: bool = True
    llm_loop_enabled: bool = True
    llm_temperature: float = 0.25
    max_retries: int = 3
    llm_call_timeout_s: int = 30
    llm_decisions_log: str = "runs/llm_decisions/exp.json"
    history_max_in_prompt: int = 100

    # Bias contribution
    bias_weight: float = 0.05

    # DSL sampling
    acq_n_candidates: int = 5000
    sample_timeout_s: float = 5.0
    init_pool_size: int = 100000
    max_nesting_depth: int = 8

    # Interactive acquisition session
    batch_size: int = 1
    acq_search_budget: int = 600
    acq_max_rounds: int = 3
    num_llm_review: int = 10

    # Sandbox / atoms
    atoms_whitelist: tuple = (
        "LatinHyperCubeSampling", "NeighborSampling", "LocalSearch", "Or",
        "MaxCysteine", "MaxHydrophobicRun", "MaxAromatic",
        "NetChargeRange", "NoNGlycosylation", "BiasSum",
    )

    # Fallback
    fallback_strategy: str = "original_antbo"

    # Strategy prompt (controls the [5] section of system.txt)
    strategy: str = "ldm-default"

    @classmethod
    def from_yaml(cls, yaml_dict: dict) -> "DSLConfig":
        """Build from a YAML dict; reject unknown keys."""
        valid_keys = set(cls.__dataclass_fields__)
        unknown = set(yaml_dict) - valid_keys
        if unknown:
            raise ValueError(f"Unknown LDM config keys: {sorted(unknown)}")
        return cls(**yaml_dict)
