"""Configuration for the case2 acquisition-tilted LDM path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from strbo_v1.gp import GPConfig


TiltedMethod = Literal[
    "m1_direct_llm_sir",
    "m1_stratified_direct_llm_sir",
    "m1_stratified_direct_llm_oversample_sir",
    "m1_stratified_direct_llm_only",
    "m1_llm_one_step",
    "m1_llm_seed_analog_oversample_sir",
]

VALID_METHODS = {
    "m1_direct_llm_sir",
    "m1_stratified_direct_llm_sir",
    "m1_stratified_direct_llm_oversample_sir",
    "m1_stratified_direct_llm_only",
    "m1_llm_one_step",
    "m1_llm_seed_analog_oversample_sir",
}

InitStrategy = Literal["seed_smiles", "llm_cold_start"]
VALID_INIT_STRATEGIES = {"seed_smiles", "llm_cold_start"}


@dataclass
class TiltedLDMCase2Config:
    """Shared knobs for M1 tilted case2 searches."""

    method: TiltedMethod
    init_size: int = 5
    init_strategy: InitStrategy = "seed_smiles"
    budget: int = 80
    batch_size: int = 1
    smiles_max_len: int = 80
    max_candidates_per_round: int = 256
    max_empty_reservoir_rounds: int = 10
    allow_early_stop: bool = True
    minimize: tuple[bool, bool] = (True, False)
    ref_point: tuple[float, float] = (0.0, 5.0)
    gp_config: GPConfig = field(default_factory=GPConfig)
    acquisition: Literal["ehvi", "mean"] = "ehvi"
    acquisition_weights: tuple[float, float] = (0.5, 0.5)
    ehvi_n_samples: int = 128
    alpha_base_measure: float = 1.0
    eta_ehvi_tilt: float = 3.0
    z_clip: float = 5.0
    eps: float = 1e-12
    m1_k_direct_llm: int = 128
    m1_q0_smoothing: float = 0.0
    m1_analog_n_llm_seeds: int = 8
    m1_analog_k_total: int = 1024
    llm_max_retries: int = 2
    llm_retry_wait_seconds: float = 0.0
    trajectory_dir: Optional[str] = None
    resume_from_trajectory: bool = False
    seed: int = 0
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.method not in VALID_METHODS:
            raise ValueError(f"method must be one of {sorted(VALID_METHODS)}, got {self.method!r}")
        if self.init_strategy not in VALID_INIT_STRATEGIES:
            raise ValueError(
                f"init_strategy must be one of {sorted(VALID_INIT_STRATEGIES)}, "
                f"got {self.init_strategy!r}"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.budget < self.init_size:
            raise ValueError(
                f"budget must be >= init_size, got budget={self.budget}, init_size={self.init_size}"
            )
        if self.eta_ehvi_tilt < 0:
            raise ValueError(f"eta_ehvi_tilt must be >= 0, got {self.eta_ehvi_tilt}")
        if self.alpha_base_measure < 0:
            raise ValueError(
                f"alpha_base_measure must be >= 0, got {self.alpha_base_measure}"
            )
        if self.m1_q0_smoothing < 0:
            raise ValueError(f"m1_q0_smoothing must be >= 0, got {self.m1_q0_smoothing}")
        if self.m1_analog_n_llm_seeds < 1:
            raise ValueError("m1_analog_n_llm_seeds must be >= 1")
        if self.m1_analog_k_total < 1:
            raise ValueError("m1_analog_k_total must be >= 1")
        if len(self.minimize) != 2:
            raise ValueError(f"minimize must have length 2, got {len(self.minimize)}")
        if len(self.ref_point) != 2:
            raise ValueError(f"ref_point must have length 2, got {len(self.ref_point)}")
        if self.acquisition not in {"ehvi", "mean"}:
            raise ValueError("acquisition must be one of ['ehvi', 'mean']")
        if len(self.acquisition_weights) != 2:
            raise ValueError("acquisition_weights must contain one weight per objective")
        if any(weight < 0 for weight in self.acquisition_weights):
            raise ValueError("acquisition_weights must be non-negative")
        if sum(self.acquisition_weights) <= 0:
            raise ValueError("at least one acquisition weight must be positive")
        if self.max_candidates_per_round < 1:
            raise ValueError("max_candidates_per_round must be >= 1")
        if self.max_empty_reservoir_rounds < 1:
            raise ValueError("max_empty_reservoir_rounds must be >= 1")
        if self.ehvi_n_samples < 1:
            raise ValueError("ehvi_n_samples must be >= 1")
        if self.llm_max_retries < 0:
            raise ValueError("llm_max_retries must be >= 0")
        if self.llm_retry_wait_seconds < 0:
            raise ValueError("llm_retry_wait_seconds must be >= 0")
