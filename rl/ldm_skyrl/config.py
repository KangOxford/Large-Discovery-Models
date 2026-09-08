"""LDM-specific configuration, and the composition with SkyRL's own config.

Two shapes on purpose.  ``LDMGeneratorSettings`` is a plain dataclass with no
SkyRL import, so it can be constructed, validated, and unit-tested on a login
node.  ``build_config`` is the part that needs SkyRL installed, and it fails
through the phase gate rather than through a bare ``ImportError``.

The knob defaults below are not neutral: each one is set against something the
slime line measured.  Where a default differs from SkyRL's own, the reason is on
the line.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ._deps import gate, require_skyrl
from .reward import VINA_SATURATION_WORKERS


@dataclass
class LDMGeneratorSettings:
    """Everything the LDM generator needs that SkyRL does not already model."""

    episode_files: list[str] = field(default_factory=list)
    max_concurrent_evaluations: int = VINA_SATURATION_WORKERS
    trajectories_per_second: float | None = None
    # Logged every step so the async claim in phase item C4 can be checked
    # rather than assumed.
    log_env_time_share: bool = True

    def __post_init__(self) -> None:
        if self.max_concurrent_evaluations < 1:
            raise ValueError(
                "max_concurrent_evaluations must be at least 1, got "
                f"{self.max_concurrent_evaluations}"
            )
        if self.trajectories_per_second is not None and self.trajectories_per_second <= 0:
            raise ValueError(
                "trajectories_per_second must be positive when set, got "
                f"{self.trajectories_per_second}"
            )


# Algorithm defaults, each justified against a measurement on the slime line.
# These are proposals for phase C, not results: the recipe's ablation ranks them
# on a different domain, and this line has not run them yet.
ALGORITHM_DEFAULTS: dict[str, Any] = {
    # Removes the division that turns a zero-variance group into 0/0. This is
    # the only one of the four that costs nothing: it deletes an operation.
    "trainer.algorithm.grpo_norm_by_std": False,
    # Masks zero-variance groups instead of letting them contribute a
    # meaningless gradient. It does not manufacture signal; filling the
    # mini-batch back up still needs extra rollouts.
    "trainer.algorithm.zero_variance_filter": True,
    "trainer.algorithm.zero_variance_filter_tol": 1e-6,
    # Top two rows of the recipe's own ablation on its own benchmark.
    "trainer.algorithm.policy_loss_type": "dppo",
    "trainer.algorithm.loss_reduction": "prompt_mean",
    "trainer.algorithm.advantage_estimator": "grpo",
}

# Fixed by this cluster, not by preference. Driver 565 means CUDA 12.7, so the
# stack has to stay on cu128/cu129 wheels; a cu130 build leaves
# torch.cuda.is_available() returning False.
CLUSTER_PINS: dict[str, str] = {
    "skyrl_commit": "b8a5caaa",
    "skyrl_release": "0.3.0",
    "vllm": "0.23.0+cu129",
    "cuda_wheel_range": "cu128 / cu129",
    "forbidden_cuda_wheel": "cu130",
}


def build_config(overrides: dict[str, Any] | None = None) -> Any:
    """Compose SkyRL's train config with the LDM defaults.

    Raises the A1 gate when SkyRL is absent, so the message names the phase item
    that installs it instead of saying only that a module was not found.
    """
    require_skyrl("A1")
    try:
        from skyrl.train.config import SkyRLTrainConfig
    except ImportError as exc:  # pragma: no cover - pin drift, not absence
        raise gate(
            "A1",
            "skyrl imports but skyrl.train.config.SkyRLTrainConfig does not; the "
            f"pin has moved away from {CLUSTER_PINS['skyrl_commit']}.",
        ) from exc

    merged = dict(ALGORITHM_DEFAULTS)
    merged.update(overrides or {})
    return SkyRLTrainConfig.from_cli_overrides(
        [f"{k}={v}" for k, v in merged.items()]
    )


def settings_as_dict(settings: LDMGeneratorSettings) -> dict[str, Any]:
    return asdict(settings)


__all__ = [
    "ALGORITHM_DEFAULTS",
    "CLUSTER_PINS",
    "LDMGeneratorSettings",
    "build_config",
    "settings_as_dict",
]
