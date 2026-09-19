"""LDM <-> RL environment glue.

This package turns one LDM task campaign into an RL environment:

- ``env.LDMEnv``      : framework-agnostic reset/step environment. One RL step
  is one LDM engine round: the policy proposes candidates, the environment
  admits, deduplicates, evaluates them and returns metric improvement as reward.
- ``factories``       : build an ``LDMEnv`` from a registered task id and mode
  (mock / real), reusing the task's own ``describe_ldm_task`` and core adapters.
- ``bridge``          : Slime integration (custom generate / reward functions).
- ``episodes``        : serializable episode specifications used as Slime
  prompt-data rows.

The environment itself never imports Slime or torch; only ``bridge`` touches
Slime, and it does so lazily inside the rollout worker.
"""

from ldm_rl.episodes import EpisodeSpec
from ldm_rl.env import EnvConfig, EnvStep, EpisodeResult, LDMEnv

__all__ = [
    "EnvConfig",
    "EnvStep",
    "EpisodeResult",
    "EpisodeSpec",
    "LDMEnv",
]
