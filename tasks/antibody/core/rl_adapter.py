"""RL environment adapter placeholder for the antibody task.

The antibody task's campaign entry point still routes through the legacy
``ldm_light.ldm_acq.run_one`` path rather than the shared ``ldm_tts`` engine, so
there is no ``CampaignRecipe``-shaped component bundle for the RL environment
to reuse yet. Until the task is migrated to the shared engine, ``build_env``
fails fast with an explicit "not wired" error instead of a generic missing
factory.
"""

from __future__ import annotations

from typing import Any


def build_rl_components(mode: str = "mock", **kwargs: Any) -> Any:
    raise NotImplementedError(
        "antibody has no RL environment adapter yet; its campaign still runs "
        "through ldm_light.ldm_acq.run_one and must be migrated to the shared "
        "ldm_tts engine before an RL environment can reuse its components."
    )
