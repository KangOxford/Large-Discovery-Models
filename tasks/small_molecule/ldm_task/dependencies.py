"""Dependency-check adapter for the small-molecule task."""

from __future__ import annotations

from typing import Any

from ldm_tts.registration.dependencies import (
    DependencyCheck,
    plan_check_context,
)
from tasks.small_molecule.core.dependencies import check_small_molecule


def check_dependencies(
    plan: dict[str, Any], *, include_optional: bool = True
) -> list[DependencyCheck]:
    _, args, env, cwd, mode = plan_check_context(plan)
    return check_small_molecule(
        args,
        env,
        cwd,
        mode=mode,
        include_optional=include_optional,
    )
