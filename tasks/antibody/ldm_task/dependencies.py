"""Dependency-check adapter for the antibody task."""

from __future__ import annotations

from typing import Any

from ldm_tts.registration.dependencies import (
    DependencyCheck,
    plan_check_context,
)
from tasks.antibody.core.dependencies import check_antibody


def check_dependencies(
    plan: dict[str, Any], *, include_optional: bool = True
) -> list[DependencyCheck]:
    del include_optional
    _, args, env, cwd, mode = plan_check_context(plan)
    return check_antibody(args, env, cwd, mode=mode)
