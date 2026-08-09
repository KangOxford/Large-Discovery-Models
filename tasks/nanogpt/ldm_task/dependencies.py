"""Dependency-check adapter for the nanoGPT task."""

from __future__ import annotations

from typing import Any

from ldm_tts.dependency_checks import (
    DependencyCheck,
    check_nanogpt,
    plan_check_context,
)


def check_dependencies(
    plan: dict[str, Any], *, include_optional: bool = True
) -> list[DependencyCheck]:
    _, args, env, cwd, mode = plan_check_context(plan)
    return check_nanogpt(
        args,
        env,
        cwd,
        mode=mode,
        include_optional=include_optional,
    )
