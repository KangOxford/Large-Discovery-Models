"""Dependency-check adapter for mutation-effect prediction."""

from __future__ import annotations

from typing import Any

from ldm_tts.registration.dependencies import DependencyCheck, plan_check_context
from tasks.ai4bio_mutation_effect_prediction.core.dependencies import (
    check_task_dependencies,
)


def check_dependencies(
    plan: dict[str, Any], *, include_optional: bool = True
) -> list[DependencyCheck]:
    task, args, env, cwd, mode = plan_check_context(plan)
    return check_task_dependencies(
        task, args, env, cwd, mode=mode, include_optional=include_optional
    )
