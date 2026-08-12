"""Manifest hook for protein inverse-folding dependency checks."""

from __future__ import annotations

from typing import Any

from ldm_tts.registration.dependencies import DependencyCheck
from tasks.protein_inverse_folding.core.dependencies import (
    check_dependencies as check_task_dependencies,
)


def check_dependencies(
    plan: dict[str, Any], *, include_optional: bool = True
) -> list[DependencyCheck]:
    return check_task_dependencies(plan, include_optional=include_optional)
