"""Task discovery and registration interface."""

from ldm_tts.registration.registry import (
    TaskDefinition,
    TaskRegistrationError,
    discover_task_definitions,
    get_task_definition,
)

__all__ = [
    "TaskDefinition",
    "TaskRegistrationError",
    "discover_task_definitions",
    "get_task_definition",
]
