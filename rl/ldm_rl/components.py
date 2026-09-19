"""Shared RL adapter bundle assembled by each task's ``build_rl_components``.

This is a dependency-light leaf module: it imports only ``ldm_tts.contracts``
and the stdlib, so a task package can lazily import it (``from
ldm_rl.components import EnvComponents``) inside its RL adapter without pulling
in the rest of ``ldm_rl`` or any Slime/torch dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ldm_tts.contracts import CandidateDomainAdapter, CandidateEvaluator, LDMTaskSpec


@dataclass(frozen=True)
class EnvComponents:
    """Task adapter bundle assembled by one task's ``build_rl_components``."""

    task_spec: LDMTaskSpec
    domain: CandidateDomainAdapter
    evaluator: CandidateEvaluator
    parse_action: Any = None  # callable(text) -> list[payload] | None (spec-declared)
    context: dict[str, Any] | None = None
    selector: Any = None  # AcquisitionSelector | None
    surrogate_encoder: Any = None  # SurrogateEncoder | None


__all__ = ["EnvComponents"]
