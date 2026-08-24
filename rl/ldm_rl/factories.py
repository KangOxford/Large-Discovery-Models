"""Build an ``LDMEnv`` from a registered LDM task.

Each registered task owns its RL adapter as ``tasks.<task_id>.core.rl_adapter``
with a ``build_rl_components(mode, **kwargs) -> EnvComponents`` entry point.
This module discovers registered tasks through ``ldm_tts.registration`` and
dispatches to the matching adapter, so the environment always reuses the same
task-owned adapters the task passes to ``CampaignRecipe``.

The environment core stays task-neutral; only the adapter bundle differs per
task. Task adapters are imported lazily so building one task never pulls in
another task's dependencies.
"""

from __future__ import annotations

import importlib
from typing import Any

from ldm_tts.contracts import CandidateDomainAdapter, CandidateEvaluator, LDMTaskSpec

from ldm_rl.components import EnvComponents
from ldm_rl.env import EnvConfig, LDMEnv


def build_env(
    task_id: str,
    *,
    mode: str = "mock",
    config: EnvConfig | None = None,
    context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> LDMEnv:
    """Assemble an environment for a registered task.

    ``kwargs`` flow to the task's RL adapter (reservoir size, real-evaluator
    paths, ...). Use ``config`` to fix the episode lifecycle/reward policy;
    otherwise a default 8-round episode is created. ``context`` overlays the
    adapter's default episode context (e.g. assay or case lists).
    """

    if mode not in {"mock", "real"}:
        raise ValueError("mode must be 'mock' or 'real'")
    build_components = _resolve_adapter(task_id)
    # ``config`` is the single source of truth for the episode's reservoir size,
    # so thread it into the adapter's kwargs (the adapter uses it for the task
    # spec's reservoir max_size and its text parser's expected_count).
    if config is not None:
        kwargs.setdefault("reservoir_size", config.reservoir_size)
        kwargs.setdefault("evaluations_per_round", config.evaluations_per_round)
    components = build_components(mode=mode, **kwargs)
    if config is None:
        config = EnvConfig(
            iterations=8,
            reservoir_size=kwargs.get("reservoir_size", 2),
            evaluations_per_round=kwargs.get("evaluations_per_round", 1),
        )
    env_context = dict(components.context or {})
    if context:
        env_context.update(context)
    return LDMEnv(
        task_spec=components.task_spec,
        domain=components.domain,
        evaluator=components.evaluator,
        config=config,
        parse_action=components.parse_action,
        context=env_context or None,
        selector=components.selector,
        surrogate_encoder=components.surrogate_encoder,
    )


def _resolve_adapter(task_id: str) -> Any:
    """Resolve a registered task's ``build_rl_components`` entry point."""

    from ldm_tts.registration.registry import get_task_definition

    # Raises KeyError with the full registered list for an unknown task id.
    definition = get_task_definition(task_id)
    module_path = f"tasks.{definition.task_id}.core.rl_adapter"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"task {task_id!r} is registered but has no RL adapter module "
            f"{module_path!r}"
        ) from exc
    adapter = getattr(module, "build_rl_components", None)
    if adapter is None:
        raise RuntimeError(
            f"task {task_id!r} RL adapter module {module_path!r} does not define "
            "build_rl_components(mode, **kwargs)"
        )
    return adapter


__all__ = [
    "EnvComponents",
    "build_env",
    # Re-exported so callers can reference the adapter component types without
    # importing the contracts module directly.
    "CandidateDomainAdapter",
    "CandidateEvaluator",
    "LDMTaskSpec",
]
