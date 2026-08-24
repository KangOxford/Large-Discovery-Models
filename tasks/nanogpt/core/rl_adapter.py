"""RL environment adapter placeholder for the nanoGPT task.

nanoGPT candidates are *search-state references*: the candidate domain admits
``state_id`` payloads, and ``NanogptEvaluator`` evaluates a state that a
task-local expander (``NanogptWarmupExpander`` / ``NanogptIterationExpander``)
materialized inside an ``OperationSearchEngine``/``SearchEngine`` first. The
shared RL environment's model is stateless (policy text -> payload -> admit ->
evaluate), so it cannot yet drive nanoGPT without a dedicated stateful
environment design.

Until that stateful adapter exists, ``build_env`` fails fast with an explicit
"not wired" error instead of a generic missing factory.
"""

from __future__ import annotations

from typing import Any


def build_rl_components(mode: str = "mock", **kwargs: Any) -> Any:
    raise NotImplementedError(
        "nanogpt has no RL environment adapter yet; its candidates are "
        "search-state references materialized by a task-local expander, which "
        "does not fit the shared stateless LDMEnv proposal->admit->evaluate "
        "model."
    )
