"""Policy-facing text rendering for the LDM environment.

The environment is text-in / text-out for the policy: the reset observation
explains the task, the objectives, the candidate domain and the exact response
schema; each step observation reports admission rejections, evaluation results,
the incumbent and the remaining budget. Structured data stays in
``EnvStep.info`` for tooling; the rendered text is what the policy sees and is
appended to the trajectory with ``loss_mask = 0``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from ldm_tts.contracts import (
    CandidateDomainSpec,
    LDMTaskSpec,
    ObjectiveSpec,
    Observation,
    ResponseSpaceSpec,
)


def _render_objectives(specs: Sequence[ObjectiveSpec]) -> str:
    lines = []
    for spec in specs:
        direction = "MAXIMIZE" if spec.direction == "maximize" else "MINIMIZE"
        lines.append(f"- {spec.name!r} ({direction}): {spec.description or 'no description'}")
    return "\n".join(lines)


def _render_domain(domain: CandidateDomainSpec) -> str:
    lines = [f"- Domain: {domain.name} ({domain.kind})"]
    if domain.representation:
        lines.append(f"- Representation: {domain.representation}")
    if domain.constraints:
        lines.append(f"- Constraints: {json.dumps(domain.constraints, sort_keys=True, default=str)}")
    return "\n".join(lines)


def _render_response_space(space: ResponseSpaceSpec) -> str:
    lines = [f"- Action format: {space.output_kind}"]
    if space.description:
        lines.append(f"- {space.description}")
    if space.schema:
        lines.append(
            "- Response schema:\n" + json.dumps(space.schema, indent=2, sort_keys=True)
        )
    return "\n".join(lines)


def render_reset_observation(
    spec: LDMTaskSpec,
    *,
    reservoir_size: int,
    context: dict[str, Any] | None = None,
    extra_instructions: str = "",
) -> str:
    """Render the initial observation (episode prompt) for one campaign."""

    producing = [
        expansion
        for expansion in spec.reservoir.expansions
        if expansion.produces_candidates
    ]
    space_names = sorted({expansion.response_space for expansion in producing})
    spaces = [space for space in spec.response_spaces if space.name in space_names]
    if not spaces:
        spaces = list(spec.response_spaces[:1])
    if not spaces:
        raise ValueError(f"task {spec.task!r} declares no usable response space")

    parts = [
        f"You are solving LDM task {spec.task!r}.",
        "",
        "Objectives:",
        _render_objectives(spec.objectives),
        "",
        "Candidate domain:",
        _render_domain(spec.candidate_domain),
        "",
        "Each turn, propose exactly "
        f"{reservoir_size} distinct candidate payload(s) as raw JSON.",
        "",
        _render_response_space(spaces[0]),
    ]
    if context:
        parts += [
            "",
            "Episode context:",
            json.dumps(context, indent=2, sort_keys=True, default=str),
        ]
    if extra_instructions:
        parts += ["", extra_instructions]
    parts += [
        "",
        "Return only the JSON. Do not add prose, markdown fences, or code.",
    ]
    return "\n".join(parts)


def _brief_payload(payload: Any, *, limit: int = 400) -> str:
    try:
        text = json.dumps(payload, sort_keys=True, default=str)
    except (TypeError, ValueError):
        text = str(payload)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def render_step_observation(
    *,
    round_idx: int,
    remaining_rounds: int,
    parse_error: str | None,
    proposals_count: int,
    rejections: Sequence[Any],
    evaluations: Sequence[Any],
    incumbent: Observation | None,
) -> str:
    """Render the post-step feedback appended to the policy transcript."""

    lines = [f"<round {round_idx}>"]
    if parse_error:
        lines += [
            "Your response could not be parsed as a proposal:",
            f"  {parse_error}",
        ]
        lines += [f"({proposals_count} raw candidate(s) admitted before the error)"]
    else:
        lines += [f"Received {proposals_count} candidate proposal(s)."]
    for rejection in rejections:
        message = getattr(rejection, "message", "") or str(rejection)
        reason = getattr(rejection, "reason", "rejected")
        lines.append(f"- Rejected ({reason}): {message}")
    for item in evaluations:
        evaluation = item.evaluation
        status = evaluation.status
        if status == "succeeded":
            metrics = json.dumps(evaluation.metrics, sort_keys=True)
            lines.append(
                f"- {item.candidate.candidate_id}: SUCCEEDED metrics={metrics}"
            )
        else:
            error = evaluation.error or "unknown error"
            lines.append(f"- {item.candidate.candidate_id}: {status.upper()} ({error})")
    if incumbent is not None:
        lines += [
            "Best so far:",
            f"  candidate: {_brief_payload(incumbent.candidate.payload)}",
            f"  metrics: {json.dumps(dict(incumbent.metrics), sort_keys=True)}",
        ]
    else:
        lines.append("Best so far: none (no successful evaluation yet).")
    if remaining_rounds > 0:
        lines.append(f"Rounds remaining in this episode: {remaining_rounds}")
    lines.append("Propose the next candidates.")
    return "\n".join(lines)
