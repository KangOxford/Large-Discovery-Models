"""Render ldm-2.0 intermediate records into model-training rows."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ldm_tts.data.ir import (
    LDMDataCollectionError,
    jdump,
    normalize_task_id,
    validate_ir_record,
)

SYSTEM_BY_TASK = {
    "nanogpt": (
        "You are a scientific search agent proposing edits to a training program "
        "under an iterative model-based (Bayesian) optimization loop. You may "
        "either propose edits through the current reservoir expansion schema or "
        "update that schema. Return ONLY the JSON action. Never predict objective values, "
        "surrogate mean/variance, acquisition, or rank."
    ),
    "smallmol": (
        "You are a scientific search agent proposing candidate molecules under "
        "an iterative multi-objective Bayesian optimization loop. Return ONLY "
        "the JSON action. Never predict docking score, activity, EHVI, "
        "uncertainty, or rank."
    ),
    "protein": (
        "You are a scientific search agent proposing candidate antibody CDRH3 "
        "sequences under an iterative Bayesian optimization loop. Return ONLY "
        "the JSON action. Never predict binding energy, uncertainty, or rank."
    ),
}

GENERIC_SYSTEM = (
    "You are a scientific search agent in an iterative model-based (Bayesian) "
    "optimization loop. Given the candidate domain, current reservoir expansion "
    "schema, and observed history, choose one action. Return ONLY the JSON action. Never "
    "predict objective values, surrogate statistics, or rank."
)


def _fmt_params(params: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for param in params:
        domain = param.get("domain")
        if isinstance(domain, Mapping):
            alphabet = domain.get("alphabet", [])
            dom = f"length={domain.get('length')}, alphabet={''.join(alphabet)}"
        elif param.get("type") == "choice":
            dom = f"choice in {domain}"
        else:
            dom = f"{param.get('type')} in {domain}"
            if param.get("scale"):
                dom += f", scale={param['scale']}"
        line = f"- {param.get('name')}: {dom}"
        if param.get("edit_op"):
            line += f"; edit_op={param['edit_op']}"
        if param.get("current_value") is not None:
            line += f"; current={param['current_value']}"
        lines.append(line)
    return "\n".join(lines) if lines else "(none)"


def render_prose(ir: Mapping[str, Any], *, include_parent_artifact: bool = True) -> str:
    """Render task + state + request as the model instruction."""

    validate_ir_record(ir)
    task = ir["task"]
    state = ir["search_state"]
    request = ir["request"]
    design_space = state["design_space"]
    parts: list[str] = []

    parts.append(
        f"# Task: {task['id']} ({task['domain']})\n"
        f"{str(task.get('description', '')).strip()}"
    )

    objectives = "\n".join(
        f"- {objective['name']}: {objective['direction']} - "
        f"{objective.get('description', '')}"
        for objective in task["objectives"]
    )
    parts.append(f"## Objectives\n{objectives}")

    space = [
        f"Representation: {design_space['representation']}",
        "\nActive expansion parameters (the surrogate represents only these; `current` is the "
        "value in the parent artifact right now):\n"
        + _fmt_params(design_space.get("active_parameters", [])),
    ]
    inactive = design_space.get("inactive_parameters", [])
    if inactive:
        space.append(
            "\nInactive expansion parameters (available to activate; activating one updates "
            "the expansion schema and surrogate representation for later candidates):\n"
            + _fmt_params(inactive)
        )
    if design_space.get("applied_this_transition"):
        space.append(
            "\nAlready applied inside this child state (do not re-apply):\n"
            + jdump(design_space["applied_this_transition"], indent=1)
        )
    if design_space.get("expansion_history"):
        history_lines = []
        for event in design_space["expansion_history"]:
            history_lines.append(
                f"- round {event.get('round')}: activated {event.get('activated')} "
                f"({event.get('reason')})"
            )
        space.append("\nExpansion history:\n" + "\n".join(history_lines))
    space.append(f"\nNew parameters may be invented: {design_space.get('allows_new_parameters')}")
    if design_space.get("description"):
        space.append(f"\n{design_space['description']}")
    parts.append("## Reservoir expansion schema (current state)\n" + "\n".join(space))

    if state.get("observations"):
        parts.append("## Observed history\n" + jdump(state["observations"], indent=1))
    if state.get("num_evaluated") is not None:
        parts.append(f"Evaluations so far: {state['num_evaluated']}")
    if state.get("best_so_far"):
        parts.append("## Best so far\n" + jdump(state["best_so_far"], indent=1))
    if state.get("surrogate_feedback"):
        parts.append("## Surrogate feedback\n" + jdump(state["surrogate_feedback"], indent=1))
    if state.get("progress"):
        parts.append("## Progress\n" + jdump(state["progress"], indent=1))
    if state.get("do_not_repeat"):
        parts.append("## Do not repeat\n" + jdump(state["do_not_repeat"]))

    raw_context = ir.get("raw_context") or {}
    if isinstance(raw_context, Mapping):
        for key in ("search_state_note", "feedback", "recent_real_evaluated", "recent_trials"):
            if raw_context.get(key):
                title = str(key).replace("_", " ").title()
                parts.append(f"## {title}\n{raw_context[key]}")
        if raw_context.get("target_context"):
            parts.append("## Target context\n" + jdump(raw_context["target_context"], indent=1)[:4000])
        if raw_context.get("candidate_pool"):
            parts.append("## Candidate reservoir\n" + jdump(raw_context["candidate_pool"], indent=1))
        if include_parent_artifact and raw_context.get("parent_train_py"):
            parts.append("## Current parent artifact\n" + str(raw_context["parent_train_py"]))

    request_lines = [
        f"Allowed actions: {request['allowed_actions']}",
        f"Number of candidates: {request.get('num_candidates')}",
    ]
    if request.get("max_edits_per_candidate"):
        request_lines.append(f"Max edits per candidate: {request['max_edits_per_candidate']}")
    if request.get("description"):
        request_lines.append(str(request["description"]))
    request_lines.append(
        '\nReturn a single JSON object: {"type": ..., "reasoning": ..., '
        '"payload": ..., "summary": ...}'
    )
    parts.append("## Your move\n" + "\n".join(request_lines))
    return "\n\n".join(parts)


def render_record(
    ir: Mapping[str, Any],
    *,
    mode: str = "prose",
    include_parent_artifact: bool = True,
) -> dict[str, Any]:
    """Render one IR row into a LlamaFactory Alpaca-style row."""

    validate_ir_record(ir)
    if mode not in {"prose", "json"}:
        raise LDMDataCollectionError(f"unsupported render mode {mode!r}")
    task_id = normalize_task_id(str(ir["task"]["id"]))
    if mode == "json":
        shown = {
            key: value
            for key, value in ir.items()
            if key not in {"action", "collection"}
        }
        instruction = jdump(shown, indent=1)
    else:
        instruction = render_prose(ir, include_parent_artifact=include_parent_artifact)
    return {
        "system": SYSTEM_BY_TASK.get(task_id, GENERIC_SYSTEM),
        "instruction": instruction,
        "input": "",
        "output": jdump(ir["action"]),
        "source": task_id,
        "action_type": ir["action"]["type"],
    }


def dataset_info_payload(sft_filename: str) -> dict[str, Any]:
    """Return the matching LlamaFactory dataset_info entry."""

    return {
        "ldm_bo_sft": {
            "file_name": sft_filename,
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
            },
        }
    }

