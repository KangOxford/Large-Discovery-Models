"""Construction and validation of the ldm-2.0 intermediate representation."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "ldm-2.0"

TASK_ID_ALIASES = {
    "small_molecule": "smallmol",
    "small-molecule": "smallmol",
    "molecule": "smallmol",
    "antibody": "protein",
    "antibody_sequence": "protein",
}


class LDMDataCollectionError(ValueError):
    """Raised when an ldm-2.0 record is structurally invalid."""


def jdump(value: Any, indent: int | None = None) -> str:
    """Dump JSON with stable unicode handling used by the data pipeline."""

    return json.dumps(value, ensure_ascii=False, indent=indent)


def normalize_task_id(task_id: str) -> str:
    """Map execution task names onto the ldm-2.0 task ids."""

    text = str(task_id).strip()
    return TASK_ID_ALIASES.get(text, text)


def validate_ir_record(ir: Mapping[str, Any]) -> None:
    """Validate the minimum contract every ldm-2.0 training record must satisfy."""

    required = {"schema_version", "task", "search_state", "request", "action"}
    missing = sorted(required - set(ir))
    if missing:
        raise LDMDataCollectionError(f"IR record missing top-level field(s): {missing}")
    if ir.get("schema_version") != SCHEMA_VERSION:
        raise LDMDataCollectionError(
            f"IR schema_version must be {SCHEMA_VERSION!r}, got {ir.get('schema_version')!r}"
        )
    for field in ("task", "search_state", "request", "action"):
        if not isinstance(ir.get(field), Mapping):
            raise LDMDataCollectionError(f"IR field {field!r} must be an object")

    task = ir["task"]
    if not task.get("id") or not task.get("domain"):
        raise LDMDataCollectionError("IR task must include id and domain")
    objectives = task.get("objectives")
    if not isinstance(objectives, list) or not objectives:
        raise LDMDataCollectionError("IR task.objectives must be a non-empty list")
    for objective in objectives:
        if objective.get("direction") not in {"minimize", "maximize"}:
            raise LDMDataCollectionError(
                f"objective {objective.get('name')!r} has invalid direction "
                f"{objective.get('direction')!r}"
            )

    design_space = ir["search_state"].get("design_space")
    if not isinstance(design_space, Mapping):
        raise LDMDataCollectionError("IR search_state.design_space must be an object")
    if design_space.get("representation") not in {"parameter_edits", "complete_design"}:
        raise LDMDataCollectionError(
            "IR design_space.representation must be parameter_edits or complete_design"
        )

    allowed = ir["request"].get("allowed_actions")
    if not isinstance(allowed, list) or not allowed:
        raise LDMDataCollectionError("IR request.allowed_actions must be a non-empty list")
    action_type = ir["action"].get("type")
    if action_type not in {"propose", "expand_design_space", "add_new_parameter"}:
        raise LDMDataCollectionError(f"invalid action.type {action_type!r}")
    if action_type not in allowed:
        raise LDMDataCollectionError(
            f"action.type {action_type!r} is not in request.allowed_actions {allowed!r}"
        )
    if not isinstance(ir["action"].get("payload"), Mapping):
        raise LDMDataCollectionError("IR action.payload must be an object")


def make_complete_design_ir(
    *,
    task_id: str,
    domain: str,
    task_description: str,
    objectives: Sequence[Mapping[str, Any]],
    design_space_description: str,
    observations: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    request_description: str,
    num_candidates: int | None = None,
    round_idx: int | None = None,
    num_evaluated: int | None = None,
    best_so_far: Mapping[str, Any] | None = None,
    progress: Mapping[str, Any] | None = None,
    do_not_repeat: Sequence[Any] | None = None,
    allows_new_parameters: bool = True,
    reasoning_available: bool = True,
    reasoning: str | None = None,
    summary: str | None = None,
    raw_context: Mapping[str, Any] | None = None,
    active_parameters: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a standard ldm-2.0 row for complete-design proposal tasks."""

    normalized_task = normalize_task_id(task_id)
    ir = {
        "schema_version": SCHEMA_VERSION,
        "task": {
            "id": normalized_task,
            "domain": domain,
            "description": task_description,
            "objectives": [dict(objective) for objective in objectives],
            "reasoning_available": bool(reasoning_available),
        },
        "search_state": {
            "round": round_idx,
            "num_evaluated": num_evaluated,
            "design_space": {
                "representation": "complete_design",
                "active_parameters": [dict(param) for param in (active_parameters or [])],
                "inactive_parameters": [],
                "expansion_history": [],
                "allows_new_parameters": bool(allows_new_parameters),
                "description": design_space_description,
            },
            "observations": [dict(observation) for observation in observations],
            "best_so_far": dict(best_so_far) if best_so_far else None,
            "surrogate_feedback": None,
            "progress": dict(progress) if progress else None,
            "do_not_repeat": list(do_not_repeat or []),
        },
        "request": {
            "allowed_actions": ["propose"],
            "num_candidates": int(num_candidates if num_candidates is not None else len(candidates)),
            "max_edits_per_candidate": None,
            "description": request_description,
        },
        "action": {
            "type": "propose",
            "reasoning": reasoning,
            "payload": {"candidates": [dict(candidate) for candidate in candidates]},
            "summary": summary,
        },
    }
    if raw_context:
        ir["raw_context"] = dict(raw_context)
    validate_ir_record(ir)
    return ir


def make_parameter_edit_ir(
    *,
    task_id: str,
    domain: str,
    task_description: str,
    objectives: Sequence[Mapping[str, Any]],
    active_parameters: Sequence[Mapping[str, Any]],
    inactive_parameters: Sequence[Mapping[str, Any]],
    action: Mapping[str, Any],
    request_description: str,
    design_space_description: str = "",
    allowed_actions: Sequence[str] | None = None,
    num_candidates: int = 1,
    max_edits_per_candidate: int | None = None,
    round_idx: int | None = None,
    num_evaluated: int | None = None,
    observations: Sequence[Mapping[str, Any]] | None = None,
    best_so_far: Mapping[str, Any] | None = None,
    surrogate_feedback: Mapping[str, Any] | None = None,
    progress: Mapping[str, Any] | None = None,
    do_not_repeat: Sequence[Any] | None = None,
    expansion_history: Sequence[Mapping[str, Any]] | None = None,
    applied_this_transition: Sequence[Mapping[str, Any]] | None = None,
    allows_new_parameters: bool = True,
    reasoning_available: bool = True,
    raw_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standard ldm-2.0 row for parameter-edit proposal tasks."""

    normalized_task = normalize_task_id(task_id)
    allowed = list(allowed_actions or ["propose"])
    if inactive_parameters and "expand_design_space" not in allowed:
        allowed.append("expand_design_space")
    design_space = {
        "representation": "parameter_edits",
        "active_parameters": [dict(param) for param in active_parameters],
        "inactive_parameters": [dict(param) for param in inactive_parameters],
        "expansion_history": [dict(item) for item in (expansion_history or [])],
        "allows_new_parameters": bool(allows_new_parameters),
        "description": design_space_description,
    }
    if applied_this_transition is not None:
        design_space["applied_this_transition"] = [
            dict(item) for item in applied_this_transition
        ]
    ir = {
        "schema_version": SCHEMA_VERSION,
        "task": {
            "id": normalized_task,
            "domain": domain,
            "description": task_description,
            "objectives": [dict(objective) for objective in objectives],
            "reasoning_available": bool(reasoning_available),
        },
        "search_state": {
            "round": round_idx,
            "num_evaluated": num_evaluated,
            "design_space": design_space,
            "observations": [dict(observation) for observation in (observations or [])],
            "best_so_far": dict(best_so_far) if best_so_far else None,
            "surrogate_feedback": dict(surrogate_feedback) if surrogate_feedback else None,
            "progress": dict(progress) if progress else None,
            "do_not_repeat": list(do_not_repeat or []),
        },
        "request": {
            "allowed_actions": allowed,
            "num_candidates": int(num_candidates),
            "max_edits_per_candidate": max_edits_per_candidate,
            "description": request_description,
        },
        "action": dict(action),
    }
    if raw_context:
        ir["raw_context"] = dict(raw_context)
    validate_ir_record(ir)
    return ir


SMALLMOL_PROMPT_HEADS = [
    "Task",
    "Target context",
    "Background",
    "Molecule context table",
    "How to use the molecule context",
    "Generation principles",
    "SMILES hygiene",
    "Generation focus",
    "History summary",
    "JSON output format",
]

SMALLMOL_ROLE_MAP = {
    "pareto_front": "pareto_front",
    "top_low_vina": "top_objective_0",
    "top_high_activity": "top_objective_1",
    "balanced_elites": "elite",
    "recent_selected": "recent",
}

_SMALLMOL_FORMAT_PROSE = re.compile(
    r"(Use compact minified JSON[^.]*\."
    r"|Return JSON only[^.]*\."
    r"|The top-level JSON value[^.]*\."
    r"|Do not include ids, scores[^.]*\."
    r"|keep each rationale under \d+ words\.?)",
    re.I,
)


def smallmol_ir_from_prompt_response(
    instruction: str,
    output: str,
    *,
    round_idx: int | None = None,
    source_id: str | None = None,
) -> dict[str, Any] | None:
    """Convert one accepted M1 direct-SMILES LLM call into ldm-2.0 IR.

    This adapter is intentionally narrow: it handles the direct SMILES proposal
    prompt used by
    ``tasks.small_molecule.core.ldm_tilted_case2.prompts.build_m1_prompt``. Seed
    planning and non-M1 prompts should use their own adapters rather than being
    forced through this schema.
    """

    sections = _sections(instruction, SMALLMOL_PROMPT_HEADS)
    if "History summary" not in sections or "Task" not in sections:
        return None
    try:
        history = json.loads(sections.get("History summary", "{}"))
    except json.JSONDecodeError:
        return None
    try:
        molecule_context = json.loads(sections.get("Molecule context table", "[]"))
    except json.JSONDecodeError:
        molecule_context = []
    try:
        parsed_output = json.loads(output)
    except json.JSONDecodeError:
        return None
    direct = parsed_output.get("direct_smiles")
    if not isinstance(direct, list):
        return None

    observations_by_design: dict[str, dict[str, Any]] = {}
    for view_name, role in SMALLMOL_ROLE_MAP.items():
        for entry in history.get(view_name, []) or []:
            smiles = entry.get("smiles")
            if not smiles:
                continue
            observation = observations_by_design.setdefault(
                str(smiles),
                {"design": str(smiles), "results": None, "roles": []},
            )
            scores = entry.get("scores")
            if scores and observation["results"] is None:
                observation["results"] = {"vina": scores[0], "activity": scores[1]}
            if role not in observation["roles"]:
                observation["roles"].append(role)

    if isinstance(molecule_context, list):
        for item in molecule_context:
            if not isinstance(item, Mapping):
                continue
            smiles = item.get("smiles")
            if smiles not in observations_by_design:
                continue
            observations_by_design[str(smiles)]["description"] = "; ".join(
                f"{key}={value}"
                for key, value in item.items()
                if key != "smiles" and value is not None
            )

    candidates = [
        {"design": item.get("smiles"), "rationale": item.get("rationale")}
        for item in direct
        if isinstance(item, Mapping) and item.get("smiles")
    ]
    if not candidates:
        return None

    alert = history.get("recent_diversity_alert")
    progress = None
    if isinstance(alert, Mapping):
        progress = {
            "stalled": True,
            "rounds_since_improvement": None,
            "description": alert.get("instruction"),
        }

    request_description = _strip_smallmol_format_prose(
        (sections.get("Task") or "").strip()
        + "\n"
        + (sections.get("Generation focus") or "").strip()
    )
    raw_context = {
        "generation_principles": (sections.get("Generation principles") or "").strip(),
        "how_to_use_context": (sections.get("How to use the molecule context") or "").strip(),
    }
    if source_id:
        raw_context["source_id"] = source_id

    return make_complete_design_ir(
        task_id="smallmol",
        domain="molecule",
        task_description=(
            (sections.get("Target context") or "").strip()
            + "\n\n"
            + (sections.get("Background") or "").strip()
        ).strip(),
        objectives=[
            {
                "name": "vina_docking",
                "direction": "minimize",
                "description": "AutoDock Vina docking score; lower is better.",
            },
            {
                "name": "neural_activity",
                "direction": "maximize",
                "description": "Target-specific activity model prediction; higher is better.",
            },
        ],
        design_space_description=(sections.get("SMILES hygiene") or "").strip(),
        observations=list(observations_by_design.values()),
        candidates=candidates,
        request_description=request_description,
        num_candidates=len(candidates),
        round_idx=round_idx,
        num_evaluated=history.get("n_evaluated"),
        progress=progress,
        do_not_repeat=history.get("avoid_exact_smiles", []) or [],
        allows_new_parameters=True,
        reasoning_available=True,
        raw_context=raw_context,
    )


def smallmol_irs_from_round_record(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract ldm-2.0 rows from a small-molecule round record."""

    rows: list[dict[str, Any]] = []
    round_idx = record.get("round_idx")
    for attempt in record.get("llm_attempts", []) or []:
        if not isinstance(attempt, Mapping):
            continue
        if attempt.get("error"):
            continue
        if attempt.get("stage") and attempt.get("stage") != "m1_direct":
            continue
        instruction = attempt.get("user_prompt")
        output = attempt.get("raw_text") or attempt.get("raw_output")
        if not isinstance(instruction, str) or not isinstance(output, str):
            continue
        ir = smallmol_ir_from_prompt_response(
            instruction,
            output,
            round_idx=int(round_idx) if round_idx is not None else None,
            source_id=str(attempt.get("source_id") or ""),
        )
        if ir is not None:
            rows.append(ir)
    return rows


def _strip_smallmol_format_prose(text: str) -> str:
    return re.sub(r"\s{2,}", " ", _SMALLMOL_FORMAT_PROSE.sub("", text or "")).strip()


def _sections(text: str, headings: Sequence[str]) -> dict[str, str]:
    indexed = []
    for heading in headings:
        idx = text.find(heading + ":")
        if idx >= 0:
            indexed.append((idx, heading))
    indexed.sort()
    sections: dict[str, str] = {}
    for pos, (idx, heading) in enumerate(indexed):
        end = indexed[pos + 1][0] if pos + 1 < len(indexed) else len(text)
        sections[heading] = text[idx + len(heading) + 1 : end].strip()
    return sections

