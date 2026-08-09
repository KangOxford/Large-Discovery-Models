"""Utilities for collecting ldm-2.0 fine-tuning rows during task execution.

The task runners already know the current search state, the request sent to the
teacher model, and the accepted model action. This module keeps the shared pieces
in one place:

* validation for the ldm-2.0 intermediate representation (IR)
* rendering IR into LlamaFactory Alpaca rows
* an append-only sink that writes IR and SFT JSONL files

Task-specific code should construct the IR as close as possible to the LLM call,
after rejected attempts have been filtered out by the task parser/runner.
Selection/evaluation metadata can be stored as provenance, but it is deliberately
not rendered into the prompt unless the task explicitly puts it in search_state.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "ldm-2.0"

TASK_ID_ALIASES = {
    "small_molecule": "smallmol",
    "small-molecule": "smallmol",
    "molecule": "smallmol",
    "antibody": "protein",
    "antibody_sequence": "protein",
}

SYSTEM_BY_TASK = {
    "nanogpt": (
        "You are a scientific search agent proposing edits to a training program "
        "under an iterative model-based (Bayesian) optimization loop. You may "
        "either propose edits within the current design space or expand that "
        "space. Return ONLY the JSON action. Never predict objective values, "
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
    "optimization loop. Given the task, the current design space, and the "
    "observed history, choose one action. Return ONLY the JSON action. Never "
    "predict objective values, surrogate statistics, or rank."
)


class LDMDataCollectionError(ValueError):
    """Raised when an ldm-2.0 record is structurally invalid."""


def jdump(value: Any, indent: int | None = None) -> str:
    """Dump JSON with stable unicode handling used by the data pipeline."""

    return json.dumps(value, ensure_ascii=False, indent=indent)


def normalize_task_id(task_id: str) -> str:
    """Map execution task names onto the ldm-2.0 task ids."""

    text = str(task_id).strip()
    return TASK_ID_ALIASES.get(text, text)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file or a JSON array from disk."""

    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise LDMDataCollectionError(f"JSON array expected in {path}")
        return rows
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    """Append one JSON object to a JSONL file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(jdump(dict(row)) + "\n")


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
        "\nActive parameters (the surrogate models only these; `current` is the "
        "value in the parent artifact right now):\n"
        + _fmt_params(design_space.get("active_parameters", [])),
    ]
    inactive = design_space.get("inactive_parameters", [])
    if inactive:
        space.append(
            "\nInactive parameters (available to activate; activating one expands "
            "the surrogate's feature space for later candidates):\n"
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
    parts.append("## Design space (current state - you may act on it)\n" + "\n".join(space))

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


@dataclass(frozen=True)
class DataCollectionPaths:
    """Paths written by :class:`DataCollectionSink`."""

    root_dir: Path
    ir_path: Path
    sft_path: Path | None
    dataset_info_path: Path | None


class DataCollectionSink:
    """Append-only writer for ldm-2.0 IR and rendered SFT records."""

    def __init__(
        self,
        root_dir: str | Path | None,
        *,
        enabled: bool = True,
        ir_filename: str = "ldm_ir.jsonl",
        sft_filename: str | None = "ldm_sft.jsonl",
        dataset_info_filename: str | None = "dataset_info.json",
        render_mode: str = "prose",
        include_parent_artifact: bool = True,
    ) -> None:
        self.enabled = bool(enabled) and root_dir is not None
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.ir_filename = ir_filename
        self.sft_filename = sft_filename
        self.dataset_info_filename = dataset_info_filename
        self.render_mode = render_mode
        self.include_parent_artifact = include_parent_artifact
        self._lock = threading.Lock()
        if self.enabled:
            assert self.root_dir is not None
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self._write_dataset_info()

    @classmethod
    def disabled(cls) -> "DataCollectionSink":
        """Return a no-op collector."""

        return cls(None, enabled=False)

    @classmethod
    def from_env(cls, *, default_root: str | Path | None = None) -> "DataCollectionSink":
        """Create a collector from environment variables.

        Environment knobs:
        * LDM_DATA_COLLECTION_ENABLED: truthy/falsey on-off switch
        * LDM_DATA_COLLECTION_DIR: output directory for JSONL artifacts
        * LDM_DATA_COLLECTION_RENDER: prose or json
        * LDM_DATA_COLLECTION_STRIP_PARENT_ARTIFACT: truthy to drop train.py text
        """

        enabled_text = os.environ.get("LDM_DATA_COLLECTION_ENABLED", "")
        explicit_dir = os.environ.get("LDM_DATA_COLLECTION_DIR", "")
        enabled = _truthy(enabled_text) if enabled_text else bool(explicit_dir)
        if not enabled:
            return cls.disabled()
        root = explicit_dir or default_root
        if root is None:
            return cls.disabled()
        render_mode = os.environ.get("LDM_DATA_COLLECTION_RENDER", "prose").strip() or "prose"
        strip_parent = _truthy(os.environ.get("LDM_DATA_COLLECTION_STRIP_PARENT_ARTIFACT", ""))
        return cls(
            root,
            enabled=True,
            render_mode=render_mode,
            include_parent_artifact=not strip_parent,
        )

    @property
    def paths(self) -> DataCollectionPaths | None:
        """Return output paths, or None when the sink is disabled."""

        if not self.enabled or self.root_dir is None:
            return None
        sft_path = self.root_dir / self.sft_filename if self.sft_filename else None
        dataset_info_path = (
            self.root_dir / self.dataset_info_filename
            if self.dataset_info_filename and self.sft_filename
            else None
        )
        return DataCollectionPaths(
            root_dir=self.root_dir,
            ir_path=self.root_dir / self.ir_filename,
            sft_path=sft_path,
            dataset_info_path=dataset_info_path,
        )

    def append(
        self,
        ir: Mapping[str, Any],
        *,
        provenance: Mapping[str, Any] | None = None,
        outcome: Mapping[str, Any] | None = None,
    ) -> None:
        """Append one training example.

        Provenance/outcome fields are stored on the IR JSONL row under a
        collection-only top-level key. The renderer ignores that key, so it cannot
        leak into the model instruction.
        """

        if not self.enabled:
            return
        validate_ir_record(ir)
        paths = self.paths
        if paths is None:
            return
        stored = json.loads(jdump(dict(ir)))
        if provenance or outcome:
            stored["collection"] = {}
            if provenance:
                stored["collection"]["provenance"] = dict(provenance)
            if outcome:
                stored["collection"]["outcome"] = dict(outcome)
        sft_row = None
        if paths.sft_path is not None:
            sft_row = render_record(
                stored,
                mode=self.render_mode,
                include_parent_artifact=self.include_parent_artifact,
            )
        with self._lock:
            append_jsonl(paths.ir_path, stored)
            if paths.sft_path is not None and sft_row is not None:
                append_jsonl(paths.sft_path, sft_row)

    def extend(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> int:
        """Append multiple IR rows and return the number written."""

        count = 0
        for row in rows:
            self.append(row, provenance=provenance)
            count += 1
        return count

    def _write_dataset_info(self) -> None:
        if self.root_dir is None or not self.dataset_info_filename or not self.sft_filename:
            return
        path = self.root_dir / self.dataset_info_filename
        path.write_text(
            jdump(dataset_info_payload(self.sft_filename), indent=2) + "\n",
            encoding="utf-8",
        )


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
    prompt used by ``strbo_v1.ldm_tilted_case2.prompts.build_m1_prompt``. Seed
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


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}
