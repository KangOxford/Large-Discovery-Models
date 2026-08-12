"""Candidate parsing and fixed-scaffold assembly for inverse folding."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EDITABLE_START = "# EDITABLE SECTION START"
EDITABLE_END = "# EDITABLE SECTION END"
ALLOWED_IMPORT_ROOTS = frozenset({"math", "torch", "typing"})
ALLOWED_OVERRIDE_KEYS = frozenset(
    {"learning_rate", "dropout", "num_encoder_layers", "batch_size"}
)
BANNED_CALLS = frozenset(
    {"__import__", "breakpoint", "compile", "eval", "exec", "input", "open"}
)
BANNED_CALL_ATTRIBUTES = frozenset(
    {"connect", "load", "popen", "request", "save", "system", "urlopen"}
)


class CandidateValidationError(ValueError):
    """Raised when a generated design violates the task response contract."""


@dataclass(frozen=True)
class CandidateProposal:
    """One accepted, normalized model action."""

    code: str
    reasoning: str
    summary: str
    config_overrides: dict[str, Any]


def parse_model_response(response: str) -> CandidateProposal:
    """Parse and validate the task's JSON code-proposal response."""

    text = _strip_fence(response.strip(), allowed_languages={"", "json"})
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CandidateValidationError(
            f"model response is not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise CandidateValidationError("model response must be a JSON object")
    code = payload.get("code")
    if not isinstance(code, str) or not code.strip():
        raise CandidateValidationError("model response must include non-empty string field 'code'")
    reasoning = payload.get("reasoning", "")
    summary = payload.get("summary", "")
    if not isinstance(reasoning, str) or not isinstance(summary, str):
        raise CandidateValidationError("reasoning and summary must be strings")
    normalized, overrides = validate_candidate_code(code)
    return CandidateProposal(
        code=normalized,
        reasoning=reasoning.strip(),
        summary=summary.strip(),
        config_overrides=overrides,
    )


def validate_candidate_code(code: str) -> tuple[str, dict[str, Any]]:
    """Validate executable design code without importing domain dependencies."""

    normalized = _strip_fence(
        code.replace("\r\n", "\n").strip(),
        allowed_languages={"", "py", "python"},
    ) + "\n"
    if len(normalized) > 120_000:
        raise CandidateValidationError("candidate code exceeds 120000 characters")
    try:
        tree = ast.parse(normalized, filename="candidate_design.py")
    except SyntaxError as exc:
        raise CandidateValidationError(
            f"candidate code has invalid syntax at line {exc.lineno}: {exc.msg}"
        ) from exc

    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    for class_name in ("StructureEncoder", "InverseFoldingModel"):
        class_node = classes.get(class_name)
        if class_node is None:
            raise CandidateValidationError(f"candidate must define class {class_name}")
        methods = {
            node.name for node in class_node.body if isinstance(node, ast.FunctionDef)
        }
        if not {"__init__", "forward"}.issubset(methods):
            raise CandidateValidationError(
                f"{class_name} must define both __init__ and forward"
            )

    overrides_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and _assigned_name(node) == "CONFIG_OVERRIDES"
    ]
    if len(overrides_nodes) != 1:
        raise CandidateValidationError(
            "candidate must define CONFIG_OVERRIDES exactly once"
        )
    try:
        overrides = ast.literal_eval(overrides_nodes[0].value)
    except (TypeError, ValueError, SyntaxError) as exc:
        raise CandidateValidationError("CONFIG_OVERRIDES must be a literal dict") from exc
    _validate_overrides(overrides)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for module in modules:
                if module.split(".", 1)[0] not in ALLOWED_IMPORT_ROOTS:
                    raise CandidateValidationError(
                        f"candidate import is not allowed: {module or '<relative>'}"
                    )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BANNED_CALLS:
                raise CandidateValidationError(
                    f"candidate call is not allowed: {node.func.id}"
                )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in BANNED_CALL_ATTRIBUTES:
                raise CandidateValidationError(
                    f"candidate call is not allowed: {node.func.attr}"
                )
        if (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("__")
            and node.attr != "__init__"
        ):
            raise CandidateValidationError(
                f"dunder attribute access is not allowed: {node.attr}"
            )

    for node in tree.body:
        if isinstance(node, ast.Expr):
            if not (
                isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                raise CandidateValidationError(
                    "candidate cannot execute expressions at module import time"
                )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            try:
                ast.literal_eval(node.value)
            except (TypeError, ValueError, SyntaxError) as exc:
                raise CandidateValidationError(
                    "top-level assignments must contain literal values"
                ) from exc
        elif not isinstance(
            node,
            (
                ast.ClassDef,
                ast.FunctionDef,
                ast.Import,
                ast.ImportFrom,
            ),
        ):
            raise CandidateValidationError(
                f"unsupported top-level statement: {type(node).__name__}"
            )

    return normalized, dict(overrides)


def design_code_without_overrides(code: str) -> str:
    """Return candidate code with its CONFIG_OVERRIDES assignment removed."""

    tree = ast.parse(code)
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.Assign, ast.AnnAssign))
        and _assigned_name(item) == "CONFIG_OVERRIDES"
    )
    lines = code.splitlines(keepends=True)
    del lines[node.lineno - 1 : node.end_lineno]
    return "".join(lines).strip() + "\n"


def assemble_candidate(scaffold: str, proposal: CandidateProposal) -> str:
    """Insert only accepted editable code and overrides into a fixed scaffold."""

    start = scaffold.find(EDITABLE_START)
    end = scaffold.find(EDITABLE_END)
    if start < 0 or end < 0 or end <= start:
        raise CandidateValidationError(
            "scaffold does not contain the expected editable-section markers"
        )
    start_line_end = scaffold.find("\n", start)
    if start_line_end < 0:
        raise CandidateValidationError("editable-section start marker has no body")
    editable_code = design_code_without_overrides(proposal.code)
    assembled = (
        scaffold[: start_line_end + 1]
        + "\n"
        + editable_code
        + "\n"
        + scaffold[end:]
    )
    assembled = _replace_config_overrides(assembled, proposal.config_overrides)
    try:
        ast.parse(assembled, filename="custom_invfold.py")
    except SyntaxError as exc:
        raise CandidateValidationError(
            f"assembled scaffold has invalid syntax at line {exc.lineno}: {exc.msg}"
        ) from exc
    return assembled


def load_and_parse_proposal(path: str | Path) -> CandidateProposal:
    """Load an already accepted candidate design from disk."""

    code = Path(path).read_text(encoding="utf-8")
    normalized, overrides = validate_candidate_code(code)
    return CandidateProposal(normalized, "", "", overrides)


def replace_config_overrides(code: str, overrides: dict[str, Any]) -> str:
    """Replace the candidate's literal override assignment deterministically."""

    _validate_overrides(overrides)
    return _replace_config_overrides(code, overrides)


def _replace_config_overrides(code: str, overrides: dict[str, Any]) -> str:
    tree = ast.parse(code)
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and _assigned_name(node) == "CONFIG_OVERRIDES"
    ]
    if len(nodes) != 1:
        raise CandidateValidationError(
            "scaffold/candidate must define CONFIG_OVERRIDES exactly once"
        )
    node = nodes[0]
    lines = code.splitlines(keepends=True)
    original_line = lines[node.lineno - 1]
    indentation = original_line[: len(original_line) - len(original_line.lstrip())]
    replacement = indentation + "CONFIG_OVERRIDES = " + repr(dict(overrides)) + "\n"
    lines[node.lineno - 1 : node.end_lineno] = [replacement]
    return "".join(lines)


def _assigned_name(node: ast.Assign | ast.AnnAssign) -> str | None:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    if len(targets) == 1 and isinstance(targets[0], ast.Name):
        return targets[0].id
    return None


def _validate_overrides(value: Any) -> None:
    if not isinstance(value, dict):
        raise CandidateValidationError("CONFIG_OVERRIDES must be a dict")
    unknown = sorted(str(key) for key in value if key not in ALLOWED_OVERRIDE_KEYS)
    if unknown:
        raise CandidateValidationError(
            f"unsupported CONFIG_OVERRIDES key(s): {', '.join(unknown)}"
        )
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise CandidateValidationError(f"CONFIG_OVERRIDES[{key!r}] must be numeric")
        if key == "learning_rate" and not 1e-6 <= float(item) <= 0.1:
            raise CandidateValidationError("learning_rate must be in [1e-6, 0.1]")
        if key == "dropout" and not 0.0 <= float(item) < 1.0:
            raise CandidateValidationError("dropout must be in [0, 1)")
        if key in {"num_encoder_layers", "batch_size"} and int(item) != item:
            raise CandidateValidationError(f"{key} must be an integer")
        if key == "num_encoder_layers" and not 1 <= int(item) <= 16:
            raise CandidateValidationError("num_encoder_layers must be in [1, 16]")
        if key == "batch_size" and not 1 <= int(item) <= 1024:
            raise CandidateValidationError("batch_size must be in [1, 1024]")


def _strip_fence(text: str, *, allowed_languages: set[str]) -> str:
    match = re.fullmatch(r"```([A-Za-z0-9_+-]*)\s*(.*?)\s*```", text, re.DOTALL)
    if match is None or match.group(1).lower() not in allowed_languages:
        return text
    return match.group(2)
