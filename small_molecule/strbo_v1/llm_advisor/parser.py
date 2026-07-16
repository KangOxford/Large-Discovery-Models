"""Parse LLM responses into blocks, with three exception types.

Public API:

* :func:`parse_blocks` — extract all triple-backtick json blocks
  from an LLM response, validate each against the schema, dispatch to
  the right dataclass. Raises :class:`ParseError` /
  :class:`SchemaError` / :class:`SemanticError`.
* :func:`validate_blocks_phase` — reject blocks that appear in the
  wrong phase (Phase A vs Phase B).
* :func:`validate_semantics` — non-schema checks that need cross-
  block state (RDKit SMILES validity, reject targets in pool, etc.).
  These are kept separate so the pure parser can run without RDKit if
  needed (e.g. for fast unit tests).

Error types:

* :class:`ParseError` — no triple-backtick json blocks found, or the
  JSON inside is malformed. Retried with feedback.
* :class:`SchemaError` — the JSON is well-formed but violates the
  block schema. Retried with feedback.
* :class:`SemanticError` — the JSON is valid but business rules are
  violated (e.g. ``override:<EMPTY>``, RDKit-invalid SMILES, reject
  target not in pool, ``review_bo`` in Phase A). Retried with feedback.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, List, Sequence

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from strbo_v1.llm_advisor.blocks import (
    LLMBlock,
    block_from_dict,
)
from strbo_v1.llm_advisor.schema import get_validator

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class ParseError(Exception):
    """No triple-backtick json blocks extracted, or JSON inside is malformed."""


class SchemaError(Exception):
    """JSON parses but violates the block schema."""


class SemanticError(Exception):
    """JSON is valid but violates business rules (phase, RDKit, in-pool, etc.)."""


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


# Match ```json ... ``` blocks. Non-greedy on the body, DOTALL so newlines
# inside the block are accepted.
_JSON_BLOCK_RE = re.compile(
    r"```json\s*(\{.*?\})\s*```",
    flags=re.DOTALL,
)


def extract_json_payloads(text: str) -> List[str]:
    """Return the raw JSON object strings from an LLM response.

    Tries three formats, in order:

    1. ``\\`\\`\\`json ... \\`\\`\\``` fenced code blocks (legacy; still
       used by :class:`MockLLMClient`).
    2. A single bare JSON object: ``{"type": ...}``.
    3. A JSON array of objects: ``[{...}, {...}]`` — used when the
       LLM wants to emit multiple actions in one round.

    Returns a list of canonical JSON-object strings (one per action),
    in the order they appear in the response. Empty / malformed
    input raises :class:`ParseError`.
    """
    text = (text or "").strip()
    if not text:
        raise ParseError("LLM response is empty")

    # 1) Fenced blocks (legacy; takes priority so MockLLMClient tests
    # keep working as-is).
    fenced = _JSON_BLOCK_RE.findall(text)
    if fenced:
        return list(fenced)

    # 2 & 3) Bare JSON: either a single object or an array of objects.
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError(
            f"no valid JSON found in LLM response "
            f"(tried fenced blocks and bare JSON; "
            f"response length={len(text)} chars; "
            f"json error: {exc.msg} at line {exc.lineno} col {exc.colno})"
        ) from exc

    if isinstance(parsed, dict):
        return [text]
    if isinstance(parsed, list):
        if not parsed:
            raise ParseError("LLM returned an empty JSON array")
        non_object_types = [
            type(item).__name__ for item in parsed
            if not isinstance(item, dict)
        ]
        if non_object_types:
            raise ParseError(
                "LLM JSON array must contain only objects, "
                f"got types: {non_object_types}"
            )
        return [json.dumps(item) for item in parsed]

    raise ParseError(
        f"top-level JSON must be an object or array, "
        f"got {type(parsed).__name__}"
    )


def _parse_one_payload(raw: str) -> Dict[str, Any]:
    """Parse a single JSON object string. Raises :class:`ParseError`."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ParseError(f"malformed JSON: {exc.msg} at line {exc.lineno} col {exc.colno}") from exc
    if not isinstance(obj, dict):
        raise ParseError(f"top-level JSON must be an object, got {type(obj).__name__}")
    return obj


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _validate_schema(validator: Draft202012Validator, payload: Dict[str, Any]) -> None:
    """Raise :class:`SchemaError` on first violation.

    Strategy:

    1. Look up the block's ``type`` field and locate the matching
       branch in the schema's ``definitions``. If no branch matches
       the type, raise a clear "unknown type" error.
    2. Validate ``payload`` against just that single branch's schema.
       This yields errors that point at the actual problem (e.g.
       ``'reason' is not one of [...]``) rather than the first
       cross-branch error from a top-level ``oneOf``.
    """
    block_type = payload.get("type") if isinstance(payload, dict) else None

    # Collect the set of valid block types from the schema.
    valid_types: Dict[str, Dict[str, Any]] = {}
    for name, branch in validator.schema.get("definitions", {}).items():
        const = branch.get("properties", {}).get("type", {}).get("const")
        if isinstance(const, str):
            valid_types[const] = branch

    if not isinstance(block_type, str) or block_type not in valid_types:
        raise SchemaError(
            f"<root>: 'type'={block_type!r} is not a known block "
            f"type; expected one of {sorted(valid_types)}"
        )

    # Build a per-branch validator and validate against it.
    branch_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **valid_types[block_type],
    }
    branch_validator = Draft202012Validator(branch_schema)
    errors = sorted(
        branch_validator.iter_errors(payload),
        key=lambda e: (len(e.absolute_path), str(e.absolute_path)),
    )
    if not errors:
        return
    err = errors[0]
    path = "/".join(str(p) for p in err.absolute_path) or "<root>"
    raise SchemaError(f"{path}: {err.message}")


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def parse_blocks(
    text: str,
    *,
    validator: Draft202012Validator | None = None,
) -> List[LLMBlock]:
    """Parse an LLM response string into a list of :class:`LLMBlock` instances.

    Steps:

    1. Extract every triple-backtick json payload (left-to-right).
    2. ``json.loads`` each one.
    3. Validate against the block schema.
    4. Dispatch to the matching block dataclass via
       :func:`blocks.block_from_dict`.

    Raises:
        ParseError: 0 blocks or malformed JSON.
        SchemaError: at least one block fails the schema.
    """
    payloads = extract_json_payloads(text)
    if not payloads:
        raise ParseError(
            "no ```json ... ``` blocks found in LLM response "
            f"(response length={len(text or '')} chars)"
        )

    v = validator if validator is not None else get_validator()
    blocks: List[LLMBlock] = []
    for i, raw in enumerate(payloads):
        obj = _parse_one_payload(raw)
        _validate_schema(v, obj)
        try:
            blocks.append(block_from_dict(obj))
        except ValueError as exc:
            # Unknown "type" or missing type — wrap as SchemaError so
            # the advisor treats it as a retryable validation failure.
            raise SchemaError(f"block #{i}: {exc}") from exc
    return blocks


# ---------------------------------------------------------------------------
# Phase filtering
# ---------------------------------------------------------------------------


def validate_blocks_phase(
    blocks: Sequence[LLMBlock], phase: str,
) -> None:
    """Raise :class:`SemanticError` if any block is disallowed in ``phase``.

    Args:
        blocks: parsed blocks from :func:`parse_blocks`.
        phase: stage identifier.  Accepted values:

            * ``"A_actions"``: pool-management actions
              (propose, reject, analog, noop).
            * ``"A_review_analogs"``: review-analogs stage
              (review_analogs only).
            * ``"B_suggestions"``: BO review
              (review_bo only).

    The check is structural — the LLM cannot avoid the constraint by
    reformatting JSON, because the block ``type`` field is constrained
    by the schema to a fixed enum.
    """
    from strbo_v1.llm_advisor.blocks import (
        PHASE_A_ACTIONS_ALLOWED,
        PHASE_A_REVIEW_ANALOGS_ALLOWED,
        PHASE_B_SUGGESTIONS_ALLOWED,
    )

    # Normalise legacy values.
    _STAGE_MAP = {
        "A_actions": PHASE_A_ACTIONS_ALLOWED,
        "A_review_analogs": PHASE_A_REVIEW_ANALOGS_ALLOWED,
        "B_suggestions": PHASE_B_SUGGESTIONS_ALLOWED,
    }

    if phase not in _STAGE_MAP:
        raise ValueError(
            f"phase must be one of {sorted(_STAGE_MAP)}, got {phase!r}"
        )
    allowed = set(_STAGE_MAP[phase])
    bad = [b for b in blocks if b.type not in allowed]
    if bad:
        types = sorted({b.type for b in bad})
        raise SemanticError(
            f"Stage {phase} disallows block types {types}; "
            f"allowed: {sorted(allowed)}"
        )


# ---------------------------------------------------------------------------
# Business semantics
# ---------------------------------------------------------------------------


def _is_rdkit_valid(smiles: str) -> bool:
    """Return True if RDKit can parse ``smiles`` (best-effort, RDKit-optional).

    If RDKit is not importable, returns True (skip chemical check; the
    orchestrator can still pass the SMILES to its scorer and let the
    scorer fail with a clear error). This keeps the parser usable in
    minimal test environments.
    """
    text = (smiles or "").strip()
    if not text:
        return False
    try:
        from rdkit import Chem                       # type: ignore
    except ImportError:
        return True
    try:
        return Chem.MolFromSmiles(text) is not None
    except Exception:                                # pragma: no cover
        return False


def validate_semantics(
    blocks: Sequence[LLMBlock],
    *,
    pool: Iterable[str] | None = None,
    phase: str | None = None,
    use_rdkit: bool = True,
    bo_suggestions: Sequence[str] | None = None,
    pool_min_size: int | None = None,
) -> None:
    """Cross-block business validation. Raises :class:`SemanticError`.

    Checks performed (best-effort, all raise on first violation):

    1. If ``phase`` is given, run :func:`validate_blocks_phase`.
    2. ``RejectBlock.targets`` must all be in ``pool`` (if pool given).
    3. ``ProposeBlock.smiles`` and ``AnalogBlock.seeds`` must be
       RDKit-valid (if ``use_rdkit``).
    4. ``ReviewBOBlock.decisions`` values must be well-formed
       (``"override:"`` followed by a non-empty SMILES).
    5. ``ReviewAnalogsBlock.decisions`` keys must be non-empty
       strings.
    6. **Pool refill gate (Phase A only)**: a bare ``noop`` block is
       rejected when ``len(pool) < pool_min_size``. The LLM must
       emit ``propose`` or ``analog`` to refill the pool. The
       advisor's retry loop catches the error and re-prompts.

    Soft checks (logged but not raised):

    * Phase B: a ``review_bo`` block with empty ``decisions`` while
      there ARE BO suggestions. The orchestrator will default all
      picks to ``"ok"``; we log for audit.

    The function returns silently when every block passes; the caller
    can then attach the blocks to the round state.
    """
    if phase is not None:
        validate_blocks_phase(blocks, phase)

    pool_set = set(pool) if pool is not None else None
    pool_list = list(pool) if pool is not None else None
    seen_types: Dict[str, int] = {}
    for i, b in enumerate(blocks):
        seen_types[b.type] = seen_types.get(b.type, 0) + 1
        # At-most-one per type per round (LLM may emit extras — reject)
        if seen_types[b.type] > 1:
            raise SemanticError(
                f"block #{i}: duplicate type {b.type!r}; "
                f"at most one block per type per round"
            )

    for i, b in enumerate(blocks):
        if b.type == "reject":
            if pool_set is not None:
                missing = [t for t in b.targets if t not in pool_set]
                if missing:
                    raise SemanticError(
                        f"block #{i} (reject): targets not in pool: {missing[:5]}"
                        f"{'...' if len(missing) > 5 else ''}"
                    )
        elif b.type == "propose":
            if use_rdkit:
                bad = [s for s in b.smiles if not _is_rdkit_valid(s)]
                if bad:
                    raise SemanticError(
                        f"block #{i} (propose): RDKit-invalid SMILES: {bad[:3]}"
                    )
        elif b.type == "analog":
            if use_rdkit:
                bad = [s for s in b.seeds if not _is_rdkit_valid(s)]
                if bad:
                    raise SemanticError(
                        f"block #{i} (analog): RDKit-invalid seed SMILES: {bad[:3]}"
                    )
        elif b.type == "noop" and phase == "A_actions":
            # Pool refill gate: Noop is forbidden when the pool is
            # below the minimum size. The LLM must refill the pool
            # via `propose` or `analog` instead. The advisor's
            # retry loop re-prompts with this error message.
            if (pool_list is not None and pool_min_size is not None
                    and len(pool_list) < pool_min_size):
                raise SemanticError(
                    f"block #{i} (noop): pool has {len(pool_list)} "
                    f"SMILES (< min {pool_min_size}); you MUST emit a "
                    f"`propose` block with new SMILES, or an "
                    f"`analog` block to expand an existing pool "
                    f"member. `noop` is not allowed when the pool "
                    f"is below the minimum size."
                )
        elif b.type == "review_bo":
            for bo_smi, ver in b.decisions.items():
                if ver.startswith("override:"):
                    new_smi = ver[len("override:"):].strip()
                    if not new_smi:
                        raise SemanticError(
                            f"block #{i} (review_bo): override target empty for {bo_smi!r}"
                        )
                    if use_rdkit and not _is_rdkit_valid(new_smi):
                        raise SemanticError(
                            f"block #{i} (review_bo): override SMILES not RDKit-valid: {new_smi!r}"
                        )
                elif ver not in ("ok", "skip"):
                    raise SemanticError(
                        f"block #{i} (review_bo): bad verdict {ver!r} for {bo_smi!r}"
                    )
            # Soft check: empty decisions when picks exist.
            # (Not raised — the orchestrator will default to "ok".)
            if not b.decisions and bo_suggestions:
                LOGGER.warning(
                    "block #%d (review_bo): no decisions for %d BO pick(s); "
                    "defaulting to 'ok' (LLM chose not to decide).",
                    i, len(bo_suggestions),
                )
        elif b.type == "review_analogs":
            for k in b.decisions:
                if not isinstance(k, str) or not k.strip():
                    raise SemanticError(
                        f"block #{i} (review_analogs): empty decision key"
                    )


# ---------------------------------------------------------------------------
# Friendly error formatter for prompt feedback
# ---------------------------------------------------------------------------


def format_error_for_prompt(exc: BaseException) -> str:
    """Format an exception for inclusion in the LLM prompt's previous-errors."""
    if isinstance(exc, (ParseError, SchemaError, SemanticError)):
        return f"{type(exc).__name__}: {exc}"
    return f"{type(exc).__name__}: {exc}"


__all__ = [
    "ParseError",
    "SchemaError",
    "SemanticError",
    "parse_blocks",
    "validate_blocks_phase",
    "validate_semantics",
    "format_error_for_prompt",
    "extract_json_payloads",
]
