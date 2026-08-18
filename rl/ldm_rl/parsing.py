"""Parser discovery for task-declared proposal response spaces.

Every LDM task declares a parser per response space in its
``ResponseSpaceSpec.parser`` field (``module:function`` notation). The
environment loads that parser and calls it with a raw policy action text. Task
declarations are heterogeneous on purpose:

- text-shaped parsers (e.g. ``parse_predictor_specs(text, *, expected_count)``)
  consume raw model text and return a list of candidate payloads;
- payload-shaped parsers (e.g. ``normalize_algorithm_spec(payload)``) normalize
  a single candidate and are wrapped by a per-task factory instead.

This module implements the text-shaped default plus the machinery to inject
declared parameters by signature.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping
from typing import Any

from ldm_tts.contracts import ResponseSpaceSpec
from ldm_tts.transport.parsing import load_json_object


def load_declared_parser(response_space: ResponseSpaceSpec) -> Callable[..., Any] | None:
    """Resolve a response-space parser declared as ``module:function``.

    Dotted ``module.function`` paths are also accepted. Returns ``None`` when
    the response space declares no parser (the factory must supply one).
    """

    parser = (response_space.parser or "").strip()
    if not parser:
        return None
    module_path, _, attr = parser.rpartition(":")
    if not module_path:
        module_path, _, attr = parser.rpartition(".")
    try:
        return getattr(importlib.import_module(module_path), attr)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"cannot load declared parser {parser!r}") from exc


def call_text_parser(
    parser: Callable[..., Any],
    text: str,
    *,
    expected_count: int,
    parameters: Mapping[str, Any] | None = None,
) -> list[Any]:
    """Call a text-shaped parser with only the keyword arguments it accepts.

    Injects ``expected_count`` (aliases ``candidate_count`` / ``count``) when
    the parser declares it, then overlays the expansion's declared parameters.
    The result is normalized to a list of candidate payloads.
    """

    signature = inspect.signature(parser)
    accepts_kwargs = any(
        item.kind is inspect.Parameter.VAR_KEYWORD
        for item in signature.parameters.values()
    )
    merged: dict[str, Any] = {}
    for name in ("expected_count", "candidate_count", "count"):
        if name in signature.parameters:
            merged[name] = expected_count
            break
    merged.update(dict(parameters or {}))
    if not accepts_kwargs:
        merged = {
            name: value for name, value in merged.items() if name in signature.parameters
        }
    result = parser(text, **merged)
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        return [result]
    raise TypeError(
        f"declared parser {getattr(parser, '__name__', parser)!r} returned "
        f"{type(result).__name__}; expected a list of candidate payloads"
    )


def parse_candidate_list(text: str, *, expected_count: int) -> list[Any]:
    """Generic ``{"candidates": [...]}`` text parser fallback."""

    payload = load_json_object(text)
    unknown = set(payload) - {"candidates"}
    if unknown:
        raise ValueError(
            "proposal response must contain only the candidates field; got "
            + ", ".join(sorted(unknown))
        )
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("proposal response candidates field must be a list")
    if len(candidates) != expected_count:
        raise ValueError(
            f"proposal response must contain exactly {expected_count} candidates, "
            f"got {len(candidates)}"
        )
    return list(candidates)
