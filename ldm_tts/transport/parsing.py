"""Shared JSON response parsing helpers for LDM task adapters."""

from __future__ import annotations

import json
import re
from typing import Any


def strip_json_fence(text: str) -> str:
    """Strip one surrounding Markdown JSON/code fence when present."""

    stripped = str(text or "").strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_json_object_text(text: str) -> str:
    """Extract a JSON object from raw, fenced, or prose-wrapped text."""

    stripped = strip_json_fence(text)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.I | re.S)
    if fenced:
        return fenced.group(1)
    if stripped.startswith("{"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    return stripped[start : end + 1]


def load_json_object(text: str) -> dict[str, Any]:
    """Load a JSON object from permissive LLM response text."""

    payload = extract_json_object_text(text)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"response must be a JSON object, got {type(data).__name__}")
    return data


def reject_keys(data: Any, banned_keys: set[str]) -> None:
    """Reject banned keys anywhere in a nested JSON-like value."""

    if isinstance(data, dict):
        for key, value in data.items():
            if key in banned_keys:
                raise ValueError(f"LLM output may not include {key!r}")
            reject_keys(value, banned_keys)
    elif isinstance(data, list):
        for item in data:
            reject_keys(item, banned_keys)


def require_allowed_keys(data: dict[str, Any], allowed_keys: set[str]) -> None:
    """Reject unknown top-level keys from a loaded JSON object."""

    unknown = set(data) - set(allowed_keys)
    if unknown:
        raise ValueError(f"unknown top-level keys: {sorted(unknown)}")


def require_list(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key!r} must be a list")
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"{key!r} entries must be objects")
    return value


def require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key!r} must be a non-empty string")
    return value.strip()


def require_number(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key!r} must be a number")
    return float(value)


def require_nonnegative_int(data: dict[str, Any], key: str) -> int:
    value = require_number(data, key)
    if value < 0 or int(value) != value:
        raise ValueError(f"{key!r} must be a non-negative integer")
    return int(value)
