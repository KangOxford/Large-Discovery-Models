"""bo/ldm/llm/response_parser.py — parses LLM JSON response.

Output schema:
  - {"rationale": "...", "update_trust_region": "<python-source>"}
  - {"rationale": "...", "update_bias": "<python-source>"}
  - {"rationale": "...", "update_trust_region": "...", "update_bias": "..."}
  - {}   ← no update

All keys are optional; missing key = keep current value.
``rationale`` is recommended but not required.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class ParsedUpdate:
    """Parsed LLM update intent."""
    update_trust_region: str | None
    update_bias: str | None
    rationale: str | None = None

    @property
    def is_noop(self) -> bool:
        return self.update_trust_region is None and self.update_bias is None


def parse_response(raw: str) -> ParsedUpdate:
    """Parse an LLM response string into a :class:`ParsedUpdate`.

    Raises ``ValueError`` if not valid JSON or has unknown top-level keys.
    """
    if not raw or not raw.strip():
        raise ValueError("LLM response is empty")

    raw = raw.strip()
    # Strip optional ```json ... ``` fences
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e

    if not isinstance(obj, dict):
        raise ValueError(f"response must be a JSON object, got {type(obj).__name__}")

    allowed = {"update_trust_region", "update_bias", "rationale"}
    unknown = set(obj) - allowed
    if unknown:
        raise ValueError(f"unknown top-level keys: {sorted(unknown)}")

    return ParsedUpdate(
        update_trust_region=obj.get("update_trust_region"),
        update_bias=obj.get("update_bias"),
        rationale=obj.get("rationale"),
    )
