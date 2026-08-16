"""core/ldm/llm/response_parser.py — parses LLM JSON response.

Output schema:
  - {"rationale": "...", "update_trust_region": "<python-source>"}
  - {"rationale": "...", "update_bias": "<python-source>"}
  - {"rationale": "...", "update_trust_region": "...", "update_bias": "..."}
  - {}   ← no update

All keys are optional; missing key = keep current value.
``rationale`` is recommended but not required.
"""
from __future__ import annotations

from dataclasses import dataclass

from ldm_tts.transport.parsing import load_json_object, require_allowed_keys


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

    try:
        obj = load_json_object(raw)
    except ValueError as exc:
        if raw.lstrip().startswith("["):
            raise ValueError("response must be a JSON object") from exc
        if "invalid JSON" not in str(exc):
            raise ValueError(f"invalid JSON: {exc}") from exc
        raise

    allowed = {"update_trust_region", "update_bias", "rationale"}
    require_allowed_keys(obj, allowed)

    return ParsedUpdate(
        update_trust_region=obj.get("update_trust_region"),
        update_bias=obj.get("update_bias"),
        rationale=obj.get("rationale"),
    )
