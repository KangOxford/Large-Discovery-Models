"""JSON parsers for case2 tilted LDM methods."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any


BANNED_SCORE_KEYS = {
    "score",
    "objective_score",
    "constraint_score",
    "acquisition_score",
    "uncertainty",
    "proxy_value",
}


@dataclass
class DirectSmilesItem:
    smiles: str
    rationale: str = ""


@dataclass
class DirectSmilesPlan:
    direct_smiles: list[DirectSmilesItem]

    def to_dict(self) -> dict[str, Any]:
        return {"direct_smiles": [asdict(item) for item in self.direct_smiles]}


@dataclass
class SeedItem:
    smiles: str
    budget: int
    intent: str = ""


@dataclass
class SeedPlan:
    seeds: list[SeedItem]

    def to_dict(self) -> dict[str, Any]:
        return {"seeds": [asdict(item) for item in self.seeds]}


def parse_m1_direct_smiles(text: str) -> DirectSmilesPlan:
    data = _load_json_object(text)
    _reject_banned_score_keys(data)
    rows = _require_list(data, "direct_smiles")
    return DirectSmilesPlan([DirectSmilesItem(_require_str(row, "smiles"), str(row.get("rationale", ""))) for row in rows])


def parse_seed_plan(text: str) -> SeedPlan:
    data = _load_json_object(text)
    _reject_banned_score_keys(data)
    rows = _require_list(data, "seeds")
    return SeedPlan([
        SeedItem(_require_str(row, "smiles"), _require_nonnegative_int(row, "budget"), str(row.get("intent", "")))
        for row in rows
    ])


def _load_json_object(text: str) -> dict[str, Any]:
    payload = _extract_json(text)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return data


def _extract_json(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object found")
    return text[start : end + 1]


def _reject_banned_score_keys(data: Any, *, extra: set[str] | None = None) -> None:
    banned = BANNED_SCORE_KEYS | (extra or set())
    if isinstance(data, dict):
        for key, value in data.items():
            if key in banned:
                raise ValueError(f"LLM output may not include {key!r}")
            _reject_banned_score_keys(value, extra=extra)
    elif isinstance(data, list):
        for item in data:
            _reject_banned_score_keys(item, extra=extra)


def _require_list(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key!r} must be a list")
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"{key!r} entries must be objects")
    return value


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key!r} must be a non-empty string")
    return value.strip()


def _require_number(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key!r} must be a number")
    return float(value)


def _require_nonnegative_int(data: dict[str, Any], key: str) -> int:
    value = _require_number(data, key)
    if value < 0 or int(value) != value:
        raise ValueError(f"{key!r} must be a non-negative integer")
    return int(value)
