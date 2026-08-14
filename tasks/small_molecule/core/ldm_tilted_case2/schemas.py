"""JSON parsers for case2 tilted LDM methods."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ldm_tts.transport.parsing import (
    load_json_object,
    reject_keys,
    require_list,
    require_nonnegative_int,
    require_str,
)


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
    data = load_json_object(text)
    _reject_banned_score_keys(data)
    rows = require_list(data, "direct_smiles")
    return DirectSmilesPlan([DirectSmilesItem(require_str(row, "smiles"), str(row.get("rationale", ""))) for row in rows])


def parse_seed_plan(text: str) -> SeedPlan:
    data = load_json_object(text)
    _reject_banned_score_keys(data)
    rows = require_list(data, "seeds")
    return SeedPlan([
        SeedItem(require_str(row, "smiles"), require_nonnegative_int(row, "budget"), str(row.get("intent", "")))
        for row in rows
    ])


def _reject_banned_score_keys(data: Any, *, extra: set[str] | None = None) -> None:
    reject_keys(data, BANNED_SCORE_KEYS | (extra or set()))
