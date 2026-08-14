"""Qualitative target/property context for M1 prompts."""

from __future__ import annotations

from typing import Any


ROLE_ORDER = (
    ("pareto_front", "balanced observed tradeoff"),
    ("top_low_vina", "low docking"),
    ("top_high_activity", "high activity"),
    ("balanced_elites", "balanced observed tradeoff"),
    ("recent_selected", "recently selected"),
    ("failures", "failed evaluation"),
)
MAX_CONTEXT_ROWS = 24
MAX_FAILURE_CONTEXT_ROWS = 3


def build_m1_molecule_context(summary: dict[str, Any], max_rows: int = MAX_CONTEXT_ROWS) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for role, outcome in ROLE_ORDER:
        items = list(summary.get(role) or [])
        if role == "failures":
            items = items[:MAX_FAILURE_CONTEXT_ROWS]
        for item in items:
            smiles = str(item.get("smiles", "")).strip() if isinstance(item, dict) else ""
            if not smiles or smiles in seen:
                continue
            seen.add(smiles)
            rows.append(_context_row(smiles, role, outcome))
            if len(rows) >= max_rows:
                return rows
    return rows


def _context_row(smiles: str, role: str, outcome: str) -> dict[str, str]:
    return {
        "smiles": smiles,
        "history_role": role,
        "observed_outcome": outcome,
        "size_class": _size_class(smiles),
        "shape_hint": _shape_hint(smiles),
        "ring_pattern": _ring_pattern(smiles),
        "heteroatom_pattern": _heteroatom_pattern(smiles),
        "polarity_hint": _polarity_hint(smiles),
        "hbond_pattern": _hbond_pattern(smiles),
        "flexibility_hint": _flexibility_hint(smiles),
        "target_relevance_notes": _target_relevance_notes(smiles),
        "proposal_lesson": _proposal_lesson(role, smiles),
    }


def _size_class(smiles: str) -> str:
    length = len(smiles)
    if length <= 5:
        return "very small"
    if length <= 14:
        return "small"
    if length <= 36:
        return "medium"
    if length <= 72:
        return "large"
    return "oversized-risk"


def _shape_hint(smiles: str) -> str:
    if _aromatic_count(smiles) >= 6:
        return "aromatic-rich"
    if _has_ring(smiles):
        return "compact cyclic"
    if smiles.count("(") >= 2:
        return "branched"
    if len(smiles) >= 18:
        return "linear"
    return "mixed"


def _ring_pattern(smiles: str) -> str:
    ring_digits = sum(char.isdigit() for char in smiles)
    if ring_digits == 0:
        return "no ring"
    if _aromatic_count(smiles) >= 6 and ring_digits >= 4:
        return "multiple aromatic rings"
    if _aromatic_count(smiles) >= 3:
        return "aromatic ring"
    if ring_digits >= 4:
        return "multiple rings"
    return "aliphatic ring"


def _heteroatom_pattern(smiles: str) -> str:
    n_count = _count_any(smiles, "Nn")
    o_count = _count_any(smiles, "Oo")
    if "S" in smiles or "s" in smiles:
        return "sulfur-containing"
    if any(token in smiles for token in ("Cl", "Br", "I", "F")):
        return "halogenated"
    if n_count >= 2 and o_count == 0:
        return "N-rich"
    if o_count >= 2 and n_count == 0:
        return "O-rich"
    if n_count and o_count:
        return "mixed N/O"
    return "few heteroatoms"


def _polarity_hint(smiles: str) -> str:
    hetero_count = _count_any(smiles, "NnOoSs")
    if _count_any(smiles, "Nn") >= 1:
        return "likely ionizable"
    if hetero_count >= 4:
        return "high"
    if hetero_count >= 1:
        return "moderate"
    return "low"


def _hbond_pattern(smiles: str) -> str:
    n_o_count = _count_any(smiles, "NnOo")
    if n_o_count >= 4:
        return "polar-dense"
    if n_o_count >= 2:
        return "donor-and-acceptor"
    if n_o_count == 1:
        return "weak H-bonding"
    return "weak H-bonding"


def _flexibility_hint(smiles: str) -> str:
    if _has_ring(smiles) and len(smiles) <= 24:
        return "rigid"
    if len(smiles) >= 32 and smiles.count("(") <= 2:
        return "highly flexible"
    return "moderately flexible"


def _target_relevance_notes(smiles: str) -> str:
    if _has_ring(smiles) or _aromatic_count(smiles) >= 3:
        return "switch-II-pocket shape-compatibility hypothesis"
    if _count_any(smiles, "NnOoSs") >= 2:
        return "polar-contact compatibility hypothesis"
    return "small-molecule growth hypothesis"


def _proposal_lesson(role: str, smiles: str) -> str:
    if role == "failures":
        return "avoid repeating this exact pattern"
    if role == "top_low_vina":
        return "preserve broad shape while varying polarity"
    if role == "top_high_activity":
        return "use as parent but diversify size"
    if role == "recent_selected":
        return "avoid near-copying recent edits"
    if _has_ring(smiles):
        return "use as scaffold neighbor anchor"
    return "use as local parent with varied heteroatoms"


def _has_ring(smiles: str) -> bool:
    return any(char.isdigit() for char in smiles)


def _aromatic_count(smiles: str) -> int:
    return _count_any(smiles, "cnosp")


def _count_any(smiles: str, chars: str) -> int:
    return sum(1 for char in smiles if char in chars)
