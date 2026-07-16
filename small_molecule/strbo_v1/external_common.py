"""Shared helpers for external structured interfaces."""

from __future__ import annotations

from typing import Any, Optional, Sequence


def normalized_request(request: Optional[dict[str, Any]]) -> dict[str, Any]:
    if request is None:
        return {}
    return {str(k).replace("-", "_"): v for k, v in request.items()}


def smiles_from(
    smiles: Optional[Sequence[str]],
    request: dict[str, Any],
    *,
    key: str = "smiles",
) -> list[str]:
    values = smiles if smiles is not None else request.get(key, [])
    if isinstance(values, str):
        values = [s.strip() for s in values.split(",")]
    return [str(s).strip() for s in values if str(s).strip()]


def item(
    smiles: str,
    *,
    ok: bool,
    value: Optional[float],
    error: Optional[str],
    details: dict[str, Any],
) -> dict[str, Any]:
    return {"smiles": smiles, "ok": ok, "value": value, "error": error, "details": details}


def ok_response(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"ok": True, "items": items, "errors": []}
