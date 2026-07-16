"""Score helpers shared by LDM-TTS task adapters."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

T = TypeVar("T")


def is_finite_number(value: Any) -> bool:
    """Return whether ``value`` can be interpreted as a finite float."""

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def as_float(value: Any) -> float | None:
    """Convert finite numeric values to float, otherwise return ``None``."""

    return float(value) if is_finite_number(value) else None


def finite_or_none(value: Any) -> float | None:
    """Alias used by existing domain code."""

    return as_float(value)


def ranked_items(
    items: Iterable[T],
    score: Callable[[T], Any],
    *,
    minimize: bool = True,
) -> list[T]:
    """Return items with finite scores sorted from best to worst."""

    scored: list[tuple[T, float]] = []
    for item in items:
        value = score(item)
        if is_finite_number(value):
            scored.append((item, float(value)))
    scored.sort(key=lambda pair: pair[1], reverse=not minimize)
    return [item for item, _score in scored]


def best_item(
    items: Iterable[T],
    score: Callable[[T], Any],
    *,
    minimize: bool = True,
) -> T | None:
    """Return the best finite-scored item, or ``None`` if none are usable."""

    ranked = ranked_items(items, score, minimize=minimize)
    return ranked[0] if ranked else None
