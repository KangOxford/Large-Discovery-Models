"""Public re-exports for AntBO kernel helpers.

The kernel module imports Torch, so resolve these names lazily. This keeps
lighter AntBO modules importable for CLI parsing and mock checks in
environments that have not installed the full GP stack yet.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "CategoricalOverlap",
    "TransformedCategorical",
    "OrdinalKernel",
    "FastStringKernel",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("tasks.antibody.core.kernels"), name)
    globals()[name] = value
    return value
