"""Resolve supported Absolut installation layouts."""

from __future__ import annotations

from pathlib import Path
from typing import Union


def resolve_absolut_executable(root: Union[str, Path]) -> Path:
    """Return the first supported Absolut executable path under *root*."""
    root_path = Path(root).expanduser().resolve()
    candidates = (
        root_path / "src" / "bin" / "Absolut",
        root_path / "src" / "AbsolutNoLib",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])
