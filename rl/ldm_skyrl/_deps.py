"""Lazy dependency resolution with phase-numbered failures.

Two rules hold everywhere in this package.

**Importable without SkyRL.**  ``rl/ldm_rl`` already works this way: its core is
importable without slime installed, and its unit tests inject fakes instead.
Keeping that property means the whole of Phase B can be written and tested on a
login node while Phase A is still deciding whether the stack installs at all.

**Never fail silently, and never fail vaguely.**  A bare ``ImportError`` tells a
reader that something is missing, not which gate has not been passed.  Every
failure here names the phase item and quotes the criterion recorded in
``phases.json`` before any measurement was taken.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

_PHASES_PATH = Path(__file__).with_name("phases.json")


class PhaseGateError(RuntimeError):
    """The code path is wired, but the phase item that would prove it has not run."""


@functools.lru_cache(maxsize=1)
def _phase_items() -> dict[str, dict[str, Any]]:
    data = json.loads(_PHASES_PATH.read_text())
    return {item["id"]: item for item in data["items"]}


def phase_item(item_id: str) -> dict[str, Any]:
    try:
        return _phase_items()[item_id]
    except KeyError:  # pragma: no cover - only reachable via a typo in this package
        raise PhaseGateError(
            f"{item_id!r} is not a phase item; known ids: {sorted(_phase_items())}"
        ) from None


def gate(item_id: str, detail: str = "") -> PhaseGateError:
    """Build the exception for an unpassed gate.  Callers ``raise`` the result."""
    item = phase_item(item_id)
    lines = [
        f"phase item {item['id']} ({item['phase']}) has not passed: {item['title']}",
        f"  criterion: {item['criterion']}",
        f"  artifact:  {item['artifact']}",
        f"  budget:    {item['budget']}",
    ]
    if item["deps"]:
        lines.append(f"  depends on: {', '.join(item['deps'])}")
    if detail:
        lines.append(f"  detail:    {detail}")
    return PhaseGateError("\n".join(lines))


def require_skyrl(item_id: str = "A1") -> Any:
    """Import ``skyrl.train``, or explain which gate is missing."""
    try:
        import skyrl.train as skyrl_train
    except ImportError as exc:
        raise gate(
            item_id,
            "skyrl is not importable in this interpreter. Pin 0.3.0 (commit b8a5caaa); "
            "SkyRL main needs CUDA 13.0 and driver r580, and this cluster runs 12.7 on 565.",
        ) from exc
    return skyrl_train


def require_ldm_rl(item_id: str = "B1") -> Any:
    """Import the framework-neutral LDM environment package.

    ``rl/ldm_rl`` lives on the ``rl`` line, not on ``main``.  This package sits on
    ``main`` so the roadmap is visible on the default branch, which means the
    import has to be deferred and the failure has to say so.
    """
    try:
        import ldm_rl
    except ImportError as exc:
        raise gate(
            item_id,
            "ldm_rl is not importable. It is published on the rl line "
            "(branch 'rl', and its tip 'pr/reward-zero-fixes'), not on main. "
            "Put <repo>/rl on PYTHONPATH from a checkout of that branch.",
        ) from exc
    return ldm_rl


def generator_base() -> type:
    """SkyRL's ``GeneratorInterface`` when it is installed, else a local stand-in.

    Subclassing the real interface matters: SkyRL's trainer type-checks nothing,
    but the abstract method list is the contract, and inheriting it means a
    signature drift in a future pin shows up as a TypeError at construction
    rather than as a silently unused override.
    """
    try:
        from skyrl.train.generators.base import GeneratorInterface
    except ImportError:
        import abc

        class _OfflineGeneratorInterface(abc.ABC):
            """Stand-in used only when skyrl is absent, so tests can run on CPU."""

            @abc.abstractmethod
            async def generate(self, input_batch):  # noqa: D102
                raise NotImplementedError

        return _OfflineGeneratorInterface
    return GeneratorInterface


__all__ = [
    "PhaseGateError",
    "gate",
    "generator_base",
    "phase_item",
    "require_ldm_rl",
    "require_skyrl",
]
