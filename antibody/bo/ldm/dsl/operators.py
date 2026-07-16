"""Operator overloads for DSL atoms.

``SearchSpaceAtom`` supports ``|`` (Or) so LLM output can use natural
Pythonic operators::

    LocalSearch("ARYY...", radius=2, restart=2, steps=100)
    LocalSearch("ARYY...", radius=2, restart=2, steps=100) | LocalSearch("VRG...", radius=3, restart=1, steps=80)
"""
from __future__ import annotations

from bo.ldm.dsl.search_space import Or, SearchSpaceAtom

__all__ = ["Or"]


def _bind_operators() -> None:
    def __or__(self, other):
        if isinstance(other, SearchSpaceAtom):
            return Or(self, other)
        return NotImplemented

    SearchSpaceAtom.__or__ = __or__


_bind_operators()
