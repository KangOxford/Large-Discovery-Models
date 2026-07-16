"""Shared utilities for the ``strbo_v1`` search loops."""

from __future__ import annotations

from collections import deque
from typing import Iterator, Optional


class FIFOSet:
    """A FIFO-ordered collection with O(1) membership.

    Combines a ``collections.deque`` (insertion-ordered, optionally
    bounded with auto-eviction of the oldest entry on append) with a
    ``set`` (O(1) membership and dedup). All mutators keep the two
    structures in sync.

    Args:
        max_size: Optional FIFO cap. When set, the underlying deque has
            ``maxlen=max_size`` and ``append()`` auto-evicts the oldest
            entry. ``None`` is unbounded.

    Use as a pending-candidate queue in BO / random search loops: the
    next candidates to score are at the front of the queue; new
    analogues are appended to the back. When ``max_size`` is set, the
    oldest entry is automatically evicted on append, bounding memory
    and ensuring older candidates eventually time out.

    Iteration is FIFO (insertion order). Pickup is either ``popleft``
    (true FIFO) or ``rng.sample(list(...), k=k)`` (random sample of
    the current queue).
    """

    def __init__(self, max_size: Optional[int] = None) -> None:
        if max_size is not None and max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self._queue: deque[str] = (
            deque(maxlen=max_size) if max_size is not None else deque()
        )
        self._set: set[str] = set()

    def add(self, smiles: str) -> bool:
        """Append ``smiles`` to the back of the queue.

        Returns ``True`` if added, ``False`` if already present (dedup).
        When the queue is bounded, the oldest entry is auto-evicted.
        """
        if smiles in self._set:
            return False
        # When the bounded queue is at capacity, the next append will
        # auto-evict the oldest entry. Remove it from the set first so
        # the queue and set stay in sync.
        if self._queue.maxlen is not None and len(self._queue) >= self._queue.maxlen:
            self._set.discard(self._queue[0])
        self._set.add(smiles)
        self._queue.append(smiles)
        return True

    def discard(self, smiles: str) -> None:
        """Remove ``smiles`` if present. No-op if absent. O(n)."""
        if smiles not in self._set:
            return
        self._set.discard(smiles)
        try:
            self._queue.remove(smiles)
        except ValueError:
            pass  # invariant: queue and set must stay in sync

    def popleft(self) -> str:
        """Remove and return the oldest entry. O(1)."""
        if not self._queue:
            raise IndexError("popleft from empty FIFOSet")
        smi = self._queue.popleft()
        self._set.discard(smi)
        return smi

    def __contains__(self, smiles: object) -> bool:
        return smiles in self._set

    def __iter__(self) -> Iterator[str]:
        return iter(self._queue)

    def __len__(self) -> int:
        return len(self._set)

    def __bool__(self) -> bool:
        return bool(self._set)

    def __repr__(self) -> str:
        return f"FIFOSet({list(self._queue)}, max_size={self._queue.maxlen})"

    @property
    def max_size(self) -> Optional[int]:
        """FIFO cap, or ``None`` if unbounded."""
        return self._queue.maxlen


# ---------------------------------------------------------------------------
# SMILES validation / canonicalization
# ---------------------------------------------------------------------------


def canonicalize_smiles_strict(text: str) -> str:
    """Validate and canonicalize a SMILES string.

    Requires RDKit (a hard dependency of the project). Empty /
    whitespace-only input raises :class:`ValueError`; any SMILES
    that RDKit cannot parse also raises :class:`ValueError` with
    the offending raw text in the message.

    RDKit's stderr parse warnings (``rdApp.*``) are silenced for the
    lifetime of the process — the same pattern used in
    :mod:`strbo_v1.gp` — so the caller's :class:`ValueError` is the
    only signal the user sees for a bad SMILES.

    Args:
        text: Raw SMILES string. Leading / trailing whitespace is
            stripped before validation.

    Returns:
        Canonical SMILES (via ``Chem.MolToSmiles(mol, canonical=True)``).

    Raises:
        ValueError: on empty input, on RDKit parse failure, or on
            ``None`` return from ``Chem.MolFromSmiles``.
    """
    text = str(text or "").strip()
    if not text:
        raise ValueError("empty SMILES")
    from rdkit import Chem, RDLogger  # local import; RDKit is a hard dep

    # Silence RDKit's stderr parse warnings; matches strbo_v1.gp pattern.
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise ValueError(f"invalid SMILES: {text!r}")
    return Chem.MolToSmiles(mol, canonical=True)


__all__ = ["FIFOSet", "canonicalize_smiles_strict"]
