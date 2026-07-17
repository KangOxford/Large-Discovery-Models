"""Canonical scoring interface shared by all ``strbo_v1`` backends.

A :data:`Scorer` is a single callable that maps a sequence of SMILES
to a sequence of scores, one per input. The interface is deliberately
minimal so any backend (Vina, a trained neural network, a docking
proxy, an oracle, ...) can plug in by implementing just one method
matching the contract:

    def __call__(self, smiles_list: Sequence[str]) -> Sequence[float]:
        ...

Contract details (enforced by the BO / random-search loops via
``_safe_score``):

- The i-th element of the returned sequence is the score of the i-th
  SMILES in the input sequence. Length must equal
  ``len(smiles_list)``.
- Lower scores are not assumed to be better universally; the LDM-TTS
  loop takes a ``minimize`` flag (default ``True`` for Vina) and the BO
  acquisition functions are written uniformly as "higher = better".
- Failed per-molecule evaluations are signalled with ``float("nan")``. The
  loops convert non-finite floats to ``None`` internally and exclude them from
  the GP fit; the SMILES is still recorded in the history log.
- Infrastructure failures should raise exceptions. The LDM-TTS loop treats
  scorer exceptions as hard run failures so missing docking dependencies,
  broken receptor preparation, or malformed scorer outputs do not masquerade as
  ordinary low-quality candidates.

The concrete implementations live in ``strbo_v1.objective_vina``
(:class:`VinaScorer`, AutoDock Vina with disk cache) and
``strbo_v1.objective_nn`` (:class:`NNScorer`, the G12C pIC50 ensemble
from ``activity_modeling/best_model.joblib``). Both re-export
:data:`Scorer` from this module for convenience; new backends should
do the same.

Multi-objective
---------------
:data:`Scorers` is the multi-objective union of :data:`Scorer` —
either a single scorer or a tuple of scorers. Use :func:`as_scorer_tuple`
to normalise any acceptable input to a tuple form. The search loops
expect the tuple form internally; the public API accepts either
``Scorer`` or ``tuple[Scorer, ...]``.

Reference-point registry
------------------------
The hypervolume and EHVI calculations need a per-objective reference
point. :data:`DEFAULT_REF` is a registry mapping scorer backend names
(``"vina"``, ``"nn"``, ``"mock"``) to their default reference value.
:func:`register_ref` adds a new backend's default at runtime;
:func:`resolve_ref_point` looks up the user-supplied override first,
falls back to the registry, and finally to ``0.0`` for unknown
backends. The LDM-TTS CLI constructs the
``objective_parts`` list from ``--objective`` and passes it to
:func:`resolve_ref_point` to derive the per-run ``ref_point`` tuple.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, TypeAlias, Union


Scorer: TypeAlias = Callable[[Sequence[str]], Sequence[float]]
"""Canonical scoring interface for the BO / random-search loops.

The i-th element of the returned sequence is the score of the i-th
SMILES in the input sequence. Failed evaluations are signalled with
``float("nan")``; the LDM-TTS loop converts non-finite floats to ``None``
internally and exclude them from the GP fit. Any callable matching
this signature is accepted as a ``scorer`` argument.
"""


Scorers: TypeAlias = Union[Scorer, Tuple[Scorer, ...]]
"""Multi-objective union: a single :data:`Scorer` or a tuple of scorers.

Each scorer in the tuple is one objective. The i-th element of the
tuple is the i-th objective's score. The search loops internally
normalise to tuple form via :func:`as_scorer_tuple`; the public API
accepts either a single scorer or a tuple of scorers for backwards
compatibility with single-objective code.
"""


def as_scorer_tuple(scorer: Scorers) -> Tuple[Scorer, ...]:
    """Normalise a single scorer or a tuple of scorers to tuple form.

    A single callable is wrapped as a one-element tuple. A tuple is
    returned as-is after validating that every element is callable.

    Args:
        scorer: Either a single :data:`Scorer` or a tuple thereof.

    Returns:
        A tuple of scorers. Length 1 for single-objective inputs.

    Raises:
        TypeError: If ``scorer`` is not callable, not a tuple, or any
            tuple element is not callable.
    """
    if callable(scorer):
        return (scorer,)
    if isinstance(scorer, tuple):
        for i, s in enumerate(scorer):
            if not callable(s):
                raise TypeError(
                    f"scorer tuple element {i} is not callable: "
                    f"got {type(s).__name__}"
                )
        return scorer
    raise TypeError(
        f"scorer must be callable or tuple of callables; "
        f"got {type(scorer).__name__}"
    )


# ---------------------------------------------------------------------------
# Reference-point registry
# ---------------------------------------------------------------------------

DEFAULT_REF: dict[str, float] = {
    "vina": 0.0,
    "nn": 5.0,
    "mock": 0.0,
}
"""Per-backend default reference point for hypervolume / EHVI.

Units / semantics:
    * ``"vina"`` (kcal/mol): 0.0 is a "neutral" reference; negative
      values mean stronger binding. The actual Vina score range varies
      by target; ``0.0`` is a conservative HV upper bound.
    * ``"nn"`` (pIC50): 5.0 is a literature baseline for weakly active
      compounds (pIC50 = 5 → 10 µM).
    * ``"mock"``: 0.0 (mock scorers span arbitrary ranges; 0 is a
      neutral default that callers can override).

Unknown backend names fall back to ``0.0``; :func:`register_ref` adds
a new backend's default at runtime. Callers such as the LDM-TTS runner
typically never mutate this dict directly; ``register_ref`` is the
public extension point.
"""


def register_ref(objective_name: str, default: float) -> None:
    """Register a default reference point for a new scorer backend.

    Mutates :data:`DEFAULT_REF` in place. The new entry is visible to
    all subsequent :func:`resolve_ref_point` calls.

    Args:
        objective_name: The backend identifier (e.g. ``"my_oracle"``).
        default: The default reference point for this backend.
    """
    if not isinstance(objective_name, str) or not objective_name:
        raise TypeError(
            f"objective_name must be a non-empty str, got "
            f"{type(objective_name).__name__}"
        )
    if not isinstance(default, (int, float)) or isinstance(default, bool):
        raise TypeError(
            f"default must be a real number, got {type(default).__name__}"
        )
    DEFAULT_REF[objective_name] = float(default)


def resolve_ref_point(
    objective_parts: Sequence[str],
    user_ref: Optional[Sequence[float]] = None,
) -> Tuple[float, ...]:
    """Resolve the per-objective reference point for HV / EHVI.

    Resolution order:
        1. ``user_ref`` (if not ``None`` and length matches
           ``len(objective_parts)``) is returned verbatim as floats.
        2. Otherwise, each objective's name is looked up in
           :data:`DEFAULT_REF`; unknown names fall back to ``0.0``.

    Args:
        objective_parts: Objective backend names in order, e.g.
            ``("vina", "nn")`` for ``--objective vina+nn``.
        user_ref: Optional user-supplied override (e.g. parsed from
            ``--ref-point 0,5``). When not ``None`` the length must
            match ``len(objective_parts)``.

    Returns:
        A tuple of reference points, one per objective.

    Raises:
        ValueError: If ``user_ref`` is provided but its length does
            not match ``len(objective_parts)``, or if
            ``objective_parts`` is empty.
    """
    n = len(objective_parts)
    if n == 0:
        raise ValueError("objective_parts is empty; cannot resolve ref_point")
    if user_ref is not None:
        if len(user_ref) != n:
            raise ValueError(
                f"user_ref length ({len(user_ref)}) does not match "
                f"objective_parts length ({n})"
            )
        return tuple(float(x) for x in user_ref)
    return tuple(DEFAULT_REF.get(name, 0.0) for name in objective_parts)


__all__ = [
    "DEFAULT_REF",
    "Scorer",
    "Scorers",
    "as_scorer_tuple",
    "register_ref",
    "resolve_ref_point",
]
