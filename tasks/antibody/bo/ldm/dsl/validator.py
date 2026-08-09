"""bo/ldm/dsl/validator.py — validation for SearchSpaceAtom and BiasAtom."""
from __future__ import annotations

from typing import List

import numpy as np

from bo.ldm.dsl.bias import BiasAtom, BiasSum
from bo.ldm.dsl.exceptions import SamplingTimeout
from bo.ldm.dsl.search_space import Or, SearchSpaceAtom


def _depth(atom) -> int:
    if isinstance(atom, Or):
        return 1 + max((_depth(c) for c in atom.children), default=0)
    return 1


def validate_search_atom(
    atom: SearchSpaceAtom,
    max_depth: int = 8,
    sample_timeout_s: float = 3.0,
) -> List[str]:
    errors: list[str] = []

    try:
        d = _depth(atom)
        if d > max_depth:
            errors.append(f"nesting depth {d} > {max_depth}. Flatten the DSL.")
            return errors
    except RecursionError:
        errors.append("DSL nesting is pathologically deep (RecursionError).")
        return errors

    try:
        rng = np.random.default_rng(42)
        samples = atom.sample(n=10, rng=rng, timeout_s=sample_timeout_s)
        if len(samples) == 0:
            errors.append(
                "Sampling produced 0 sequences — the trust region may be "
                "empty or too restrictive."
            )
    except SamplingTimeout as e:
        errors.append(str(e))
    except NotImplementedError:
        pass  # LocalSearch doesn't support sample — that's OK
    except Exception as e:
        errors.append(f"Sampling error: {type(e).__name__}: {e}")

    return errors


def validate_bias_atom(
    atom: BiasAtom,
    max_depth: int = 4,
) -> List[str]:
    errors: list[str] = []

    def _bias_depth(a: BiasAtom) -> int:
        if isinstance(a, BiasSum):
            return 1 + max((_bias_depth(c) for c in a.atoms), default=0)
        return 1

    try:
        d = _bias_depth(atom)
        if d > max_depth:
            errors.append(f"bias nesting depth {d} > {max_depth}.")
    except RecursionError:
        errors.append("bias nesting pathologically deep.")
    return errors
