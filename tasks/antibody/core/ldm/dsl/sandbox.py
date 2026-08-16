"""core/ldm/dsl/sandbox.py — safe_exec_dsl.

Executes LLM-provided Python source in a restricted namespace.
"""
from __future__ import annotations

import ast
from typing import Any

from tasks.antibody.core.ldm.dsl.bias import (
    BiasAtom,
    BiasSum,
    MaxAromatic,
    MaxCysteine,
    MaxHydrophobicRun,
    NetChargeRange,
    NoNGlycosylation,
)
from tasks.antibody.core.ldm.dsl.exceptions import DSLSyntaxError
from tasks.antibody.core.ldm.dsl.search_space import (
    LatinHyperCubeSampling,
    LocalSearch,
    NeighborSampling,
    Or,
    SearchSpaceAtom,
)


def _build_namespace(whitelist: tuple[str, ...]) -> dict[str, Any]:
    candidates: dict[str, type] = {
        "LatinHyperCubeSampling": LatinHyperCubeSampling,
        "NeighborSampling": NeighborSampling,
        "LocalSearch": LocalSearch,
        "Or": Or,
        "MaxCysteine": MaxCysteine,
        "MaxHydrophobicRun": MaxHydrophobicRun,
        "MaxAromatic": MaxAromatic,
        "NetChargeRange": NetChargeRange,
        "NoNGlycosylation": NoNGlycosylation,
        "BiasSum": BiasSum,
    }
    ns: dict[str, Any] = {}
    for name in whitelist:
        if name in candidates:
            ns[name] = candidates[name]
    ns["__builtins__"] = {}
    return ns


def _is_pure_expression(source: str) -> bool:
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError:
        return False
    return len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr)


def safe_exec_dsl(
    source: str,
    whitelist: tuple[str, ...],
    expect_kind: type | None = None,
) -> Any:
    namespace = _build_namespace(whitelist)

    if _is_pure_expression(source):
        try:
            result = eval(source, namespace)
        except Exception as e:
            raise DSLSyntaxError(f"{type(e).__name__}: {e}") from e
        if not isinstance(result, (SearchSpaceAtom, BiasAtom)):
            raise DSLSyntaxError(
                f"Expression evaluated to {type(result).__name__}, "
                f"expected SearchSpaceAtom or BiasAtom."
            )
        if expect_kind is not None and not isinstance(result, expect_kind):
            raise DSLSyntaxError(
                f"Expression evaluated to {type(result).__name__}, "
                f"expected {expect_kind.__name__}."
            )
        return result

    try:
        exec(source, namespace)
    except Exception as e:
        raise DSLSyntaxError(f"{type(e).__name__}: {e}") from e

    atoms = [
        v for v in namespace.values()
        if isinstance(v, (SearchSpaceAtom, BiasAtom))
        and (expect_kind is None or isinstance(v, expect_kind))
    ]
    if len(atoms) == 0:
        raise DSLSyntaxError(
            f"exec succeeded but no atom instance found. "
            f"Available names: {sorted(namespace.keys())}"
        )
    if len(atoms) > 1:
        raise DSLSyntaxError(
            f"exec produced {len(atoms)} atom instances; expected exactly 1."
        )
    return atoms[0]
