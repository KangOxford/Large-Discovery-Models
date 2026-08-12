"""Protein-specific candidate encoding and Gaussian-process reranking."""

from __future__ import annotations

import ast
import hashlib
import math
import re
from typing import Any

import numpy as np

from ldm_tts.optimization.gp import GPPrediction, RBFGPSurrogate, SearchObservation


FEATURE_VERSION = "protein_ast_config_hash_v1"
OVERRIDE_NAMES = ("learning_rate", "dropout", "num_encoder_layers", "batch_size")
HELPER_NAMES = ("_rbf", "_dihedrals", "_orientations", "knn_graph")
MODULE_NAMES = (
    "Linear",
    "LayerNorm",
    "Dropout",
    "ReLU",
    "GELU",
    "SiLU",
    "Sequential",
    "ModuleList",
    "Embedding",
    "MultiheadAttention",
)
HASH_DIMS = 16
SURROGATE_DIMENSION = (
    2 * len(OVERRIDE_NAMES)
    + 9
    + len(HELPER_NAMES)
    + len(MODULE_NAMES)
    + 4
    + 2
    + HASH_DIMS
)


def encode_candidate(
    code: str,
    *,
    config_overrides: dict[str, Any] | None = None,
    parameter_count: float | None = None,
) -> tuple[float, ...]:
    """Encode one validated candidate into a compact, deterministic vector."""

    tree = ast.parse(code)
    overrides = config_overrides or {}
    vector: list[float] = []
    for name in OVERRIDE_NAMES:
        raw = overrides.get(name)
        vector.append(_signed_log1p(float(raw)) if raw is not None else 0.0)
        vector.append(1.0 if raw is not None else 0.0)

    calls = [_call_name(node) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    call_counts = {name: calls.count(name) for name in set(calls) if name}
    structural_counts = (
        sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree)),
        sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree)),
        len(calls),
        sum(isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add) for node in ast.walk(tree)),
        sum(isinstance(node, (ast.For, ast.ListComp)) for node in ast.walk(tree)),
        sum(isinstance(node, ast.Subscript) for node in ast.walk(tree)),
        sum(isinstance(node, ast.Attribute) for node in ast.walk(tree)),
        len(list(ast.walk(tree))),
        len(code.splitlines()),
    )
    vector.extend(math.log1p(value) for value in structural_counts)
    vector.extend(math.log1p(call_counts.get(name, 0)) for name in HELPER_NAMES)
    vector.extend(math.log1p(call_counts.get(name, 0)) for name in MODULE_NAMES)
    vector.extend(
        math.log1p(sum(1 for item in calls if item in names))
        for names in (
            {"cat", "stack"},
            {"sum", "mean", "max"},
            {"normalize", "softmax", "log_softmax"},
            {"matmul", "bmm", "einsum"},
        )
    )
    vector.append(0.0 if parameter_count is None else math.log1p(float(parameter_count)))
    vector.append(0.0 if parameter_count is None else 1.0)
    vector.extend(_hashed_tokens(code, HASH_DIMS))
    return tuple(float(value) for value in vector)


def _call_name(node: ast.Call) -> str:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def _hashed_tokens(code: str, dimensions: int) -> list[float]:
    vector = np.zeros(dimensions, dtype=float)
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[-+]?\d+(?:\.\d+)?", code)
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        vector[(value >> 1) % dimensions] += 1.0 if value & 1 else -1.0
    return (vector / math.sqrt(max(1, len(tokens)))).tolist()


def _signed_log1p(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value)


__all__ = [
    "FEATURE_VERSION",
    "GPPrediction",
    "RBFGPSurrogate",
    "SearchObservation",
    "SURROGATE_DIMENSION",
    "encode_candidate",
]
