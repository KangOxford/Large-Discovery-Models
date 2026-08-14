"""Versioned source representation for GP-guided quantizer selection."""

from __future__ import annotations

import ast

from ldm_tts.contracts import Candidate, SurrogateSpaceSpec
from ldm_tts.optimization.records import SurrogateVector


FEATURE_VERSION = "quantizer_ast_v1"
FEATURE_DIMENSION = 18


class QuantizerSourceEncoder:
    def describe(self) -> SurrogateSpaceSpec:
        return SurrogateSpaceSpec(
            kind="vector",
            representation=(
                "18 normalized AST and quantization-policy indicators derived from "
                "the admitted AdaptiveKVQuantizer source."
            ),
            dimension_policy="fixed",
            dimension=FEATURE_DIMENSION,
            encoder=(
                "tasks.llm_kv_adaptive_quantization.core.surrogate:"
                "QuantizerSourceEncoder"
            ),
            version=FEATURE_VERSION,
        )

    def encode(self, candidate: Candidate) -> SurrogateVector:
        source = str(candidate.payload["code"])
        tree = ast.parse(source)
        nodes = tuple(ast.walk(tree))
        names = {
            node.id.lower()
            for node in nodes
            if isinstance(node, ast.Name)
        } | {
            node.attr.lower()
            for node in nodes
            if isinstance(node, ast.Attribute)
        }
        constants = [
            float(node.value)
            for node in nodes
            if isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ]
        values = (
            min(len(source.encode("utf-8")) / 64_000.0, 1.0),
            min(len(nodes) / 2_000.0, 1.0),
            min(sum(isinstance(node, ast.FunctionDef) for node in nodes) / 20.0, 1.0),
            min(sum(isinstance(node, ast.Call) for node in nodes) / 100.0, 1.0),
            min(sum(isinstance(node, ast.If) for node in nodes) / 30.0, 1.0),
            min(sum(isinstance(node, (ast.For, ast.While)) for node in nodes) / 20.0, 1.0),
            min(sum(isinstance(node, ast.Dict) for node in nodes) / 20.0, 1.0),
            min(sum(isinstance(node, ast.Subscript) for node in nodes) / 100.0, 1.0),
            float(any(value == 2 for value in constants)),
            float(any(value == 3 for value in constants)),
            float(any(value == 4 for value in constants)),
            float(any(value == 8 for value in constants)),
            float(any(value == 16 for value in constants)),
            float(any("residual" in name for name in names)),
            float(any("group" in name for name in names)),
            float(any("svd" in name or "subspace" in name for name in names)),
            float(any("observer" in name or "observe" in name for name in names)),
            float(any("layer" in name or "preset" in name for name in names)),
        )
        return SurrogateVector(
            values=tuple(float(value) for value in values),
            version=FEATURE_VERSION,
            source_id=candidate.candidate_id,
            metadata={"encoder": "normalized_ast_policy_indicators"},
        )
