"""Candidate parsing and admission for AdaptiveKVQuantizer implementations."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from ldm_tts.contracts import Candidate, CandidateRejection, RawProposal
from ldm_tts.data import DataCollectionSink, make_complete_design_ir


REQUIRED_METHODS = frozenset({
    "reset_request",
    "needs_prefill_qkv_observer",
    "query_observation_position",
    "observe_prefill_qkv",
    "quantize_key",
    "quantize_value",
    "estimate_bits",
})
REQUIRED_SIGNATURES = {
    "reset_request": ("self", "request_meta", "budget_state"),
    "needs_prefill_qkv_observer": ("self",),
    "query_observation_position": ("self",),
    "observe_prefill_qkv": (
        "self",
        "layer_id",
        "query_states",
        "key_states",
        "value_states",
        "attention_meta",
    ),
    "quantize_key": ("self", "layer_id", "key_states", "cache_meta"),
    "quantize_value": ("self", "layer_id", "value_states", "cache_meta"),
    "estimate_bits": (
        "self",
        "layer_id",
        "kv_kind",
        "seq_len",
        "head_dim",
        "cache_meta",
    ),
}
FORBIDDEN_CALLS = frozenset(
    {
        "compile",
        "eval",
        "exec",
        "open",
        "__import__",
        "breakpoint",
        "input",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
    }
)


def extract_candidate_source(payload: Any) -> str:
    if isinstance(payload, dict):
        payload = payload.get("code", payload.get("candidate", ""))
    if not isinstance(payload, str):
        raise ValueError("proposal must be source text or an object with a code field")
    text = payload.strip()
    fenced = re.fullmatch(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    if not text:
        raise ValueError("candidate source is empty")
    return text + "\n"


def validate_candidate_source(source: str) -> ast.ClassDef:
    if len(source.encode("utf-8")) > 64_000:
        raise ValueError("candidate source exceeds the 64 KiB admission limit")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"candidate has invalid Python syntax at line {exc.lineno}") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.ClassDef):
        raise ValueError("candidate must contain exactly one top-level class")
    class_node = tree.body[0]
    if class_node.name != "AdaptiveKVQuantizer":
        raise ValueError("candidate class must be named AdaptiveKVQuantizer")
    if class_node.bases or class_node.keywords or class_node.decorator_list:
        raise ValueError("candidate class may not use bases, metaclasses, or decorators")
    method_nodes = [
        node for node in class_node.body if isinstance(node, ast.FunctionDef)
    ]
    methods = {node.name: node for node in method_nodes}
    if len(methods) != len(method_nodes):
        raise ValueError("candidate may not define duplicate method names")
    missing = sorted(REQUIRED_METHODS - methods.keys())
    if missing:
        raise ValueError("candidate is missing required method(s): " + ", ".join(missing))
    for name, expected in REQUIRED_SIGNATURES.items():
        method = methods[name]
        actual = tuple(argument.arg for argument in method.args.args)
        if (
            actual != expected
            or method.args.posonlyargs
            or method.args.kwonlyargs
            or method.args.vararg
            or method.args.kwarg
            or method.args.defaults
            or any(item is not None for item in method.args.kw_defaults)
            or method.decorator_list
        ):
            raise ValueError(
                f"candidate method {name} must have signature ({', '.join(expected)})"
            )
    for node in ast.walk(class_node):
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.Global,
                ast.Nonlocal,
                ast.AsyncFunctionDef,
                ast.Await,
                ast.Yield,
                ast.YieldFrom,
            ),
        ):
            raise ValueError(f"candidate may not contain {type(node).__name__}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("candidate may not access dunder attributes")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("candidate may not access dunder names")
        if isinstance(node, ast.arg) and node.arg.startswith("__"):
            raise ValueError("candidate may not declare dunder arguments")
        if (
            isinstance(node, ast.FunctionDef)
            and node.name.startswith("__")
            and node.name != "__init__"
        ):
            raise ValueError("candidate may not declare dunder methods")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                raise ValueError(f"candidate may not call {node.func.id}")
    return class_node


def canonical_source_key(source: str) -> str:
    tree = ast.parse(source)
    return hashlib.sha256(
        ast.dump(tree, include_attributes=False).encode("utf-8")
    ).hexdigest()


@dataclass
class QuantizerCandidateDomain:
    """Admit complete quantizer classes and collect only accepted proposals."""

    sink: DataCollectionSink = field(default_factory=DataCollectionSink.disabled)

    def admit(self, proposal: RawProposal) -> Candidate | CandidateRejection:
        try:
            source = extract_candidate_source(proposal.payload)
            validate_candidate_source(source)
            key = canonical_source_key(source)
        except ValueError as exc:
            return CandidateRejection("invalid_quantizer", str(exc), proposal.source)
        candidate = Candidate(
            candidate_id=f"quantizer-{key[:12]}",
            payload={"code": source},
            canonical_key=key,
            source=proposal.source,
            metadata={
                "source_bytes": len(source.encode("utf-8")),
                **{
                    name: proposal.metadata[name]
                    for name in (
                        "requested_spec",
                        "proposal_spec",
                        "proposal_repaired",
                    )
                    if name in proposal.metadata
                },
            },
        )
        if bool(proposal.metadata.get("collectable")):
            ir = make_complete_design_ir(
                task_id="llm_kv_adaptive_quantization",
                domain="adaptive KV-cache quantizer Python class",
                task_description=(
                    "Design AdaptiveKVQuantizer under the pinned MLS-Bench "
                    "editable-region contract."
                ),
                objectives=[
                    {
                        "name": "official_score",
                        "direction": "maximize",
                        "description": "Official MLS-Bench aggregate score.",
                    }
                ],
                observations=[],
                candidates=[{"code": source}],
                design_space_description=(
                    "One complete class replacing only official editable lines 41-172."
                ),
                request_description="Propose one contract-valid quantizer class.",
                num_candidates=1,
                reasoning_available=False,
            )
            self.sink.append(
                ir,
                provenance={
                    "candidate_id": candidate.candidate_id,
                    "source": proposal.source,
                },
            )
        return candidate


def parse_proposal_response(text: str) -> list[dict[str, str]]:
    raw = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return [{"code": extract_candidate_source(text)}]
    if isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
        payload = payload["candidates"]
    elif isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not payload:
        raise ValueError("proposal response must contain at least one candidate")
    return [{"code": extract_candidate_source(item)} for item in payload]
