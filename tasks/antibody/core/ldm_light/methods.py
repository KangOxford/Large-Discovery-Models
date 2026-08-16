"""Stable names and semantics for antibody LDM publication methods."""
from __future__ import annotations


METHOD_SPECS = {
    "direct_max": {"base_measure": "direct", "reduction": "max", "uses_acquisition": True},
    "direct_softmax": {"base_measure": "direct", "reduction": "softmax", "uses_acquisition": True},
    "policy_max": {"base_measure": "policy", "reduction": "max", "uses_acquisition": True},
    "policy_softmax": {"base_measure": "policy", "reduction": "softmax", "uses_acquisition": True},
    "llm_gen": {"base_measure": "direct", "reduction": "none", "uses_acquisition": False},
    "legacy_policy_max": {"base_measure": "legacy_policy", "reduction": "max", "uses_acquisition": True},
}
METHOD_CHOICES = tuple(METHOD_SPECS)


def normalize_method(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "directmax": "direct_max",
        "directsoftmax": "direct_softmax",
        "policymax": "policy_max",
        "policysoftmax": "policy_softmax",
        "pure_llm": "llm_gen",
        "llm": "llm_gen",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in METHOD_SPECS:
        raise ValueError(
            f"Unknown antibody method {value!r}; choose one of {METHOD_CHOICES}"
        )
    return normalized
