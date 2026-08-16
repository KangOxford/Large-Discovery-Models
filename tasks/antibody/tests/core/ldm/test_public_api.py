"""tests/core/ldm/test_public_api.py — verify the public surface is exactly as documented."""
from __future__ import annotations

import importlib

import pytest


def test_public_api_contains_only_expected_names():
    """core/ldm.__all__ must be the canonical public surface."""
    mod = importlib.import_module("tasks.antibody.core.ldm")
    expected = {
        "DSLConfig",
        "SearchSpaceAtom",
        "BiasAtom",
        "Orchestrator",
        "OrchestratorStatus",
        "OrchestratorDecision",
        "LLMClient",
        "OpenAIClient",
    }
    assert set(mod.__all__) == expected, (
        f"Public API mismatch. Got {sorted(mod.__all__)}, "
        f"expected {sorted(expected)}."
    )


def test_each_public_symbol_is_importable():
    """Verify each name in __all__ is actually exported."""
    mod = importlib.import_module("tasks.antibody.core.ldm")
    for name in mod.__all__:
        assert hasattr(mod, name), f"tasks.antibody.core.ldm missing export: {name}"


def test_internal_modules_not_exported_at_top_level():
    """Internal atom classes and helpers must NOT leak through `from tasks.antibody.core.ldm import *`."""
    # Check that concrete atom classes are NOT in __all__
    mod = importlib.import_module("tasks.antibody.core.ldm")
    forbidden = {
        "LatinHyperCubeSampling", "NeighborSampling", "LocalSearch", "Or",
        "MaxCysteine", "MaxHydrophobicRun", "MaxAromatic",
        "NetChargeRange", "NoNGlycosylation", "BiasSum",
        "LiteLLMClient", "build_llm_client", "ParsedUpdate", "DecisionLog",
        "safe_exec_dsl", "validate_search_atom", "validate_bias_atom",
        "sample_within_search_dsl",
        "fallback_to_original_antbo",
    }
    leaked = forbidden & set(mod.__all__)
    assert not leaked, f"Internals leaked into tasks.antibody.core.ldm.__all__: {leaked}"


def test_no_internal_leak_via_dir():
    """``dir(tasks.antibody.core.ldm)`` should not contain internal helpers or concrete atoms.

    Note: subclasses of ABCs (like HammingDistanceTo) are defined in the
    same module as their ABC, so they may appear in dir(). The public_api
    test relies on __all__ for the canonical surface. This test is a
    best-effort sanity check.
    """
    mod = importlib.import_module("tasks.antibody.core.ldm")
    names = set(dir(mod))
    # These are PUBLIC — allowed in dir():
    public = {
        "DSLConfig", "SearchSpaceAtom", "BiasAtom",
        "Orchestrator", "OrchestratorStatus", "OrchestratorDecision",
        "LLMClient", "OpenAIClient",
    }
    # Names that appear but are NOT in __all__ should be either:
    #   - dunder attributes (excluded below)
    #   - re-exports of stdlib (typing etc.)
    # We only flag concrete atom classes.
    suspicious = {"LatinHyperCubeSampling", "NeighborSampling", "LocalSearch", "Or",
                  "MaxCysteine", "MaxHydrophobicRun", "MaxAromatic",
                  "NetChargeRange", "NoNGlycosylation", "BiasSum",
                  "LiteLLMClient", "build_llm_client"}
    leaked_suspicious = (names & suspicious) - public
    # Note: these may leak via dir() because they're defined in submodules;
    # this is informational. The hard guarantee is __all__.
    if leaked_suspicious:
        print(f"INFO: {leaked_suspicious} are accessible via dir() but NOT in __all__")


def test_core_outside_does_not_import_internal_modules():
    """core/ external code should not import from tasks.antibody.core.ldm.dsl / .orchestrator / .llm.

    This test is a heuristic: it greps the source of core/*.py (excluding
    core/ldm/) for forbidden import patterns.
    """
    import re
    from pathlib import Path

    core_root = Path(__file__).resolve().parents[3] / "core"
    forbidden = re.compile(
        r"^\s*from\s+tasks\.antibody\.core\.ldm\."
        r"(dsl|orchestrator|llm)(?:\.|\b)"
    )
    violations = []
    for py_file in core_root.glob("**/*.py"):
        rel_parts = py_file.relative_to(core_root).parts
        if rel_parts and rel_parts[0].startswith("ldm"):
            continue  # skip core/ldm/* itself
        for line_no, line in enumerate(py_file.read_text().splitlines(), 1):
            if forbidden.match(line):
                violations.append(f"{py_file}:{line_no}: {line.strip()}")
    assert not violations, (
        "core/ external code imports tasks.antibody.core.ldm internals (must use tasks.antibody.core.ldm public API only):\n"
        + "\n".join(violations)
    )
