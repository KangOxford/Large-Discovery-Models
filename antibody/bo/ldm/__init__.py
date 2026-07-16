"""bo/ldm — public package.

This module is the SOLE import path for code outside ``bo/ldm``. It
re-exports only the public surface:

    - DSLConfig                  — config dataclass
    - SearchSpaceAtom            — ABC for trust-region atoms
    - BiasAtom                   — ABC for bias atoms
    - Orchestrator               — main LDM orchestrator
    - OrchestratorStatus         — input to Orchestrator.step
    - OrchestratorDecision       — return value of Orchestrator.step
    - LLMClient                  — abstract LLM backend
    - OpenAIClient               — concrete LLMClient (OpenAI SDK, .env-configured)

Anything else in this subpackage is PRIVATE. In particular:

    - bo.ldm.dsl.*               — concrete atom classes (Neighbor, Or, ...)
    - bo.ldm.orchestrator.prompts — prompt templates
    - bo.ldm.orchestrator.fallback — fallback strategies
    - bo.ldm.orchestrator.decision_log — log writer
    - bo.ldm.llm.response_parser — JSON parser
    - bo.ldm.llm.openai_backend   — (re-exported above as OpenAIClient)

bo/ external code MUST NOT import from private submodules directly. The
enforcement mechanism is the public_api test in tests/bo/ldm/test_public_api.py
which greps for forbidden imports under bo/.
"""
from __future__ import annotations

from bo.ldm.config import DSLConfig
from bo.ldm.dsl.bias import BiasAtom
from bo.ldm.dsl.search_space import SearchSpaceAtom
from bo.ldm.integrate import (
    apply_decision,
    build_status,
    sample_candidates,
    score_with_bias,
)
from bo.ldm.llm.client import LLMClient
from bo.ldm.llm.openai_backend import OpenAIClient
from bo.ldm.orchestrator.loop import Orchestrator, OrchestratorDecision
from bo.ldm.orchestrator.status import OrchestratorStatus

__all__ = [
    "DSLConfig",
    "SearchSpaceAtom",
    "BiasAtom",
    "Orchestrator",
    "OrchestratorStatus",
    "OrchestratorDecision",
    "LLMClient",
    "OpenAIClient",
    "build_status",
    "apply_decision",
    "sample_candidates",
    "score_with_bias",
]