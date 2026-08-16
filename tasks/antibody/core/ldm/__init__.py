"""core/ldm — public package.

This module is the SOLE import path for code outside ``core/ldm``. It
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

    - tasks.antibody.core.ldm.dsl.*               — concrete atom classes (Neighbor, Or, ...)
    - tasks.antibody.core.ldm.orchestrator.prompts — prompt templates
    - tasks.antibody.core.ldm.orchestrator.fallback — fallback strategies
    - tasks.antibody.core.ldm.orchestrator.decision_log — log writer
    - tasks.antibody.core.ldm.llm.response_parser — JSON parser
    - tasks.antibody.core.ldm.llm.openai_backend   — (re-exported above as OpenAIClient)

core/ external code MUST NOT import from private submodules directly. The
enforcement mechanism is the public_api test in tests/core/ldm/test_public_api.py
which greps for forbidden imports under core/.
"""
from __future__ import annotations

from tasks.antibody.core.ldm.config import DSLConfig
from tasks.antibody.core.ldm.dsl.bias import BiasAtom
from tasks.antibody.core.ldm.dsl.search_space import SearchSpaceAtom
from tasks.antibody.core.ldm.llm.client import LLMClient
from tasks.antibody.core.ldm.llm.openai_backend import OpenAIClient
from tasks.antibody.core.ldm.orchestrator.loop import Orchestrator, OrchestratorDecision
from tasks.antibody.core.ldm.orchestrator.status import OrchestratorStatus

__all__ = [
    "DSLConfig",
    "SearchSpaceAtom",
    "BiasAtom",
    "Orchestrator",
    "OrchestratorStatus",
    "OrchestratorDecision",
    "LLMClient",
    "OpenAIClient",
]
