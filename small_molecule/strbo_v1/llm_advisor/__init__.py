"""LLM advisor public re-exports kept for the LDM-TTS loop.

The legacy advisor/orchestrator/parser stack has been removed from this
repository. This package initializer now exposes only the block dataclasses,
client implementations, and environment configuration used by the current
test-time-search workflow.
"""

from strbo_v1.llm_advisor.blocks import (
    AnalogueVerdict,
    AnalogBlock,
    GeneratorHint,
    LLMBlock,
    NoopBlock,
    PHASE_A_ACTIONS_ALLOWED,
    PHASE_A_REVIEW_ANALOGS_ALLOWED,
    PHASE_B_SUGGESTIONS_ALLOWED,
    ProposeBlock,
    RejectBlock,
    RejectReason,
    ReviewAnalogsBlock,
    ReviewBOBlock,
    block_from_dict,
)
from strbo_v1.llm_advisor.config import (
    DEFAULT_LLM_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLMClientConfig,
    load_env,
    refresh_module_globals,
)
from strbo_v1.llm_advisor.client import (
    LLMClient,
    MockLLMClient,
    OpenAIChatClient,
)

__all__ = [
    # config
    "DEFAULT_LLM_MODEL",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLMClientConfig",
    "load_env",
    "refresh_module_globals",
    # clients
    "LLMClient",
    "MockLLMClient",
    "OpenAIChatClient",
    # blocks
    "AnalogueVerdict",
    "GeneratorHint",
    "RejectReason",
    "ReviewBOBlock",
    "ProposeBlock",
    "RejectBlock",
    "AnalogBlock",
    "ReviewAnalogsBlock",
    "NoopBlock",
    "LLMBlock",
    "PHASE_A_ACTIONS_ALLOWED",
    "PHASE_A_REVIEW_ANALOGS_ALLOWED",
    "PHASE_B_SUGGESTIONS_ALLOWED",
    "block_from_dict",
]
