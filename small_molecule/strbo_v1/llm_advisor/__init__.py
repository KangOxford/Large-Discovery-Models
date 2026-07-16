"""LLM advisor public re-exports.

Importing this module is cheap; it pulls in only data containers and
the (lazy) schema validator. The LLM client and orchestrator are
imported on demand to keep the import graph narrow for callers that
only need the dataclasses.
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
    LLMClientConfig,
    load_env,
)
from strbo_v1.llm_advisor.fallback import (
    fallback_actions,
    fallback_review_analogs,
    fallback_review_suggestions,
)
from strbo_v1.llm_advisor.advisor import LLMAdvisor, LLMAttemptRecord
from strbo_v1.llm_advisor.parser import (
    ParseError,
    SchemaError,
    SemanticError,
    format_error_for_prompt,
    parse_blocks,
    validate_blocks_phase,
    validate_semantics,
)
from strbo_v1.llm_advisor.round_state import (
    PostSuggestionState,
    PreActionState,
    PreReviewAnalogsState,
)
from strbo_v1.llm_advisor.schema import BLOCKS_SCHEMA_JSON, get_validator
from strbo_v1.llm_advisor.state import AnalogueRecord, GPSummary, PickRecord, ScoreValue

__all__ = [
    # config
    "LLMClientConfig",
    "load_env",
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
    # state
    "GPSummary",
    "PickRecord",
    "AnalogueRecord",
    "ScoreValue",
    "PreActionState",
    "PreReviewAnalogsState",
    "PostSuggestionState",
    # schema
    "BLOCKS_SCHEMA_JSON",
    "get_validator",
    # parser
    "parse_blocks",
    "validate_blocks_phase",
    "validate_semantics",
    "format_error_for_prompt",
    "ParseError",
    "SchemaError",
    "SemanticError",
    # fallback
    "fallback_actions",
    "fallback_review_analogs",
    "fallback_review_suggestions",
    # advisor
    "LLMAdvisor",
    "LLMAttemptRecord",
]
