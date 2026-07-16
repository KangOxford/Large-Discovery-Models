"""LLMAdvisor: three-stage decision making with retry + fallback.

Public API:

* :class:`LLMAdvisor` — the single entry point the orchestrator calls.
  Has three public methods, one per stage:

  - :meth:`decide_actions` — Stage A1: pool-management actions.
  - :meth:`decide_review_analogs` — Stage A2: review generated analogues.
  - :meth:`decide_review_suggestions` — Stage B: review BO suggestions.

  All three return ``(blocks, attempts, fallback_used)``.

* :class:`LLMAttemptRecord` — value type recorded for each
  LLM-attempt.  Stored in the trajectory.

The advisor keeps the per-stage retry loop self-contained:

* The state passed in is treated as immutable (frozen dataclass); the
  advisor uses :func:`dataclasses.replace` to construct a per-attempt
  copy with the next ``attempt`` index and accumulated
  ``previous_errors``.  The ``previous_errors`` tuple is *only* the
  errors from the current stage's previous attempts — it does **not**
  leak across stages.

* On success: returns the parsed + validated blocks.  The
  ``fallback_used`` flag is ``False``.

* On retry exhaustion: returns the stage-specific fallback blocks.
  The ``fallback_used`` flag is ``True``; the last (failed) attempt
  is still in :attr:`attempts` so the trajectory records the full
  failure history.

* Transport errors (network, timeout) are caught and converted into
  a :class:`ParseError` (the message is the exception class+str); the
  retry loop treats it the same as a parse failure.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from strbo_v1.llm_advisor.blocks import LLMBlock
from strbo_v1.llm_advisor.client import LLMClient
from strbo_v1.llm_advisor.fallback import (
    fallback_actions,
    fallback_review_analogs,
    fallback_review_suggestions,
)
from strbo_v1.llm_advisor.parser import (
    ParseError,
    SchemaError,
    SemanticError,
    format_error_for_prompt,
    parse_blocks,
    validate_blocks_phase,
    validate_semantics,
)
from strbo_v1.llm_advisor.prompt import (
    SYSTEM_ACTIONS,
    SYSTEM_REVIEW_ANALOGS,
    SYSTEM_REVIEW_SUGGESTIONS,
    format_system_actions,
    format_system_review_analogs,
    format_system_review_suggestions,
    render_user_actions,
    render_user_review_analogs,
    render_user_suggestions,
)
from strbo_v1.llm_advisor.round_state import (
    PostSuggestionState,
    PreActionState,
    PreReviewAnalogsState,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attempt record
# ---------------------------------------------------------------------------


@dataclass
class LLMAttemptRecord:
    """One LLM call (or attempted call) in a stage.

    Recorded in the trajectory. ``raw_response`` is the full text the
    LLM returned (or ``""`` on transport error). ``validation_errors``
    is a list of human-readable strings (one per failed validation
    step). ``duration_ms`` is wall-clock for the LLM round trip
    (excluding local validation).
    """

    attempt: int
    raw_response: str
    stage: str = ""
    system_prompt: str = ""
    user_prompt: str = ""
    parsed_blocks: List[Dict[str, Any]] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    json_mode: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt": self.attempt,
            "stage": self.stage,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "raw_response": self.raw_response,
            "raw_output": self.raw_response,
            "parsed_blocks": list(self.parsed_blocks),
            "validation_errors": list(self.validation_errors),
            "duration_ms": self.duration_ms,
            "json_mode": self.json_mode,
        }


# ---------------------------------------------------------------------------
# Advisor
# ---------------------------------------------------------------------------


@dataclass
class LLMAdvisor:
    """Three-stage LLM decision maker with bounded retry.

    Args:
        llm: an :class:`LLMClient` (production or mock).
        max_retries: total attempts per stage (1 = no retry, 3 = 3
            attempts, default).  ``1`` means "use the LLM once and
            fall back if it fails".
        use_rdkit: when True (default), :func:`validate_semantics`
            also checks that SMILES are RDKit-parseable.  Set to False
            in minimal test environments.
        guidance: Optional free-form text appended to all three
            system prompts (Stage A1 actions, A2 review-analogs, B
            review-suggestions). Use to steer the LLM's behaviour
            without changing code. Empty string disables.
    """

    llm: LLMClient
    max_retries: int = 3
    use_rdkit: bool = True
    guidance: str = ""

    def __post_init__(self) -> None:
        if self.max_retries < 1:
            raise ValueError(
                f"max_retries must be >= 1, got {self.max_retries}"
            )

    # -- Public: three-stage entry points --------------------------------

    def decide_actions(
        self, state: PreActionState,
    ) -> Tuple[List[LLMBlock], List[LLMAttemptRecord], bool]:
        """Stage A1: pool-management actions (propose/reject/analog/noop)."""
        return self._decide_with_retry(
            state=state,
            stage="A_actions",
            system=format_system_actions(
                round_idx=state.round_idx, guidance=self.guidance,
            ),
            user_renderer=render_user_actions,
            fallback_fn=fallback_actions,
            # Pool-min enforcement: reject noop when pool < pool_min_size.
            pool_min_size=getattr(state, "pool_min_size", None),
        )

    def decide_review_analogs(
        self, state: PreReviewAnalogsState,
    ) -> Tuple[List[LLMBlock], List[LLMAttemptRecord], bool]:
        """Stage A2: review generated analogues (keep/reject)."""
        return self._decide_with_retry(
            state=state,
            stage="A_review_analogs",
            system=format_system_review_analogs(
                round_idx=state.round_idx, guidance=self.guidance,
            ),
            user_renderer=render_user_review_analogs,
            fallback_fn=fallback_review_analogs,
        )

    def decide_review_suggestions(
        self, state: PostSuggestionState,
    ) -> Tuple[List[LLMBlock], List[LLMAttemptRecord], bool]:
        """Stage B: review BO suggestions (ok/skip/override)."""
        return self._decide_with_retry(
            state=state,
            stage="B_suggestions",
            system=format_system_review_suggestions(
                round_idx=state.round_idx, guidance=self.guidance,
            ),
            user_renderer=render_user_suggestions,
            fallback_fn=fallback_review_suggestions,
        )

    # -- Internal: shared retry loop -----------------------------------

    def _decide_with_retry(
        self,
        *,
        state,
        stage: str,
        system: str,
        user_renderer,
        fallback_fn,
        pool_min_size: Optional[int] = None,
    ) -> Tuple[List[LLMBlock], List[LLMAttemptRecord], bool]:
        """Bounded-retry loop shared by all three stages.

        The state passed in is treated as immutable; we use
        :func:`dataclasses.replace` to produce per-attempt copies
        with the next ``attempt`` index and accumulated
        ``previous_errors`` (errors are *not* shared across stages —
        each stage has its own).
        """
        attempts: List[LLMAttemptRecord] = []
        previous_errors: List[str] = []

        for attempt_idx in range(1, self.max_retries + 1):
            state_for_attempt = dataclasses.replace(
                state,
                attempt=attempt_idx,
                previous_errors=tuple(previous_errors),
            )
            user_prompt = user_renderer(state_for_attempt)

            t0 = time.monotonic()
            try:
                raw = self.llm.chat(system=system, user=user_prompt,
                                    json_mode=True)
            except Exception as exc:                            # transport
                err = f"transport: {type(exc).__name__}: {exc}"
                attempts.append(LLMAttemptRecord(
                    attempt=attempt_idx,
                    stage=stage,
                    system_prompt=system,
                    user_prompt=user_prompt,
                    raw_response="",
                    parsed_blocks=[],
                    validation_errors=[err],
                    duration_ms=(time.monotonic() - t0) * 1000.0,
                ))
                previous_errors.append(err)
                LOGGER.warning(
                    "LLM stage=%s attempt %d/%d transport error: %s",
                    stage, attempt_idx, self.max_retries, err,
                )
                continue

            # Try to parse + validate.
            try:
                blocks = parse_blocks(raw)
                validate_blocks_phase(blocks, stage)
                # Pool-min enforcement (Stage A1 only) — passes
                # the LDM's `pool_min_size` to the validator so
                # that noop is rejected when pool is below minimum.
                validate_semantics(
                    blocks,
                    pool=(state.pool or None) if stage == "A_actions" else None,
                    phase=stage,
                    use_rdkit=self.use_rdkit,
                    pool_min_size=pool_min_size if stage == "A_actions" else None,
                )
            except (ParseError, SchemaError, SemanticError) as exc:
                err = format_error_for_prompt(exc)
                attempts.append(LLMAttemptRecord(
                    attempt=attempt_idx,
                    stage=stage,
                    system_prompt=system,
                    user_prompt=user_prompt,
                    raw_response=raw,
                    parsed_blocks=[],
                    validation_errors=[err],
                    duration_ms=(time.monotonic() - t0) * 1000.0,
                ))
                previous_errors.append(err)
                LOGGER.warning(
                    "LLM stage=%s attempt %d/%d validation error: %s",
                    stage, attempt_idx, self.max_retries, err,
                )
                continue

            # Success.
            attempts.append(LLMAttemptRecord(
                attempt=attempt_idx,
                stage=stage,
                system_prompt=system,
                user_prompt=user_prompt,
                raw_response=raw,
                parsed_blocks=[b.to_dict() for b in blocks],
                validation_errors=[],
                duration_ms=(time.monotonic() - t0) * 1000.0,
            ))
            return blocks, attempts, False

        # Retries exhausted: use the stage-specific fallback.
        blocks = fallback_fn(state)
        LOGGER.warning(
            "LLM stage=%s: all %d attempts failed; using fallback "
            "(%d blocks).",
            stage, self.max_retries, len(blocks),
        )
        return blocks, attempts, True


__all__ = ["LLMAdvisor", "LLMAttemptRecord"]
