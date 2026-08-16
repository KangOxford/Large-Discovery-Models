"""DSL exception hierarchy.

- :class:`DSLSyntaxError` — Python-level failure inside ``safe_exec_dsl``.
- :class:`DSLValidationError` — DSL parsed but rejected (too deep,
  sampling failure, etc.).
"""
from __future__ import annotations


class DSLSyntaxError(Exception):
    """Raised when ``safe_exec_dsl`` cannot execute the LLM-provided source.

    The original Python exception is chained via ``__cause__``.
    """


class DSLValidationError(Exception):
    """Raised when a parsed DSL atom fails validation.

    Subclasses provide specific reasons; :class:`Orchestrator` formats the
    message back to the LLM so it can revise its DSL.
    """


class SamplingTimeout(DSLValidationError):
    """``atom.sample()`` produced zero sequences within the timeout.

    Indicates the trust region is empty, too restrictive, or the
    rejection sampling acceptance rate is too low.
    """


class NestingTooDeep(DSLValidationError):
    """Atom nesting depth exceeds the configured cap."""
