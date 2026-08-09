"""core/ldm/llm/client.py — abstract LLM client.

Injectable into ``Orchestrator``. Tests use a mock; production uses
``litellm_backend.LiteLLMClient``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract LLM client. ``core/`` external code uses this interface only."""

    @abstractmethod
    def call(self, prompt: str, temperature: float, timeout_s: int) -> str:
        """Return the raw LLM response string. Raises on transport error."""

    def close(self) -> None:  # pragma: no cover
        """Optional cleanup. Default no-op."""
        pass