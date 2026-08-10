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

    def call_many(
        self,
        prompt: str,
        temperature: float,
        timeout_s: int,
        n: int,
    ) -> list[str]:
        """Return independent completions, falling back to sequential calls."""
        if int(n) <= 0:
            raise ValueError("n must be positive")
        return [
            self.call(prompt, temperature=temperature, timeout_s=timeout_s)
            for _ in range(int(n))
        ]

    def close(self) -> None:  # pragma: no cover
        """Optional cleanup. Default no-op."""
        pass
