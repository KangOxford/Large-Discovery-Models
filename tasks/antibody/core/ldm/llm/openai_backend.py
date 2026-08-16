"""core/ldm/llm/openai_backend.py — OpenAI SDK-based LLMClient.

Single concrete LLMClient implementation. Configuration via .env file
(loaded by python-dotenv at construction time):

  LLM_API_KEY    : required
  LLM_BASE_URL   : required (endpoint)
  LLM_MODEL      : optional; defaults to ``OpenAIClient.MODEL`` below
  LLM_MAX_TOKENS : optional; positive integer passed as ``max_tokens``

The model name is intentionally hardcoded (per the convention note in
``.env``). Pass ``model=...`` to override per-instance.

There is NO fallback to CLI, litellm, or any other backend. If the SDK
is missing or env vars are absent, the constructor raises immediately.
"""
from __future__ import annotations

import os
from typing import Any

from tasks.antibody.core.ldm.llm.client import LLMClient


class OpenAIClient(LLMClient):
    """LLMClient backed by the official ``openai`` SDK (OpenAI-compatible HTTP API)."""

    # Hardcoded default model name (see ``.env`` note).
    MODEL: str = "DeepSeek-V4-Flash"

    def __init__(self, model: str | None = None, **kwargs: Any) -> None:
        # Load .env only if keys are not already set in the environment.
        # This way tests can monkeypatch env vars before constructing.
        if not os.environ.get("LLM_API_KEY") or not os.environ.get("LLM_BASE_URL"):
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass  # dotenv is optional; rely on os.environ directly

        api_key = os.environ.get("LLM_API_KEY")
        base_url = os.environ.get("LLM_BASE_URL")
        if not api_key:
            raise RuntimeError(
                "LLM_API_KEY not set. Copy .env.example to .env and fill in, "
                "or `export LLM_API_KEY=...`."
            )
        if not base_url:
            raise RuntimeError(
                "LLM_BASE_URL not set. Copy .env.example to .env and fill in, "
                "or `export LLM_BASE_URL=...`."
            )

        self.model = model or os.environ.get("LLM_MODEL") or self.MODEL
        self.max_tokens = self._max_tokens_from_env()

        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai package is not installed; run `pip install openai`."
            ) from e

        self._client = OpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def _disable_thinking_requested() -> bool:
        value = os.environ.get("LLM_DISABLE_THINKING", "").strip().lower()
        return value in {"1", "true", "yes", "on"}

    @staticmethod
    def _max_tokens_from_env() -> int | None:
        raw = os.environ.get("LLM_MAX_TOKENS") or os.environ.get("LDM_LLM_MAX_TOKENS")
        if raw is None or not raw.strip():
            return None
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError("LLM_MAX_TOKENS must be a positive integer") from exc
        if value <= 0:
            raise ValueError("LLM_MAX_TOKENS must be a positive integer")
        return value

    def make_chat_completion_kwargs(
        self,
        prompt: str,
        temperature: float,
        timeout_s: int,
        **overrides: Any,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "timeout": timeout_s,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self._disable_thinking_requested():
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        kwargs.update(overrides)
        return kwargs

    def call(self, prompt: str, temperature: float, timeout_s: int) -> str:
        kwargs = self.make_chat_completion_kwargs(prompt, temperature, timeout_s)
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def call_many(
        self,
        prompt: str,
        temperature: float,
        timeout_s: int,
        n: int,
    ) -> list[str]:
        """Request independent choices in one OpenAI-compatible call."""
        if int(n) <= 0:
            raise ValueError("n must be positive")
        kwargs = self.make_chat_completion_kwargs(
            prompt,
            temperature,
            timeout_s,
            n=int(n),
        )
        response = self._client.chat.completions.create(**kwargs)
        return [choice.message.content or "" for choice in response.choices]
