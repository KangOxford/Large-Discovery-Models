"""bo/ldm/llm/openai_backend.py — OpenAI SDK-based LLMClient.

Single concrete LLMClient implementation. Configuration via .env file
(loaded by python-dotenv at construction time):

  LLM_API_KEY    : required
  LLM_BASE_URL   : required (endpoint)
  LLM_MODEL      : optional; defaults to ``OpenAIClient.MODEL`` below

The model name is intentionally hardcoded (per the convention note in
``.env``). Pass ``model=...`` to override per-instance.

There is NO fallback to CLI, litellm, or any other backend. If the SDK
is missing or env vars are absent, the constructor raises immediately.
"""
from __future__ import annotations

import os
from typing import Any

from bo.ldm.llm.client import LLMClient


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

        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai package is not installed; run `pip install openai`."
            ) from e

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

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model or os.environ.get("LLM_MODEL") or self.MODEL

    @staticmethod
    def _disable_thinking_requested() -> bool:
        value = os.environ.get("LLM_DISABLE_THINKING", "").strip().lower()
        return value in {"1", "true", "yes", "on"}

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
        if self._disable_thinking_requested():
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        kwargs.update(overrides)
        return kwargs

    def call(self, prompt: str, temperature: float, timeout_s: int) -> str:
        kwargs = self.make_chat_completion_kwargs(prompt, temperature, timeout_s)
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
