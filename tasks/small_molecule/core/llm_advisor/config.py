"""Configuration loader for the LLM advisor.

Loads ``.env`` from the project root via ``python-dotenv``. Two-tier
fallback:

1. Env vars already in ``os.environ`` take precedence
   (e.g. ``export LLM_API_KEY=...`` in the shell).
2. ``.env`` provides the default values when env vars are unset.

``.env`` is the single source of truth for the LLM advisor's
credentials (``LLM_API_KEY``, ``LLM_BASE_URL``). There is no
``LLM_MODEL`` env var — the model is hardcoded to
:data:`DEFAULT_LLM_MODEL`. The ``--llm-model`` CLI flag overrides
it per-run.

Public surface:

* :func:`load_env` — call once at process start; idempotent.
* :data:`DEFAULT_LLM_MODEL` — the hardcoded model name
  (``"DeepSeek-V4-Flash"``).
* :data:`LLM_API_KEY` / :data:`LLM_BASE_URL` / :data:`LLM_MODEL` —
  cached env values, populated by :func:`load_env`.
* :class:`LLMClientConfig` — typed bundle.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


# The single supported LLM model. Hardcoded — no env var, no override
# at this layer. Tests can pass an explicit ``model=`` kwarg to
# :meth:`LLMClientConfig.from_env` or the constructor.
DEFAULT_LLM_MODEL: str = "DeepSeek-V4-Flash"


# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Return the directory containing ``tasks.small_molecule.core/`` (i.e. the repo root)."""
    # this file: tasks.small_molecule.core/llm_advisor/config.py
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------


def load_env() -> None:
    """Load ``.env`` from the project root (idempotent).

    Two-tier fallback:
    1. env vars already set in ``os.environ`` take precedence
       (``load_dotenv(..., override=False)``).
    2. ``.env`` provides the default values when env vars are unset.

    If ``.env`` is missing AND env vars are unset,
    :meth:`LLMClientConfig.from_env` will raise a clear
    ``ValueError`` on the next access.
    """
    env_path = _project_root() / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


# ---------------------------------------------------------------------------
# Typed config bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMClientConfig:
    """Typed LLM client settings.

    Pulled from the environment after :func:`load_env` has run. Use
    :func:`LLMClientConfig.from_env` to construct.

    Attributes:
        api_key: ``LLM_API_KEY`` from env.
        base_url: ``LLM_BASE_URL`` from env (trailing slash stripped).
        model: The model name. Defaults to
            :data:`DEFAULT_LLM_MODEL` (``"DeepSeek-V4-Flash"``); can
            be overridden via the ``model=`` kwarg of
            :meth:`from_env` or directly in the constructor. There is
            **no** ``LLM_MODEL`` environment variable — the model is
            a code-level choice.
    """

    api_key: str
    base_url: str
    model: str = DEFAULT_LLM_MODEL

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError(
                "LLM_API_KEY is empty; set it in .env or "
                "export LLM_API_KEY in the environment"
            )
        if not self.base_url:
            raise ValueError(
                "LLM_BASE_URL is empty; set it in .env or "
                "export LLM_BASE_URL in the environment"
            )
        if not self.model:
            raise ValueError("LLM model name is empty")

    @classmethod
    def from_env(cls, *, model: Optional[str] = None) -> "LLMClientConfig":
        """Build a config from environment variables.

        Args:
            model: Optional explicit model name. Defaults to
                :data:`DEFAULT_LLM_MODEL`. The ``LLM_MODEL`` env var
                is intentionally NOT read; the model is a code-level
                choice.
        """
        load_env()
        return cls(
            api_key=os.environ.get("LLM_API_KEY", ""),
            base_url=os.environ.get("LLM_BASE_URL", "").rstrip("/"),
            model=model or DEFAULT_LLM_MODEL,
        )


# Module-level convenience handles. Populated when :func:`load_env` is
# called (e.g. at the top of ``client.py``). They reflect the
# environment at the time of the first read; tests can call
# :func:`load_env` again to refresh.
#
# Note: ``LLM_MODEL`` is a hardcoded string (the default model name);
# it is NOT pulled from the environment.

LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL: str = os.environ.get("LLM_BASE_URL", "").rstrip("/")
LLM_MODEL: str = DEFAULT_LLM_MODEL


def refresh_module_globals() -> None:
    """Refresh the module-level ``LLM_API_KEY`` / ``LLM_BASE_URL``
    constants after ``load_env()`` has populated ``os.environ``.

    ``LLM_MODEL`` is a hardcoded default and is not refreshed.
    """
    global LLM_API_KEY, LLM_BASE_URL
    LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
    LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "").rstrip("/")


__all__ = [
    "DEFAULT_LLM_MODEL",
    "LLMClientConfig",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "load_env",
    "refresh_module_globals",
]
