"""Large Discovery Model test-time search.

Import from focused package interfaces such as :mod:`ldm_tts.contracts` or
the module that owns the required behavior. The package root intentionally
performs no eager imports, so task discovery does not require optimization,
transport, or training dependencies.
"""

from __future__ import annotations

from typing import Any

__all__: tuple[str, ...] = ()


def __getattr__(name: str) -> Any:
    """Resolve historical root exports lazily during the migration."""

    from ldm_tts.compat import resolve

    try:
        value = resolve(name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    from ldm_tts.compat import COMPAT_EXPORTS

    return sorted(set(globals()) | COMPAT_EXPORTS)
