"""Stable shared-runner adapter for the antibody task."""

from __future__ import annotations

from typing import Any

from tasks.antibody.core import workflow as _workflow
from tasks.antibody.core.workflow import *  # noqa: F401,F403


def parse_args(argv: list[str] | None = None) -> Any:
    return _workflow.parse_args(argv)


def describe_ldm_task(*args: Any, **kwargs: Any) -> Any:
    return _workflow.describe_ldm_task(*args, **kwargs)


def main(argv: list[str] | None = None) -> int | None:
    return _workflow.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
