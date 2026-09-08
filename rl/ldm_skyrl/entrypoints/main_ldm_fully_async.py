"""Fully async training entrypoint.

The reason this variant exists on this line specifically: a GP call was measured
at 67 s and a fully successful step issues up to 80 of them, so under the
synchronous trainer the reward computation sits on the critical path of every
step.  ``FullyAsyncRayPPOTrainer`` overlaps rollout with training, which is a
claim about wall-clock that phase item C4 has to confirm before it is repeated.
"""

from __future__ import annotations

import sys

from .._deps import require_skyrl
from ..config import build_config


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    require_skyrl("A1")
    cfg = build_config()
    from ..runner import run_training  # noqa: PLC0415 - deferred with skyrl

    return run_training(cfg, argv, fully_async=True)


if __name__ == "__main__":
    raise SystemExit(main())
