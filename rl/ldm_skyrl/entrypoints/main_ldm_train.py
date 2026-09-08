"""Synchronous training entrypoint.

    python -m ldm_skyrl.entrypoints.main_ldm_train \
        data.train_data="['/path/to/episodes.jsonl']" \
        trainer.policy.model.path=<checkpoint>

The async variant is the one phase item C4 compares this against; keeping both
means "async is faster here" can be measured rather than asserted.
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

    return run_training(cfg, argv, fully_async=False)


if __name__ == "__main__":
    raise SystemExit(main())
