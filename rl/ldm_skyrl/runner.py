"""The part that only exists once the stack is proven.

Everything else in this package runs on a login node.  This module is where the
trainer is actually constructed, and it is gated on phase item A5 -- the one-GPU
smoke that shows the loop closes at all.  Writing an untested body here would
mean the first real failure surfaces somewhere inside Ray with no indication of
which assumption broke.
"""

from __future__ import annotations

from typing import Any

from ._deps import gate


def run_training(cfg: Any, argv: list[str], *, fully_async: bool) -> int:
    trainer = "FullyAsyncRayPPOTrainer" if fully_async else "RayPPOTrainer"
    raise gate(
        "A5",
        f"would construct {trainer} here. The trainer is deliberately not wired "
        "until the constant-reward smoke closes the loop on one GPU: until then "
        "a failure inside Ray cannot be attributed to the stack rather than to "
        "the LDM wiring.",
    )


__all__ = ["run_training"]
