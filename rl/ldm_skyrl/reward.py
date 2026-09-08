"""Bounding the concurrency of the expensive half of the rollout.

Docking and the GP are the slow part.  Two measurements from the slime line set
the bounds:

* ``vina_max_workers=32`` had no basis; the measured saturation point is 8.
* A single ``kernel=sk`` GP call takes about 67 s, and a fully successful step
  issues up to 80 of them.

So the reward path needs a ceiling that is a property of the machine, not of the
batch size.  SkyRL's own rate limiter
(``generator.rate_limit.{trajectories_per_second,max_concurrency}``) is the right
place for this; this module holds the measured defaults and a plain asyncio
fallback for tests, which run without Ray.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")

# Measured on this cluster; see ldm_rl/results/FINDINGS.md section 3.
VINA_SATURATION_WORKERS = 8
GP_SECONDS_PER_CALL_SK = 67.0


@dataclass
class RewardConcurrency:
    """A ceiling on in-flight evaluations, with the reason recorded next to it."""

    max_concurrency: int = VINA_SATURATION_WORKERS
    reason: str = (
        "docking throughput saturates at 8 workers; above that the queue grows "
        "without the step getting faster"
    )

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError(
                f"max_concurrency must be at least 1, got {self.max_concurrency}"
            )
        self._sem = asyncio.Semaphore(self.max_concurrency)
        self.peak_in_flight = 0
        self._in_flight = 0

    async def run(self, fn: Callable[[], Awaitable[T]]) -> T:
        async with self._sem:
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
            try:
                return await fn()
            finally:
                self._in_flight -= 1

    def as_skyrl_rate_limit(self, trajectories_per_second: float | None = None) -> dict[str, Any]:
        """The same ceiling expressed as SkyRL generator config."""
        cfg: dict[str, Any] = {"enabled": True, "max_concurrency": self.max_concurrency}
        if trajectories_per_second is not None:
            cfg["trajectories_per_second"] = trajectories_per_second
        return cfg


__all__ = ["GP_SECONDS_PER_CALL_SK", "RewardConcurrency", "VINA_SATURATION_WORKERS"]
