"""Shared (file-backed) GP history for cross-episode real-mode RL.

The small-molecule real evaluator (Vina + NN) and the EHVI acquisition selector
normally live on a per-environment basis. For a *continuous* campaign semantics
(a single GP that warms up and keeps absorbing every real evaluation across all
episodes), the history is persisted in an append-only JSONL file:

    {"smiles": "...", "vina": -6.8, "activity": 6.4}

``SharedTiltedSelector.fit`` reads the shared history (ignoring the per-episode
local history the env passes in), so the EHVI reward always reflects the global
GP. ``SharedEvaluator.evaluate`` appends every successful evaluation to the same
file, so the GP grows in real time.
"""

from __future__ import annotations

import fcntl
import json
import os
from typing import Any


class SharedHistoryStore:
    """Append-only JSONL store of ``(smiles, vina, activity)`` rows with flock."""

    def __init__(self, path: str):
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def load(self) -> list[tuple[str, float, float]]:
        rows: list[tuple[str, float, float]] = []
        if not os.path.exists(self.path):
            return rows
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    rows.append(
                        (str(obj["smiles"]), float(obj["vina"]), float(obj["activity"]))
                    )
                except Exception:  # noqa: BLE001 - skip malformed rows
                    continue
        return rows

    def append(self, smiles: str, vina: float, activity: float) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.write(
                json.dumps(
                    {"smiles": smiles, "vina": vina, "activity": activity},
                    sort_keys=True,
                )
                + "\n"
            )
            fh.flush()
            fcntl.flock(fh, fcntl.LOCK_UN)

    def count(self) -> int:
        return len(self.load())


class SharedTiltedSelector:
    """Wraps ``TiltedAcquisitionSelector`` to fit on the shared history."""

    def __init__(self, inner: Any, store: SharedHistoryStore):
        self.inner = inner
        self.store = store

    def describe(self):
        return self.inner.describe()

    def fit(self, history) -> None:
        # Ignore the per-episode history; fit on the global shared history.
        rows = self.store.load()
        self.inner.history = [
            (smiles, (vina, activity)) for smiles, vina, activity in rows
        ]

    def select(self, candidates, representations, *, count: int = 1):
        return self.inner.select(candidates, representations, count=count)


class SharedEvaluator:
    """Wraps ``SmilesCandidateEvaluator`` to append successes to the store."""

    def __init__(self, inner: Any, store: SharedHistoryStore):
        self.inner = inner
        self.store = store

    def evaluate(self, candidate):
        result = self.inner.evaluate(candidate)
        if getattr(result, "status", None) == "succeeded":
            metrics = getattr(result, "metrics", {}) or {}
            if "vina" in metrics and "activity" in metrics:
                self.store.append(
                    str(candidate.payload["smiles"]),
                    float(metrics["vina"]),
                    float(metrics["activity"]),
                )
        return result
