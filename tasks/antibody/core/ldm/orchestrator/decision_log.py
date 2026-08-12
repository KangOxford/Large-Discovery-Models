"""core/ldm/orchestrator/decision_log.py — append-only JSON file writer."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from ldm_tts.engine.run_store import AtomicJsonLog, utc_timestamp


class DecisionLog:
    """Append-only JSON log of orchestrator decisions.

    On each :meth:`append`, the file is rewritten with the new entry list.
    Writes are atomic (temp file + rename). A module-level lock guards
    concurrent appends from multiple processes.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._log = AtomicJsonLog(
            self.path,
            {"experiment_id": self.path.stem, "config_snapshot": {}, "decisions": []},
        )

    def _write(self, data: dict) -> None:
        self._log.write(data)

    def _read(self) -> dict:
        return self._log.read()

    def update_config_snapshot(self, config: dict[str, Any]) -> None:
        """Update the file-level config_snapshot (idempotent)."""
        self._log.update(lambda data: data.__setitem__("config_snapshot", config))

    def append(self, entry: dict) -> None:
        """Append a single decision entry."""
        payload = dict(entry)

        def append_entry(data: dict[str, Any]) -> None:
            payload.setdefault("timestamp", utc_timestamp())
            data["decisions"].append(payload)

        self._log.update(append_entry)
