"""Shared trajectory and JSON log helpers for LDM-TTS runs."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def utc_timestamp() -> str:
    """Return a stable UTC timestamp for JSON records."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL rows from ``path``."""

    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


class JsonlTrajectoryRecorder:
    """Append JSONL round records and write companion JSON artifacts."""

    def __init__(
        self,
        trajectory_dir: str | Path | None,
        *,
        config_snapshot: dict[str, Any] | None = None,
        existing_rounds: Sequence[dict[str, Any]] | None = None,
        rounds_filename: str = "rounds.jsonl",
        reset_rounds_file: bool = False,
        sort_keys: bool = False,
    ) -> None:
        self.trajectory_dir = Path(trajectory_dir) if trajectory_dir else None
        self.rounds_filename = rounds_filename
        self.sort_keys = sort_keys
        self.rounds: list[dict[str, Any]] = list(existing_rounds or [])
        if self.trajectory_dir:
            self.trajectory_dir.mkdir(parents=True, exist_ok=True)
            if reset_rounds_file:
                (self.trajectory_dir / self.rounds_filename).write_text("", encoding="utf-8")
            if config_snapshot is not None:
                self.write_json("config.json", config_snapshot)

    def append_round(self, record: dict[str, Any]) -> None:
        """Append a round to memory and to ``rounds.jsonl`` when enabled."""

        self.rounds.append(record)
        if self.trajectory_dir:
            with (self.trajectory_dir / self.rounds_filename).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=self.sort_keys) + "\n")

    def write_json(self, filename: str, payload: Any, *, indent: int = 2) -> Path | None:
        """Write an artifact inside ``trajectory_dir`` and return its path."""

        if self.trajectory_dir is None:
            return None
        path = self.trajectory_dir / filename
        path.write_text(json.dumps(payload, indent=indent, ensure_ascii=False), encoding="utf-8")
        return path


class AtomicJsonLog:
    """Small atomic JSON document log used by LDM decision logs."""

    def __init__(self, path: str | Path, default_payload: dict[str, Any]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.write(default_payload)

    def read(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def write(self, payload: dict[str, Any]) -> None:
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=self.path.name + ".",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def update(self, mutator) -> dict[str, Any]:
        """Apply ``mutator`` to the document under a process-local lock."""

        with _LOCK:
            payload = self.read()
            mutator(payload)
            self.write(payload)
            return payload
