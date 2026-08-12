"""Persist intermediate records and rendered training rows."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from ldm_tts.data.ir import LDMDataCollectionError, jdump, validate_ir_record
from ldm_tts.data.rendering import dataset_info_payload, render_record

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file or a JSON array from disk."""

    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise LDMDataCollectionError(f"JSON array expected in {path}")
        return rows
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    """Append one JSON object to a JSONL file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(jdump(dict(row)) + "\n")


@dataclass(frozen=True)
class DataCollectionPaths:
    """Paths written by :class:`DataCollectionSink`."""

    root_dir: Path
    ir_path: Path
    sft_path: Path | None
    dataset_info_path: Path | None


class DataCollectionSink:
    """Append-only writer for ldm-2.0 IR and rendered SFT records."""

    def __init__(
        self,
        root_dir: str | Path | None,
        *,
        enabled: bool = True,
        ir_filename: str = "ldm_ir.jsonl",
        sft_filename: str | None = "ldm_sft.jsonl",
        dataset_info_filename: str | None = "dataset_info.json",
        render_mode: str = "prose",
        include_parent_artifact: bool = True,
    ) -> None:
        self.enabled = bool(enabled) and root_dir is not None
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.ir_filename = ir_filename
        self.sft_filename = sft_filename
        self.dataset_info_filename = dataset_info_filename
        self.render_mode = render_mode
        self.include_parent_artifact = include_parent_artifact
        self._lock = threading.Lock()
        if self.enabled:
            assert self.root_dir is not None
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self._write_dataset_info()

    @classmethod
    def disabled(cls) -> "DataCollectionSink":
        """Return a no-op collector."""

        return cls(None, enabled=False)

    @classmethod
    def from_env(cls, *, default_root: str | Path | None = None) -> "DataCollectionSink":
        """Create a collector from environment variables.

        Environment knobs:
        * LDM_DATA_COLLECTION_ENABLED: truthy/falsey on-off switch
        * LDM_DATA_COLLECTION_DIR: output directory for JSONL artifacts
        * LDM_DATA_COLLECTION_RENDER: prose or json
        * LDM_DATA_COLLECTION_STRIP_PARENT_ARTIFACT: truthy to drop train.py text
        """

        enabled_text = os.environ.get("LDM_DATA_COLLECTION_ENABLED", "")
        explicit_dir = os.environ.get("LDM_DATA_COLLECTION_DIR", "")
        enabled = _truthy(enabled_text) if enabled_text else bool(explicit_dir)
        if not enabled:
            return cls.disabled()
        root = explicit_dir or default_root
        if root is None:
            return cls.disabled()
        render_mode = os.environ.get("LDM_DATA_COLLECTION_RENDER", "prose").strip() or "prose"
        strip_parent = _truthy(os.environ.get("LDM_DATA_COLLECTION_STRIP_PARENT_ARTIFACT", ""))
        return cls(
            root,
            enabled=True,
            render_mode=render_mode,
            include_parent_artifact=not strip_parent,
        )

    @property
    def paths(self) -> DataCollectionPaths | None:
        """Return output paths, or None when the sink is disabled."""

        if not self.enabled or self.root_dir is None:
            return None
        sft_path = self.root_dir / self.sft_filename if self.sft_filename else None
        dataset_info_path = (
            self.root_dir / self.dataset_info_filename
            if self.dataset_info_filename and self.sft_filename
            else None
        )
        return DataCollectionPaths(
            root_dir=self.root_dir,
            ir_path=self.root_dir / self.ir_filename,
            sft_path=sft_path,
            dataset_info_path=dataset_info_path,
        )

    def append(
        self,
        ir: Mapping[str, Any],
        *,
        provenance: Mapping[str, Any] | None = None,
        outcome: Mapping[str, Any] | None = None,
    ) -> None:
        """Append one training example.

        Provenance/outcome fields are stored on the IR JSONL row under a
        collection-only top-level key. The renderer ignores that key, so it cannot
        leak into the model instruction.
        """

        if not self.enabled:
            return
        validate_ir_record(ir)
        paths = self.paths
        if paths is None:
            return
        stored = json.loads(jdump(dict(ir)))
        if provenance or outcome:
            stored["collection"] = {}
            if provenance:
                stored["collection"]["provenance"] = dict(provenance)
            if outcome:
                stored["collection"]["outcome"] = dict(outcome)
        sft_row = None
        if paths.sft_path is not None:
            sft_row = render_record(
                stored,
                mode=self.render_mode,
                include_parent_artifact=self.include_parent_artifact,
            )
        with self._lock:
            append_jsonl(paths.ir_path, stored)
            if paths.sft_path is not None and sft_row is not None:
                append_jsonl(paths.sft_path, sft_row)

    def extend(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> int:
        """Append multiple IR rows and return the number written."""

        count = 0
        for row in rows:
            self.append(row, provenance=provenance)
            count += 1
        return count

    def _write_dataset_info(self) -> None:
        if self.root_dir is None or not self.dataset_info_filename or not self.sft_filename:
            return
        path = self.root_dir / self.dataset_info_filename
        path.write_text(
            jdump(dataset_info_payload(self.sft_filename), indent=2) + "\n",
            encoding="utf-8",
        )


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}
