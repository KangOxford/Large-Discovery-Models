"""EpisodeSpec JSONL to a SkyRL prompt dataset.

The slime line feeds episodes through ``--prompt-data <jsonl> --input-key prompt
--label-key label``.  SkyRL instead asks a dataset for rows carrying the prompt
and an ``env_extras`` payload, which the generator reads back.  The conversion is
mechanical; what is worth being careful about is that the row count equals the
line count, because a dataset silently shorter than the training batch size
turns into a confusing assertion much later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Sequence


class EpisodeDataset:
    """Sequence of ``{"prompt": ..., "env_class": ..., "env_extras": ...}`` rows."""

    ENV_CLASS = "ldm_acquisition"

    def __init__(self, data_files: str | Path | Sequence[str | Path]) -> None:
        if isinstance(data_files, (str, Path)):
            data_files = [data_files]
        self.paths = [Path(p) for p in data_files]
        missing = [str(p) for p in self.paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "episode files do not exist: " + ", ".join(missing)
            )
        self.rows: list[dict[str, Any]] = []
        for path in self.paths:
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    spec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno} is not valid JSON") from exc
                self.rows.append(self._to_row(spec))

    @staticmethod
    def _to_row(spec: dict[str, Any]) -> dict[str, Any]:
        # ``prompt`` is deliberately left empty: the opening prompt comes from
        # ``env.reset()`` at rollout time, so storing a copy here would create a
        # second source of truth that drifts the moment prompts.py changes.
        return {"prompt": [], "env_class": EpisodeDataset.ENV_CLASS, "env_extras": spec}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.rows)


__all__ = ["EpisodeDataset"]
