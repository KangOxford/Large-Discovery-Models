"""Prepare provenance-grouped LDM rationale data for full-parameter SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ldm_tts.data import read_jsonl, render_record, validate_ir_record

TRAIN_DATASET = "ldm_rationale_train"
EVAL_DATASET = "ldm_rationale_eval"
TRAIN_FILENAME = "ldm_rationale_train.jsonl"
EVAL_FILENAME = "ldm_rationale_eval.jsonl"
DATASET_INFO_FILENAME = "dataset_info.json"
SPLIT_SUMMARY_FILENAME = "split_summary.json"

_COLUMNS = {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "system": "system",
}


@dataclass(frozen=True)
class PreparationReport:
    input_rows: int
    train_rows: int
    eval_rows: int
    train_groups: int
    eval_groups: int
    skipped_reasoning_unavailable: int
    skipped_missing_reasoning: int
    eval_group_keys: tuple[str, ...]


def _has_reasoning(row: Mapping[str, Any]) -> bool:
    reasoning = row["action"].get("reasoning")
    return isinstance(reasoning, str) and bool(reasoning.strip())


def _group_key(row: Mapping[str, Any], source: Path) -> str:
    collection = row.get("collection")
    provenance = (
        collection.get("provenance") if isinstance(collection, Mapping) else None
    )
    if isinstance(provenance, Mapping):
        for field in (
            "run_id",
            "trajectory_id",
            "campaign_id",
            "trajectory_dir",
            "run_dir",
            "source_path",
        ):
            value = provenance.get(field)
            if value is not None and str(value).strip():
                return f"{field}:{value}"

        components = []
        for field in ("antigen", "seed"):
            value = provenance.get(field)
            if value is not None and str(value).strip():
                components.append((field, str(value)))
        if components:
            return "provenance:" + json.dumps(components, separators=(",", ":"))

    # A source file is the safest available group when historical IR lacks
    # per-row provenance. Multiple runs without provenance must use one file each.
    return f"file:{source.expanduser().resolve()}"


def _dataset_entry(filename: str) -> dict[str, Any]:
    return {
        "file_name": filename,
        "formatting": "alpaca",
        "columns": dict(_COLUMNS),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _stable_group_order(group_keys: Sequence[str], seed: int) -> list[str]:
    return sorted(
        group_keys,
        key=lambda key: hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest(),
    )


def prepare_dataset(
    input_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    eval_fraction: float = 0.1,
    seed: int = 42,
    overwrite: bool = False,
) -> PreparationReport:
    """Filter augmented IR and render deterministic group-disjoint SFT shards."""

    if not input_paths:
        raise ValueError("at least one input IR file is required")
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be between 0 and 1")

    grouped: dict[str, list[dict[str, Any]]] = {}
    input_rows = 0
    skipped_unavailable = 0
    skipped_missing = 0

    for input_path in input_paths:
        source = Path(input_path)
        for row in read_jsonl(source):
            input_rows += 1
            validate_ir_record(row)
            if row["task"].get("reasoning_available") is False:
                skipped_unavailable += 1
                continue
            if not _has_reasoning(row):
                skipped_missing += 1
                continue
            grouped.setdefault(_group_key(row, source), []).append(row)

    if len(grouped) < 2:
        raise ValueError(
            "at least two eligible provenance groups are required for a held-out split; "
            "provide a run/trajectory identifier, antigen/seed, or one input file per run"
        )

    group_keys = _stable_group_order(tuple(grouped), seed)
    eval_group_count = min(
        len(group_keys) - 1, max(1, math.ceil(len(group_keys) * eval_fraction))
    )
    eval_group_keys = set(group_keys[:eval_group_count])

    train_ir = [
        row for key in group_keys if key not in eval_group_keys for row in grouped[key]
    ]
    eval_ir = [
        row for key in group_keys if key in eval_group_keys for row in grouped[key]
    ]
    train_rows = [render_record(row) for row in train_ir]
    eval_rows = [render_record(row) for row in eval_ir]

    report = PreparationReport(
        input_rows=input_rows,
        train_rows=len(train_rows),
        eval_rows=len(eval_rows),
        train_groups=len(group_keys) - eval_group_count,
        eval_groups=eval_group_count,
        skipped_reasoning_unavailable=skipped_unavailable,
        skipped_missing_reasoning=skipped_missing,
        eval_group_keys=tuple(sorted(eval_group_keys)),
    )

    destination = Path(output_dir)
    targets = {
        TRAIN_FILENAME: train_rows,
        EVAL_FILENAME: eval_rows,
        DATASET_INFO_FILENAME: {
            TRAIN_DATASET: _dataset_entry(TRAIN_FILENAME),
            EVAL_DATASET: _dataset_entry(EVAL_FILENAME),
        },
        SPLIT_SUMMARY_FILENAME: asdict(report),
    }
    existing = [destination / name for name in targets if (destination / name).exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"refusing to overwrite existing output(s): {names}")

    destination.mkdir(parents=True, exist_ok=True)
    _write_jsonl(destination / TRAIN_FILENAME, train_rows)
    _write_jsonl(destination / EVAL_FILENAME, eval_rows)
    for filename in (DATASET_INFO_FILENAME, SPLIT_SUMMARY_FILENAME):
        (destination / filename).write_text(
            json.dumps(targets[filename], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare reasoning-eligible, provenance-grouped LDM data for full SFT."
    )
    parser.add_argument("--input", action="append", required=True, dest="input_paths")
    parser.add_argument("--output-dir", default="data/generated/full_sft")
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = prepare_dataset(
        args.input_paths,
        args.output_dir,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
