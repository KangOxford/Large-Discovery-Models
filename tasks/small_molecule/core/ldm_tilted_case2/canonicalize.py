"""SMILES canonicalization and hard filters for tilted case2 reservoirs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

from tasks.small_molecule.core.ldm_tilted_case2.candidate_record import CandidateRecord, SourceRecord
from tasks.small_molecule.core.ldm_tilted_case2.config import TiltedLDMCase2Config


LOGGER = logging.getLogger(__name__)
_CANONICALIZE_WORKER = Path(__file__).with_name("canonicalize_worker.py")
_CANONICAL_CACHE: dict[str, str | None] = {}


@dataclass
class RawCandidate:
    smiles: str
    source_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    base_support_level: str | None = None
    base_support_value: float = 0.0


def canonicalize_smiles(smiles: str) -> str | None:
    """Return canonical SMILES, or None when RDKit rejects the input."""
    text = str(smiles or "").strip()
    if not text or "." in text:
        return None
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return None
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if "." in canonical:
            return None
        return canonical
    except ImportError:
        return text


def canonicalize_smiles_isolated(smiles: Sequence[str]) -> list[str | None]:
    """Canonicalize a batch without exposing the campaign process to RDKit crashes."""
    texts = [str(value or "").strip() for value in smiles]
    results: list[str | None] = [None] * len(texts)
    accepted = [(index, text) for index, text in enumerate(texts) if text and "." not in text]
    missing = list(
        dict.fromkeys(
            text for _index, text in accepted if text not in _CANONICAL_CACHE
        )
    )
    _CANONICAL_CACHE.update(zip(missing, _run_canonicalizer(missing)))
    for index, text in accepted:
        results[index] = _CANONICAL_CACHE[text]
    return results


def _run_canonicalizer(smiles: list[str]) -> list[str | None]:
    if not smiles:
        return []
    try:
        process = subprocess.run(
            [sys.executable, str(_CANONICALIZE_WORKER)],
            input=json.dumps(smiles),
            text=True,
            capture_output=True,
            timeout=60.0,
            check=False,
        )
        if process.returncode == 0:
            values = json.loads(process.stdout)
            if not isinstance(values, list) or len(values) != len(smiles):
                raise ValueError("canonicalization worker returned an invalid result shape")
            return [value if isinstance(value, str) and value else None for value in values]
    except (OSError, subprocess.SubprocessError, ValueError):
        process = None

    if len(smiles) == 1:
        return_code = None if process is None else process.returncode
        LOGGER.warning(
            "RDKit canonicalization worker rejected candidate returncode=%s smiles=%r",
            return_code,
            smiles[0][:200],
        )
        return [None]

    midpoint = len(smiles) // 2
    return _run_canonicalizer(smiles[:midpoint]) + _run_canonicalizer(smiles[midpoint:])


def build_candidate_records(
    raw_records: Sequence[RawCandidate],
    sources: Sequence[SourceRecord],
    history: Sequence[tuple[str, Sequence[float | None]]],
    cfg: TiltedLDMCase2Config,
) -> tuple[list[CandidateRecord], dict[str, int]]:
    """Apply validity, length, evaluated and duplicate hard filters."""
    _ = {source.source_id for source in sources}
    drops: Counter[str] = Counter()
    history_smiles = [smiles for smiles, _scores in history]
    canonical = canonicalize_smiles_isolated(
        [raw.smiles for raw in raw_records] + history_smiles
    )
    raw_canonical = canonical[: len(raw_records)]
    evaluated = {
        value for value in canonical[len(raw_records) :] if value is not None
    }
    merged: dict[str, CandidateRecord] = {}

    for raw, canonical_smiles in zip(raw_records, raw_canonical):
        if canonical_smiles is None:
            drops["invalid"] += 1
            continue
        if len(canonical_smiles) > cfg.smiles_max_len:
            drops["overlength"] += 1
            continue
        if canonical_smiles in evaluated:
            drops["evaluated"] += 1
            continue
        if canonical_smiles in merged:
            _merge_candidate(merged[canonical_smiles], raw)
            drops["duplicate"] += 1
            continue
        merged[canonical_smiles] = CandidateRecord(
            raw_smiles=raw.smiles,
            canonical_smiles=canonical_smiles,
            method=cfg.method,
            sources=[raw.source_id],
            occurrence_by_source={raw.source_id: 1},
            base_support_level=raw.base_support_level,
            base_support_value=float(raw.base_support_value),
            metadata=dict(raw.metadata),
        )

    _update_source_valid_counts(sources, merged.values())
    return list(merged.values()), dict(drops)


def _merge_candidate(record: CandidateRecord, raw: RawCandidate) -> None:
    record.occurrence_by_source[raw.source_id] = (
        record.occurrence_by_source.get(raw.source_id, 0) + 1
    )
    if raw.source_id not in record.sources:
        record.sources.append(raw.source_id)
    record.metadata.setdefault("merged_raw_smiles", []).append(raw.smiles)
    record.metadata.setdefault("merged_rationales", []).append(
        str(raw.metadata.get("rationale", ""))
    )


def _update_source_valid_counts(
    sources: Sequence[SourceRecord],
    records: Sequence[CandidateRecord],
) -> None:
    counts: Counter[str] = Counter()
    for record in records:
        for source_id, count in record.occurrence_by_source.items():
            counts[source_id] += int(count)
    for source in sources:
        source.valid_count = int(counts.get(source.source_id, 0))
