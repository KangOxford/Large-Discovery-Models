import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tasks.small_molecule.core.ldm_tilted_case2 import canonicalize as canonicalize_module
from tasks.small_molecule.core.ldm_tilted_case2.canonicalize import RawCandidate, build_candidate_records
from tasks.small_molecule.core.ldm_tilted_case2.candidate_record import SourceRecord
from tasks.small_molecule.core.ldm_tilted_case2.config import TiltedLDMCase2Config


def cfg():
    return TiltedLDMCase2Config(method="m1_direct_llm_sir", smiles_max_len=10)


def test_invalid_smiles_dropped():
    candidates, drops = build_candidate_records(
        [RawCandidate("not a smiles", "s1")],
        [SourceRecord("s1", "direct_llm", None, 1.0, 1)],
        [],
        cfg(),
    )
    assert candidates == []
    assert drops["invalid"] == 1


def test_dot_disconnected_mixtures_dropped():
    candidates, drops = build_candidate_records(
        [RawCandidate("CCO.CCN", "s1")],
        [SourceRecord("s1", "direct_llm", None, 1.0, 1)],
        [],
        cfg(),
    )
    assert candidates == []
    assert drops["invalid"] == 1


def test_overlength_smiles_dropped():
    candidates, drops = build_candidate_records(
        [RawCandidate("CCCCCCCCCCCC", "s1")],
        [SourceRecord("s1", "direct_llm", None, 1.0, 1)],
        [],
        cfg(),
    )
    assert candidates == []
    assert drops["overlength"] == 1


def test_evaluated_smiles_dropped():
    candidates, drops = build_candidate_records(
        [RawCandidate("CCO", "s1")],
        [SourceRecord("s1", "direct_llm", None, 1.0, 1)],
        [("CCO", (-1.0, 6.0))],
        cfg(),
    )
    assert candidates == []
    assert drops["evaluated"] == 1


def test_duplicate_canonical_smiles_merge_source_counts():
    candidates, drops = build_candidate_records(
        [
            RawCandidate("CCO", "s1"),
            RawCandidate("CCO", "s1"),
            RawCandidate("CCO", "s2"),
        ],
        [
            SourceRecord("s1", "direct_llm", None, 0.5, 2),
            SourceRecord("s2", "direct_llm", None, 0.5, 1),
        ],
        [],
        cfg(),
    )
    assert len(candidates) == 1
    assert candidates[0].occurrence_by_source == {"s1": 2, "s2": 1}
    assert drops["duplicate"] == 2


def test_native_worker_failure_only_drops_crashing_candidate(monkeypatch):
    def fake_run(_command, *, input, **_kwargs):
        smiles = json.loads(input)
        if "crash" in smiles:
            return SimpleNamespace(returncode=-11, stdout="")
        return SimpleNamespace(returncode=0, stdout=json.dumps(smiles))

    monkeypatch.setattr(canonicalize_module.subprocess, "run", fake_run)
    canonicalize_module._CANONICAL_CACHE.clear()

    assert canonicalize_module.canonicalize_smiles_isolated(["CCO", "crash", "CCN"]) == [
        "CCO",
        None,
        "CCN",
    ]


def test_isolated_canonicalization_caches_successes_and_failures(monkeypatch):
    calls = []

    def fake_run(_command, *, input, **_kwargs):
        smiles = json.loads(input)
        calls.append(smiles)
        return SimpleNamespace(returncode=0, stdout=json.dumps(["CCO", None]))

    monkeypatch.setattr(canonicalize_module.subprocess, "run", fake_run)
    canonicalize_module._CANONICAL_CACHE.clear()

    assert canonicalize_module.canonicalize_smiles_isolated(["OCC", "invalid"]) == [
        "CCO",
        None,
    ]
    assert canonicalize_module.canonicalize_smiles_isolated(["invalid", "OCC"]) == [
        None,
        "CCO",
    ]
    assert calls == [["OCC", "invalid"]]
