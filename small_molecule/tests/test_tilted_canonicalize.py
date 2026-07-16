import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from strbo_v1.ldm_tilted_case2.canonicalize import RawCandidate, build_candidate_records
from strbo_v1.ldm_tilted_case2.candidate_record import SourceRecord
from strbo_v1.ldm_tilted_case2.config import TiltedLDMCase2Config


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
