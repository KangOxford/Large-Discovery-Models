"""Tests for the LLM-advisor block dataclasses."""

import pytest

from strbo_v1.llm_advisor import (
    LLMBlock,
    NoopBlock,
    PHASE_A_ACTIONS_ALLOWED,
    PHASE_A_REVIEW_ANALOGS_ALLOWED,
    PHASE_B_SUGGESTIONS_ALLOWED,
    ProposeBlock,
    RejectBlock,
    AnalogBlock,
    ReviewAnalogsBlock,
    ReviewBOBlock,
    block_from_dict,
)


# ---------------------------------------------------------------------------
# to_dict round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "block",
    [
        ReviewBOBlock(rationale="r1", decisions={"CCO": "ok", "CCC": "override:CC(=O)O"}),
        ProposeBlock(rationale="r2", smiles=["CCO", "CCN"], rationale_per_mol={"CCO": "ethanol"}),
        RejectBlock(rationale="r3", targets=["CCO"], reason="likely_toxic"),
        AnalogBlock(rationale="r4", seeds=["CCO"], generator_hint="conservative", n_per_seed=3),
        ReviewAnalogsBlock(rationale="r5", decisions={"A1": "keep", "A2": "reject"}),
        NoopBlock(rationale="r6"),
    ],
)
def test_to_dict_roundtrip(block: LLMBlock) -> None:
    """Every block's to_dict() must be parseable back into an equal block."""
    d = block.to_dict()
    # type field is always first
    assert d["type"] == block.type
    parsed = block_from_dict(d)
    assert parsed.to_dict() == d


# ---------------------------------------------------------------------------
# block_from_dict dispatch
# ---------------------------------------------------------------------------


def test_block_from_dispatch_all_types() -> None:
    cases = {
        "review_bo": ReviewBOBlock,
        "propose": ProposeBlock,
        "reject": RejectBlock,
        "analog": AnalogBlock,
        "review_analogs": ReviewAnalogsBlock,
        "noop": NoopBlock,
    }
    for tname, cls in cases.items():
        obj = block_from_dict({"type": tname})
        assert isinstance(obj, cls), f"expected {cls.__name__}, got {type(obj).__name__}"


def test_block_from_dict_unknown_type() -> None:
    with pytest.raises(ValueError, match="unknown block type"):
        block_from_dict({"type": "weird_block"})


def test_block_from_dict_missing_type() -> None:
    with pytest.raises(ValueError, match="missing 'type' field"):
        block_from_dict({"rationale": "no type"})


def test_block_from_dict_not_dict() -> None:
    with pytest.raises(ValueError, match="must be a dict"):
        block_from_dict([1, 2, 3])        # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Stage allow sets
# ---------------------------------------------------------------------------


def test_phase_a_actions_allowed_set() -> None:
    assert set(PHASE_A_ACTIONS_ALLOWED) == {
        "propose", "reject", "analog", "noop",
    }


def test_phase_a_review_analogs_allowed_set() -> None:
    assert set(PHASE_A_REVIEW_ANALOGS_ALLOWED) == {"review_analogs"}


def test_phase_b_suggestions_allowed_set() -> None:
    assert set(PHASE_B_SUGGESTIONS_ALLOWED) == {"review_bo"}


def test_stages_disjoint() -> None:
    union = set(PHASE_A_ACTIONS_ALLOWED) | set(PHASE_A_REVIEW_ANALOGS_ALLOWED) \
        | set(PHASE_B_SUGGESTIONS_ALLOWED)
    assert len(union) == 6  # all six block types, no overlap


# ---------------------------------------------------------------------------
# Block-specific field defaults
# ---------------------------------------------------------------------------


def test_review_bo_default_empty_dict() -> None:
    b = ReviewBOBlock()
    assert b.type == "review_bo"
    assert b.rationale == ""
    assert b.decisions == {}


def test_propose_rationale_per_mol_default() -> None:
    b = ProposeBlock(rationale="r", smiles=["CCO"])
    assert b.rationale_per_mol == {}


def test_analog_default_n_per_seed() -> None:
    b = AnalogBlock(rationale="r", seeds=["CCO"])
    assert b.n_per_seed == 5
    assert b.generator_hint is None
    assert b.reasyn_config_override is None


def test_reject_has_all_reasons() -> None:
    """All five RejectReason enum values must be valid choices."""
    valid_reasons = {
        "too_similar_to_history",
        "likely_toxic",
        "synthetically_infeasible",
        "out_of_scope_pharmacophore",
        "no_signal_for_target",
    }
    for reason in valid_reasons:
        b = RejectBlock(rationale="r", targets=["CCO"], reason=reason)  # type: ignore[arg-type]
        assert b.reason == reason
