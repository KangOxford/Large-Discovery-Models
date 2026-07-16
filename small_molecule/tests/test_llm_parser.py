"""Tests for the LLM-advisor parser.

Covers:

* :func:`parse_blocks` happy path with one / multiple / out-of-order
  blocks.
* :class:`ParseError` triggers: no ```json``` blocks, malformed
  JSON.
* :class:`SchemaError` triggers: missing required fields, bad type
  enum, value out of range, additional properties.
* :func:`validate_blocks_phase` — Phase A and Phase B allow sets.
* :func:`validate_semantics` — RDKit-invalid SMILES, reject target
  not in pool, duplicate blocks, ``override:`` empty target, etc.
"""

import json

import pytest

from strbo_v1.llm_advisor import (
    AnalogBlock,
    NoopBlock,
    ParseError,
    ProposeBlock,
    RejectBlock,
    ReviewAnalogsBlock,
    ReviewBOBlock,
    SchemaError,
    SemanticError,
    parse_blocks,
    validate_blocks_phase,
    validate_semantics,
)


# ---------------------------------------------------------------------------
# parse_blocks: happy path
# ---------------------------------------------------------------------------


def test_parse_single_block() -> None:
    text = '```json\n{"type":"noop","rationale":"r"}\n```'
    blocks = parse_blocks(text)
    assert len(blocks) == 1
    assert isinstance(blocks[0], NoopBlock)
    assert blocks[0].rationale == "r"


def test_parse_multiple_blocks() -> None:
    text = (
        '```json\n{"type":"noop","rationale":"a"}\n```\n\n'
        '```json\n{"type":"propose","rationale":"b","smiles":["CCO"]}\n```\n'
    )
    blocks = parse_blocks(text)
    assert len(blocks) == 2
    assert isinstance(blocks[0], NoopBlock)
    assert isinstance(blocks[1], ProposeBlock)
    assert blocks[1].smiles == ["CCO"]


def test_parse_out_of_order_blocks() -> None:
    """Order in the LLM response is preserved in the returned list."""
    text = (
        '```json\n{"type":"propose","rationale":"p","smiles":["CCO"]}\n```\n'
        '```json\n{"type":"noop","rationale":"n"}\n```'
    )
    blocks = parse_blocks(text)
    assert [b.type for b in blocks] == ["propose", "noop"]


def test_parse_with_prose_around_blocks() -> None:
    text = (
        "Here are my decisions:\n"
        '```json\n{"type":"noop","rationale":"a"}\n```\n'
        "That is all.\n"
    )
    blocks = parse_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].rationale == "a"


# ---------------------------------------------------------------------------
# parse_blocks: failure modes
# ---------------------------------------------------------------------------


def test_parse_no_json_blocks_raises_parse_error() -> None:
    with pytest.raises(ParseError, match="no"):
        parse_blocks("just plain text, no fences")


def test_parse_empty_string_raises_parse_error() -> None:
    with pytest.raises(ParseError):
        parse_blocks("")


def test_parse_malformed_json_raises_parse_error() -> None:
    text = '```json\n{not valid json at all}\n```'
    with pytest.raises(ParseError, match="malformed JSON"):
        parse_blocks(text)


def test_parse_top_level_not_object_raises_parse_error() -> None:
    """Top-level JSON arrays are silently skipped (regex only matches objects).

    This is intentional — the LLM is only ever asked to emit block
    objects, not arrays. The parser's regex is anchored on ``{``.
    """
    text = '```json\n[1, 2, 3]\n```'
    with pytest.raises(ParseError, match="no"):
        parse_blocks(text)


def test_parse_object_with_invalid_inner_value_raises_parse_error() -> None:
    """Object-shaped JSON is parsed first; invalid JSON inside still raises ParseError."""
    # The object opens with { so the regex captures it; the contents are malformed.
    text = '```json\n{"type": "noop", invalid}\n```'
    with pytest.raises(ParseError, match="malformed JSON"):
        parse_blocks(text)


def test_parse_missing_required_field_raises_schema_error() -> None:
    """review_bo requires 'decisions'."""
    text = '```json\n{"type":"review_bo","rationale":"r"}\n```'
    with pytest.raises(SchemaError, match="decisions"):
        parse_blocks(text)


def test_parse_unknown_block_type_raises_schema_error() -> None:
    text = '```json\n{"type":"weird_block","rationale":"r"}\n```'
    with pytest.raises(SchemaError, match="not a known block type"):
        parse_blocks(text)


def test_parse_bad_enum_value_raises_schema_error() -> None:
    text = (
        '```json\n'
        '{"type":"reject","rationale":"r","targets":["CCO"],'
        '"reason":"not_a_valid_reason"}'
        '\n```'
    )
    with pytest.raises(SchemaError, match="not_a_valid_reason"):
        parse_blocks(text)


def test_parse_additional_property_rejected() -> None:
    text = (
        '```json\n'
        '{"type":"noop","rationale":"r","extra_field":"oops"}'
        '\n```'
    )
    with pytest.raises(SchemaError, match="extra_field"):
        parse_blocks(text)


def test_parse_rationale_too_long_raises_schema_error() -> None:
    text = (
        '```json\n'
        f'{{"type":"reject","rationale":"{"x"*201}","targets":["CCO"],'
        '"reason":"likely_toxic"}'
        '\n```'
    )
    with pytest.raises(SchemaError, match="maxLength|too long|201"):
        parse_blocks(text)


# ---------------------------------------------------------------------------
# validate_blocks_phase
# ---------------------------------------------------------------------------


def test_stage_a1_blocks_pass() -> None:
    blocks = [
        NoopBlock(rationale="n"),
        ProposeBlock(rationale="p", smiles=["CCO"]),
    ]
    validate_blocks_phase(blocks, "A_actions")         # no exception


def test_stage_a2_blocks_pass() -> None:
    blocks = [ReviewAnalogsBlock(rationale="r", decisions={"A1": "keep"})]
    validate_blocks_phase(blocks, "A_review_analogs")  # no exception


def test_stage_b_blocks_pass() -> None:
    blocks = [ReviewBOBlock(rationale="r", decisions={"CCO": "ok"})]
    validate_blocks_phase(blocks, "B_suggestions")     # no exception


def test_stage_a1_rejects_review_bo() -> None:
    blocks = [ReviewBOBlock(rationale="r", decisions={"CCO": "ok"})]
    with pytest.raises(SemanticError, match="review_bo"):
        validate_blocks_phase(blocks, "A_actions")


def test_stage_a1_rejects_review_analogs() -> None:
    """Review-analogs is not allowed in Stage A1 (only in Stage A2)."""
    blocks = [ReviewAnalogsBlock(rationale="r", decisions={"A1": "keep"})]
    with pytest.raises(SemanticError, match="review_analogs"):
        validate_blocks_phase(blocks, "A_actions")


def test_stage_b_rejects_propose() -> None:
    blocks = [ProposeBlock(rationale="p", smiles=["CCO"])]
    with pytest.raises(SemanticError, match="propose"):
        validate_blocks_phase(blocks, "B_suggestions")


def test_stage_b_rejects_noop() -> None:
    blocks = [NoopBlock(rationale="n")]
    with pytest.raises(SemanticError, match="noop"):
        validate_blocks_phase(blocks, "B_suggestions")


def test_phase_invalid_argument() -> None:
    with pytest.raises(ValueError, match="phase must be"):
        validate_blocks_phase([], "C")        # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# validate_semantics
# ---------------------------------------------------------------------------


def test_validate_semantics_phase_a_pool_check() -> None:
    """Reject targets must be in the pool."""
    blocks = [RejectBlock(rationale="r", targets=["XYZ"], reason="likely_toxic")]
    with pytest.raises(SemanticError, match="not in pool"):
        validate_semantics(blocks, pool=["CCO", "CCN"], phase="A_actions", use_rdkit=False)


def test_validate_semantics_phase_a_pool_check_passes() -> None:
    blocks = [RejectBlock(rationale="r", targets=["CCO"], reason="likely_toxic")]
    validate_semantics(blocks, pool=["CCO", "CCN"], phase="A_actions", use_rdkit=False)


def test_validate_semantics_duplicate_blocks() -> None:
    """At most one block per type per round."""
    blocks = [
        NoopBlock(rationale="a"),
        NoopBlock(rationale="b"),
    ]
    with pytest.raises(SemanticError, match="duplicate type"):
        validate_semantics(blocks, phase="A_actions", use_rdkit=False)


def test_validate_semantics_rdkit_propose() -> None:
    """Bad SMILES in propose -> SemanticError when use_rdkit=True."""
    blocks = [ProposeBlock(rationale="p", smiles=["NOT_A_VALID_SMILES!!"])]
    with pytest.raises(SemanticError, match="RDKit-invalid"):
        validate_semantics(blocks, phase="A_actions", use_rdkit=True)


def test_validate_semantics_rdkit_analog_seeds() -> None:
    blocks = [AnalogBlock(rationale="a", seeds=["@#$%"])]
    with pytest.raises(SemanticError, match="RDKit-invalid"):
        validate_semantics(blocks, phase="A_actions", use_rdkit=True)


def test_validate_semantics_review_bo_override_empty() -> None:
    """override: with empty SMILES -> SemanticError."""
    blocks = [ReviewBOBlock(rationale="r", decisions={"CCO": "override:"})]
    with pytest.raises(SemanticError, match="override target empty"):
        validate_semantics(blocks, phase="B_suggestions", use_rdkit=False)


def test_validate_semantics_review_bo_bad_verdict() -> None:
    blocks = [ReviewBOBlock(rationale="r", decisions={"CCO": "nonsense"})]
    with pytest.raises(SemanticError, match="bad verdict"):
        validate_semantics(blocks, phase="B_suggestions", use_rdkit=False)


def test_validate_semantics_review_bo_ok_skip() -> None:
    """ok / skip / override: are all valid."""
    blocks = [ReviewBOBlock(rationale="r", decisions={
        "A": "ok",
        "B": "skip",
        "C": "override:CCO",
    })]
    validate_semantics(blocks, phase="B_suggestions", use_rdkit=False)


def test_validate_semantics_review_analogs_empty_key() -> None:
    blocks = [ReviewAnalogsBlock(rationale="r", decisions={"": "keep"})]
    with pytest.raises(SemanticError, match="empty decision key"):
        validate_semantics(blocks, phase="A_review_analogs", use_rdkit=False)


def test_validate_semantics_no_use_rdkit_skips_rdkit_checks() -> None:
    """With use_rdkit=False, even garbage SMILES pass."""
    blocks = [ProposeBlock(rationale="p", smiles=["NOT_VALID_X"])]
    validate_semantics(blocks, phase="A_actions", use_rdkit=False)


# ---------------------------------------------------------------------------
# extract_json_payloads: bare JSON + array formats
# ---------------------------------------------------------------------------


def _extract(text: str) -> list:
    """Import inside the helper to avoid touching the top-of-file imports."""
    from strbo_v1.llm_advisor.parser import extract_json_payloads
    return extract_json_payloads(text)


def test_extract_bare_json_object() -> None:
    """A bare JSON object (no fences) is treated as a single action."""
    text = '{"type": "noop", "rationale": "r"}'
    payloads = _extract(text)
    assert len(payloads) == 1
    obj = json.loads(payloads[0])
    assert obj["type"] == "noop"
    assert obj["rationale"] == "r"


def test_extract_bare_json_object_with_whitespace() -> None:
    """Leading / trailing whitespace is tolerated."""
    text = '   \n  {"type": "noop", "rationale": "r"}  \n'
    payloads = _extract(text)
    assert len(payloads) == 1
    assert json.loads(payloads[0])["type"] == "noop"


def test_extract_bare_json_array() -> None:
    """A bare JSON array is split into one payload per element."""
    text = json.dumps([
        {"type": "propose", "rationale": "a", "smiles": ["CCO"]},
        {"type": "reject", "rationale": "b", "targets": ["CCN"]},
    ])
    payloads = _extract(text)
    assert len(payloads) == 2
    assert json.loads(payloads[0])["type"] == "propose"
    assert json.loads(payloads[1])["type"] == "reject"


def test_extract_single_element_array() -> None:
    """A 1-element JSON array is NOT collapsed to a bare object —
    it stays as a 1-element list (uniformity)."""
    text = '[{"type": "noop", "rationale": "r"}]'
    payloads = _extract(text)
    assert len(payloads) == 1
    assert json.loads(payloads[0])["type"] == "noop"


def test_extract_fenced_still_works() -> None:
    """Backward compat: ```json ... ``` blocks still work."""
    text = '```json\n{"type":"noop","rationale":"r"}\n```'
    payloads = _extract(text)
    assert len(payloads) == 1
    assert json.loads(payloads[0])["type"] == "noop"


def test_extract_multiple_fenced_blocks() -> None:
    """Multiple fenced blocks in left-to-right order."""
    text = (
        '```json\n{"type":"propose","smiles":["CCO"]}\n```\n'
        '```json\n{"type":"reject","targets":["CCN"]}\n```'
    )
    payloads = _extract(text)
    assert len(payloads) == 2
    assert json.loads(payloads[0])["type"] == "propose"
    assert json.loads(payloads[1])["type"] == "reject"


def test_extract_mixed_fenced_and_bare_picks_fenced() -> None:
    """If the response has both fenced and bare JSON, fenced takes
    priority (legacy behavior preserved)."""
    text = (
        '```json\n{"type":"noop","rationale":"fenced"}\n```\n'
        '{"type":"propose","rationale":"bare","smiles":["CCO"]}'
    )
    payloads = _extract(text)
    assert len(payloads) == 1
    assert json.loads(payloads[0])["rationale"] == "fenced"


def test_extract_invalid_json_raises_parse_error() -> None:
    """Plain text (no JSON at all) raises ParseError with a clear
    message that mentions both fenced and bare attempts."""
    with pytest.raises(ParseError, match="no valid JSON found"):
        _extract("just some prose, no json anywhere")


def test_extract_array_with_non_object_raises() -> None:
    """[1, 2, 3] is not a valid action array."""
    with pytest.raises(ParseError, match="must contain only objects"):
        _extract("[1, 2, 3]")


def test_extract_empty_array_raises() -> None:
    """[] (no actions) raises ParseError."""
    with pytest.raises(ParseError, match="empty JSON array"):
        _extract("[]")


def test_extract_empty_string_raises() -> None:
    """Empty / whitespace-only input raises ParseError."""
    with pytest.raises(ParseError, match="empty"):
        _extract("")
    with pytest.raises(ParseError, match="empty"):
        _extract("   \n  \t  ")


def test_extract_malformed_json_raises() -> None:
    """A string that looks like JSON but isn't parseable raises."""
    with pytest.raises(ParseError, match="no valid JSON found"):
        _extract('{"type": "noop", "rationale":')


# ---------------------------------------------------------------------------
# parse_blocks end-to-end with bare JSON
# ---------------------------------------------------------------------------


def test_parse_blocks_handles_bare_object() -> None:
    """Full parse_blocks call: bare JSON object -> 1 NoopBlock."""
    text = '{"type": "noop", "rationale": "hello"}'
    blocks = parse_blocks(text)
    assert len(blocks) == 1
    assert isinstance(blocks[0], NoopBlock)
    assert blocks[0].rationale == "hello"


def test_parse_blocks_handles_bare_array() -> None:
    """Full parse_blocks call: bare JSON array -> 2 blocks."""
    text = json.dumps([
        {"type": "noop", "rationale": "a"},
        {"type": "propose", "rationale": "b", "smiles": ["CCO"]},
    ])
    blocks = parse_blocks(text)
    assert len(blocks) == 2
    assert isinstance(blocks[0], NoopBlock)
    assert isinstance(blocks[1], ProposeBlock)
    assert blocks[1].smiles == ["CCO"]


# ---------------------------------------------------------------------------
# review_bo: empty decisions (the 0-picks case)
# ---------------------------------------------------------------------------


def test_review_bo_with_empty_decisions_is_valid() -> None:
    """Empty decisions is the correct answer when 0 BO picks exist."""
    text = '{"type": "review_bo", "rationale": "no picks", "decisions": {}}'
    blocks = parse_blocks(text)
    assert len(blocks) == 1
    assert isinstance(blocks[0], ReviewBOBlock)
    assert blocks[0].decisions == {}


def test_review_bo_with_nonempty_decisions_still_valid() -> None:
    """Regression: non-empty decisions still works."""
    text = '{"type": "review_bo", "rationale": "x", "decisions": {"CCO": "ok"}}'
    blocks = parse_blocks(text)
    assert blocks[0].decisions == {"CCO": "ok"}


def test_review_bo_with_bad_decision_value_still_rejected() -> None:
    """Per-value regex still rejects garbage values (e.g. "garbage")."""
    text = '{"type": "review_bo", "rationale": "x", "decisions": {"CCO": "garbage"}}'
    with pytest.raises(SchemaError):
        parse_blocks(text)


# ---------------------------------------------------------------------------
# pool_min_size: Noop rejected when pool is below minimum
# ---------------------------------------------------------------------------


def test_noop_rejected_when_pool_below_min() -> None:
    """Phase A Noop is rejected when the pool is below pool_min_size."""
    from strbo_v1.llm_advisor.parser import validate_semantics
    blocks = [NoopBlock(rationale="x")]
    with pytest.raises(SemanticError, match="pool has 1 SMILES"):
        validate_semantics(
            blocks, phase="A_actions",
            pool=["CCO"],
            pool_min_size=3,
        )


def test_noop_accepted_when_pool_at_or_above_min() -> None:
    """Phase A Noop passes when the pool is at or above pool_min_size."""
    from strbo_v1.llm_advisor.parser import validate_semantics
    blocks = [NoopBlock(rationale="x")]
    # Pool size 3 == min 3 -> accepted.
    validate_semantics(
        blocks, phase="A_actions",
        pool=["CCO", "CCN", "CCC"],
        pool_min_size=3,
    )
    # Pool size 5 > min 3 -> accepted.
    validate_semantics(
        blocks, phase="A_actions",
        pool=["CCO", "CCN", "CCC", "CCCC", "CCCCC"],
        pool_min_size=3,
    )


def test_noop_accepted_when_pool_min_size_unset() -> None:
    """When pool_min_size is None (default), the constraint is inactive."""
    from strbo_v1.llm_advisor.parser import validate_semantics
    blocks = [NoopBlock(rationale="x")]
    # Empty pool + pool_min_size=None -> accepted (no constraint).
    validate_semantics(blocks, phase="A_actions", pool=[], pool_min_size=None)
    # Empty pool + pool_min_size=0 -> also accepted (off).
    validate_semantics(blocks, phase="A_actions", pool=[], pool_min_size=0)


def test_propose_accepted_when_pool_below_min() -> None:
    """A propose block is the correct answer when the pool is below min."""
    from strbo_v1.llm_advisor.parser import validate_semantics
    blocks = [ProposeBlock(rationale="refill", smiles=["c1ccccc1N"])]
    validate_semantics(
        blocks, phase="A_actions",
        pool=["CCO"],
        pool_min_size=3,
    )
