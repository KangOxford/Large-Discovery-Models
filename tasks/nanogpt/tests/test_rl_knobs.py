"""Action-space invariants: schema coverage, constraints, canonical identity."""

from __future__ import annotations

import pytest

from tasks.nanogpt.core import rl_knobs as K


def test_knob_order_covers_the_schema_exactly():
    # Every fixed-length encoding is indexed by KNOB_ORDER, so a knob added to
    # the schema without being ordered would silently drop out of the features.
    assert set(K.KNOB_ORDER) == set(K.knob_names())
    assert set(K.DEFAULTS) == set(K.knob_names())
    assert len(K.KNOB_ORDER) == len(set(K.KNOB_ORDER))


def test_defaults_are_admissible():
    assert K.admit(K.DEFAULTS) == K.validate(K.DEFAULTS)


def test_defaults_match_the_committed_trainer_values():
    # "Baseline" has to mean the same thing here and in train.py, or the
    # reference point the reward is measured against is not the trainer's.
    import re, pathlib

    src = (pathlib.Path(K.__file__).parent.parent / "scripts" / "train.py").read_text()
    for name, expected in K.DEFAULTS.items():
        match = re.search(rf"^{name} = ([^#\n]+)", src, re.M)
        assert match, f"{name} is not a top-level assignment in train.py"
        assert eval(match.group(1).strip()) == expected, name  # noqa: S307


@pytest.mark.parametrize("total_batch", [262144, 524288, 1048576])
def test_device_batch_96_is_never_legal(total_batch):
    # Documented trap: every TOTAL_BATCH_SIZE choice is a power of two, so
    # 96 * 2048 can never divide it. A quarter of the raw choice grid is
    # unreachable and the domain must say so rather than let train.py assert.
    cfg = {**K.DEFAULTS, "DEVICE_BATCH_SIZE": 96, "TOTAL_BATCH_SIZE": total_batch}
    assert not K.divisibility_ok(cfg)
    with pytest.raises(K.ProposalError, match="not divisible"):
        K.check_hard_constraints(cfg)


def test_legal_batching_pairs_pass():
    for total_batch in (262144, 524288, 1048576):
        for device_batch in (32, 64, 128):
            cfg = {**K.DEFAULTS, "DEVICE_BATCH_SIZE": device_batch,
                   "TOTAL_BATCH_SIZE": total_batch}
            if total_batch % (device_batch * K.MAX_SEQ_LEN) == 0:
                K.check_hard_constraints(cfg)


def test_validate_rejects_out_of_range_and_bad_choices():
    with pytest.raises(K.ProposalError, match="DEPTH"):
        K.validate({**K.DEFAULTS, "DEPTH": 99})
    with pytest.raises(K.ProposalError, match="HEAD_DIM"):
        K.validate({**K.DEFAULTS, "HEAD_DIM": 77})
    with pytest.raises(K.ProposalError, match="missing knob"):
        K.validate({k: v for k, v in K.DEFAULTS.items() if k != "DEPTH"})


def test_validate_normalises_types_and_case():
    clean = K.validate({**K.DEFAULTS, "DEPTH": 8.0, "WINDOW_PATTERN": " sssl "})
    assert clean["DEPTH"] == 8 and isinstance(clean["DEPTH"], int)
    assert clean["WINDOW_PATTERN"] == "SSSL"


def test_canonical_key_is_stable_and_discriminating():
    a = K.validate(K.DEFAULTS)
    b = K.validate({**K.DEFAULTS, "WINDOW_PATTERN": "sssl", "DEPTH": 8.0})
    assert K.canonical_key(a) == K.canonical_key(b)
    assert K.canonical_key(a) != K.canonical_key({**a, "DEPTH": 9})
    # The budget is part of the identity: the same knobs at a different
    # wall-clock budget are a different measurement, not a cache hit.
    assert K.canonical_key(a, 300) != K.canonical_key(a, 150)


def test_geometry_matches_the_trainer_formulas():
    geo = K.geometry(K.validate(K.DEFAULTS))
    # depth 8 * aspect 64 = 512, already a multiple of head_dim 128
    assert geo["model_dim"] == 512
    assert geo["num_heads"] == 4
    # SSSL over 8 layers -> 2 long layers, last already long
    assert geo["long_frac"] == pytest.approx(0.25)
    # 524288 / (128 * 2048) = 2
    assert geo["grad_accum_steps"] == pytest.approx(2.0)


def test_model_dim_rounds_up_to_head_dim():
    # depth 3 * aspect 32 = 96, which is not a multiple of 128
    model_dim, heads = K.derive_shape(3, 32, 128)
    assert model_dim == 128 and heads == 1


def test_last_layer_is_always_full_context():
    assert K.window_sizes(4, "SSSS")[-1] == K.MAX_SEQ_LEN
    assert K.window_sizes(8, "SSSS").count(K.MAX_SEQ_LEN) == 1


def test_with_defaults_layers_pinned_under_proposed():
    cfg = K.with_defaults(["DEPTH"], {"DEPTH": 12}, {"HEAD_DIM": 64})
    assert cfg["DEPTH"] == 12 and cfg["HEAD_DIM"] == 64
    assert cfg["MATRIX_LR"] == K.DEFAULTS["MATRIX_LR"]


def test_no_analytic_objective_is_exposed():
    # The predecessor shipped surrogate_val_bpb() here and it was trained on.
    # Its ranking was inverted against the real trainer, so this module must
    # not offer anything that looks like a score.
    banned = [name for name in dir(K) if "bpb" in name.lower()]
    assert banned == [], f"rl_knobs must expose no objective estimate, found {banned}"
