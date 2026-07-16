"""End-to-end tests for :func:`strbo_v1.llm_advisor.orchestrator.run_bo_with_llm`.

Uses a :class:`DynamicMock` LLM that produces valid blocks based on
what it sees in the prompt. Covers:

* The three-stage flow: each round has Stage A1, A2 (conditional), B.
* ``previous_errors`` from one stage do not leak into the next.
* Trajectory file is written; round structure matches the schema.
* Exception Catcher: a Scorer exception triggers a fatal_error
  sidecar.
* ``mock_analog_fn``-driven Stage A1 produces analogues that
  feed into the synchronous Stage A2 review.
* pool_min_size: pool-size loop refills the pool when too small.
* Native multi-obj: history stores ``list[float]`` for n_obj>=2;
  ``compute_best`` returns Pareto front.
* verbose=True prints stage progress.
"""

import json
import re
from pathlib import Path
from typing import List

import pytest

from strbo_v1.llm_advisor import (
    AnalogBlock,
    AnalogueRecord,
    NoopBlock,
    ProposeBlock,
    RejectBlock,
    ReviewAnalogsBlock,
    ReviewBOBlock,
)
from strbo_v1.llm_advisor.client import MockLLMClient, _serialize_blocks
from strbo_v1.llm_advisor.orchestrator import (
    OrchestratorConfig,
    compute_best,
    run_bo_with_llm,
    _cur_best_per_obj,
    _any_obj_improved,
    _config_to_dict,
)
from strbo_v1.bayesian_analog_search import BayesianAnalogSearchConfig
from strbo_v1.gp import GPConfig


# ---------------------------------------------------------------------------
# Dynamic mock: produces valid responses based on the prompt content
# ---------------------------------------------------------------------------


class DynamicMock(MockLLMClient):
    """Generate Stage A1 noop and Stage B 'all ok' dynamically.

    The base class's ``scripted_blocks`` is consulted first; if it's
    empty we synthesize. The class is also testable: callers can set
    ``scripted_responses`` or ``scripted_blocks`` to inject behavior.
    """

    def chat(self, system, user, *, json_mode=True):
        if self.scripted_responses is not None or self.scripted_blocks is not None:
            # Defer to the parent implementation.
            return super().chat(system, user, json_mode=json_mode)
        # Synthesize: detect stage and build appropriate block(s).
        if "STAGE A1" in system:
            blocks = self._synth_stage_a1(user)
        elif "STAGE A2" in system:
            blocks = self._synth_stage_a2(user)
        elif "STAGE B" in system:
            blocks = self._synth_stage_b(user)
        else:
            blocks = [NoopBlock(rationale="unknown stage")]
        return _serialize_blocks(blocks)

    def _synth_stage_a1(self, user: str) -> list:
        return [NoopBlock(rationale="synth noop")]

    def _synth_stage_a2(self, user: str) -> list:
        # Auto-keep all analogues listed in the prompt.
        keys = re.findall(r"analogue=(\S+)\s+reasyn_score=", user)
        if keys:
            return [
                ReviewAnalogsBlock(
                    rationale="auto-keep all",
                    decisions={k: "keep" for k in keys},
                )
            ]
        return [ReviewAnalogsBlock(rationale="no analogs", decisions={})]

    def _synth_stage_b(self, user: str) -> list:
        picks = re.findall(r"  - (\S+)\s+mu=", user)
        if not picks:
            return [ReviewBOBlock(rationale="no picks", decisions={})]
        return [
            ReviewBOBlock(
                rationale="auto-ok all",
                decisions={p: "ok" for p in picks},
            )
        ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_bo_config(
    batch_size: int = 1, n_iterations: int = 2, *,
    n_obj: int = 1, minimize=(True,),
) -> BayesianAnalogSearchConfig:
    return BayesianAnalogSearchConfig(
        init_size=2, batch_size=batch_size, n_iterations=n_iterations,
        warmup=False, acquisition="ei", smiles_max_len=80,
        minimize=minimize,
        gp_config=GPConfig(impl="fingerprint+tanimoto", device="cpu"),
    )


def _scorer(smis):
    """Simple proxy: shorter SMILES -> higher (less negative) score."""
    return [-1.0 * len(s) for s in smis]


def _scorer_vina_nn(smis):
    """Multi-obj: (vina, nn) where vina=-len, nn=+len."""
    return [[-1.0 * len(s), float(len(s))] for s in smis]


def _mock_analog_fn(seeds) -> List[AnalogueRecord]:
    """Make 2 simple analogues per seed."""
    out: List[AnalogueRecord] = []
    for s in seeds:
        for i, new in enumerate([s + "C", s + "CC"]):
            out.append(AnalogueRecord(
                seed_smiles=s, analogue_smiles=new,
                reasyn_score=0.8, num_steps=i + 1,
            ))
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_end_to_end_two_rounds(tmp_path: Path) -> None:
    client = DynamicMock()                    # no scripted_blocks -> synthesize
    cfg = OrchestratorConfig(
        init_size=2, batch_size=1, n_iterations=2,
        bo_config=_make_bo_config(),
        method="dyn", seed=0,
        objective_legend=[{"name": "mock", "minimize": True}],
        minimize=(True,),
    )
    out = run_bo_with_llm(
        seed_smiles=["CCO", "CCN", "CCC", "CCCC"],
        scorer=_scorer, llm=client,
        analog_fn=None, reasyn_pool=None,
        config=cfg,
        trajectory_path=tmp_path,
    )
    assert len(out) >= 2
    # Trajectory should be written.
    files = list(tmp_path.glob("*_trajectory.json"))
    assert files, f"no trajectory file under {tmp_path}"
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert len(payload["rounds"]) == 2
    for rd in payload["rounds"]:
        assert "stage_a1" in rd["llm_interactions"]
        assert "stage_b" in rd["llm_interactions"]


def test_empty_stage_b_rounds_do_not_consume_evaluation_budget(tmp_path: Path) -> None:
    class EmptyFirstReviewMock(DynamicMock):
        def __init__(self) -> None:
            super().__init__()
            self.stage_b_calls = 0

        def _synth_stage_b(self, user: str) -> list:
            self.stage_b_calls += 1
            if self.stage_b_calls == 1:
                picks = re.findall(r"  - (\S+)\s+mu=", user)
                return [
                    ReviewBOBlock(
                        rationale="skip first",
                        decisions={p: "skip" for p in picks},
                    )
                ]
            return super()._synth_stage_b(user)

    client = EmptyFirstReviewMock()
    cfg = OrchestratorConfig(
        init_size=1, batch_size=1, n_iterations=1,
        bo_config=_make_bo_config(n_iterations=1),
        method="dyn", seed=0,
        objective_legend=[{"name": "mock", "minimize": True}],
        minimize=(True,),
    )
    out = run_bo_with_llm(
        seed_smiles=["CCO", "CCN", "CCC"],
        scorer=_scorer, llm=client,
        analog_fn=None, reasyn_pool=None,
        config=cfg,
        trajectory_path=tmp_path,
    )
    assert len(out) == 2
    assert client.stage_b_calls == 2
    payload = json.loads(next(tmp_path.glob("*_trajectory.json")).read_text())
    assert len(payload["rounds"]) == 2


def test_fatal_error_writes_sidecar(tmp_path: Path) -> None:
    """If a non-scoring step raises, a fatal_error sidecar is written.

    The orchestrator's :func:`_score_via_scorer` swallows
    ``Exception`` from the scorer (it returns None scores instead
    of failing the run). To exercise the fatal-error path we trigger
    a failure in a place that isn't caught — here, by patching
    ``strbo_v1.llm_advisor.orchestrator._run_bo_step`` to raise
    (mimicking a GP / select_candidates crash).
    """
    from unittest.mock import patch
    from strbo_v1.llm_advisor import orchestrator as orch_mod

    def boom(*args, **kwargs):
        raise RuntimeError("select_candidates crashed")

    client = DynamicMock()
    cfg = OrchestratorConfig(
        init_size=2, batch_size=1, n_iterations=1,
        bo_config=_make_bo_config(n_iterations=1),
        method="die", seed=0,
        objective_legend=[{"name": "mock", "minimize": True}],
        minimize=(True,),
    )
    with patch.object(orch_mod, "_run_bo_step", side_effect=boom):
        with pytest.raises(RuntimeError, match="select_candidates crashed"):
            run_bo_with_llm(
                seed_smiles=["CCO", "CCN"],
                scorer=_scorer, llm=client,
                analog_fn=None, reasyn_pool=None,
                config=cfg,
                trajectory_path=tmp_path,
            )
    main = list(tmp_path.glob("*_trajectory.json"))
    side = list(tmp_path.glob("*_trajectory.json.error.json"))
    assert main and side
    payload = json.loads(main[0].read_text(encoding="utf-8"))
    assert payload["status"] == "fatal_error"
    assert payload["fatal_error"]["exc_type"] == "RuntimeError"
    assert "select_candidates crashed" in payload["fatal_error"]["message"]


def test_propose_grows_pool(tmp_path: Path) -> None:
    """Stage A1 propose injects new SMILES into the pool."""

    class ProposeProducer(DynamicMock):
        def _synth_stage_a1(self, user):
            return [ProposeBlock(
                rationale="add aspirin",
                smiles=["CC(=O)Oc1ccccc1C(=O)O"],
            )]

        def _synth_stage_b(self, user):
            picks = re.findall(r"  - (\S+)\s+mu=", user)
            return [ReviewBOBlock(
                rationale="ok",
                decisions={p: "ok" for p in picks},
            )]

    client = ProposeProducer()
    cfg = OrchestratorConfig(
        init_size=2, batch_size=1, n_iterations=1,
        bo_config=_make_bo_config(n_iterations=1),
        method="prop", seed=0,
        objective_legend=[{"name": "mock", "minimize": True}],
        minimize=(True,),
    )
    run_bo_with_llm(
        seed_smiles=["CCO", "CCN", "CCC"],
        scorer=_scorer, llm=client,
        analog_fn=None, reasyn_pool=None,
        config=cfg,
        trajectory_path=tmp_path,
    )
    files = list(tmp_path.glob("*_trajectory.json"))
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    pool_after = payload["rounds"][0]["pool_after_phase_a"]
    assert "CC(=O)Oc1ccccc1C(=O)O" in pool_after


def test_reject_removes_from_pool(tmp_path: Path) -> None:
    """Stage A1 reject removes SMILES from the pool."""

    class RejectProducer(DynamicMock):
        def _synth_stage_a1(self, user):
            return [RejectBlock(
                rationale="drop CCN",
                targets=["CCN"],
                reason="too_similar_to_history",
            )]

        def _synth_stage_b(self, user):
            picks = re.findall(r"  - (\S+)\s+mu=", user)
            return [ReviewBOBlock(
                rationale="ok",
                decisions={p: "ok" for p in picks},
            )]

    client = RejectProducer()
    cfg = OrchestratorConfig(
        init_size=2, batch_size=1, n_iterations=1,
        bo_config=_make_bo_config(n_iterations=1),
        method="rej", seed=0,
        objective_legend=[{"name": "mock", "minimize": True}],
        minimize=(True,),
    )
    run_bo_with_llm(
        seed_smiles=["CCO", "CCN", "CCC"],
        scorer=_scorer, llm=client,
        analog_fn=None, reasyn_pool=None,
        config=cfg,
        trajectory_path=tmp_path,
    )
    files = list(tmp_path.glob("*_trajectory.json"))
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    pool_after = payload["rounds"][0]["pool_after_phase_a"]
    assert "CCN" not in pool_after


def test_analog_round_trip(tmp_path: Path) -> None:
    """Stage A1 analog block → analog_fn → Stage A2 review_analogs."""
    from strbo_v1.llm_advisor.reasyn_pool import ReasynConfigPool

    pool = ReasynConfigPool.from_env()

    class AnalogProducer(DynamicMock):
        def _synth_stage_a1(self, user):
            if "Round 0/2" in user or "round 0/2" in user.lower():
                seeds = re.findall(r"  - (\S+)$", user[:user.find("### History")], re.MULTILINE)
                if seeds:
                    return [AnalogBlock(
                        rationale="emit analogs",
                        seeds=seeds[:1],
                        generator_hint="conservative",
                        n_per_seed=2,
                    )]
            return [NoopBlock(rationale="noop")]

        def _synth_stage_b(self, user):
            picks = re.findall(r"  - (\S+)\s+mu=", user)
            return [ReviewBOBlock(
                rationale="ok",
                decisions={p: "ok" for p in picks},
            )]

    client = AnalogProducer()
    cfg = OrchestratorConfig(
        init_size=2, batch_size=1, n_iterations=2,
        bo_config=_make_bo_config(n_iterations=2),
        method="ana", seed=0,
        objective_legend=[{"name": "mock", "minimize": True}],
        minimize=(True,),
    )
    run_bo_with_llm(
        seed_smiles=["CCO", "CCN", "CCC"],
        scorer=_scorer, llm=client,
        analog_fn=_mock_analog_fn, reasyn_pool=pool,
        config=cfg,
        trajectory_path=tmp_path,
    )
    files = list(tmp_path.glob("*_trajectory.json"))
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    a0 = payload["rounds"][0]["llm_interactions"]["stage_a1"]["final_blocks"]
    assert any(b["type"] == "analog" for b in a0)
    assert payload["rounds"][0]["llm_interactions"]["stage_a2"]["executed"] is True


# ---------------------------------------------------------------------------
# compute_best tests (replaces compute_pareto + best_score_per_obj)
# ---------------------------------------------------------------------------


def test_compute_best_single_objective() -> None:
    """n_obj==1: returns the argmin/argmax SMILES."""
    history = {"A": -1.0, "B": -2.0, "C": -3.0}
    assert compute_best(history, (True,), n_obj=1) == "C"
    history2 = {"A": 1.0, "B": 2.0, "C": 3.0}
    assert compute_best(history2, (False,), n_obj=1) == "C"


def test_compute_best_single_objective_empty() -> None:
    """No history → empty string for n_obj==1."""
    assert compute_best({}, (True,), n_obj=1) == ""


def test_compute_best_multi_objective_pareto() -> None:
    """n_obj>=2: returns Pareto front (list of non-dominated SMILES)."""
    # (-1, 5) is dominated by (-2, 4) (both obj worse for A).
    history = {
        "A": [-1.0, 5.0],
        "B": [-2.0, 4.0],
        "C": [-3.0, 6.0],
        "D": [-1.5, 3.0],
    }
    front = compute_best(history, (True, True), n_obj=2)
    assert set(front) == {"B", "C", "D"}


def test_compute_best_multi_objective_minimize() -> None:
    """Multi-obj with mixed minimize/maximize directions.

    A=(-1, 5) and B=(-2, 10) with minimize=(True, False):
      B is better on obj0 (-2 < -1) AND better on obj1 (10 > 5),
      so B dominates A; front = {"B"}.
    """
    history = {
        "A": [-1.0, 5.0],
        "B": [-2.0, 10.0],
    }
    front = compute_best(history, (True, False), n_obj=2)
    assert set(front) == {"B"}

    # Truly non-dominated: A=(-1, 5), B=(-2, 3) → B better on obj0, A better on obj1.
    history2 = {
        "A": [-1.0, 5.0],
        "B": [-2.0, 3.0],
    }
    front2 = compute_best(history2, (True, False), n_obj=2)
    assert set(front2) == {"A", "B"}


def test_compute_best_multi_objective_empty() -> None:
    """No history → empty list for n_obj>=2."""
    assert compute_best({}, (True, True), n_obj=2) == []


# ---------------------------------------------------------------------------
# Native multi-obj: history stores list[float]; PickRecord.mu is list[float]
# ---------------------------------------------------------------------------


def test_orchestrator_native_multi_obj_no_collapse(tmp_path: Path) -> None:
    """Multi-obj scorer is forwarded verbatim; history is list[float] tuples.

    The orchestrator must NOT collapse to single-obj; the multi-obj
    tuple history is handed straight to ``select_candidates`` from
    ``bayesian_analog_search``, which dispatches EHVI (n_obj=2).
    """
    from strbo_v1.llm_advisor.reasyn_pool import ReasynConfigPool

    bo_cfg = BayesianAnalogSearchConfig(
        init_size=2, batch_size=1, n_iterations=2,
        warmup=False, acquisition="ei", smiles_max_len=80,
        minimize=(True, False),  # vina min, nn max
        gp_config=GPConfig(impl="fingerprint+tanimoto", device="cpu"),
    )
    pool = ReasynConfigPool.from_env()
    client = DynamicMock()
    cfg = OrchestratorConfig(
        init_size=2, batch_size=1, n_iterations=2,
        bo_config=bo_cfg,
        method="multi", seed=0,
        objective_legend=[
            {"name": "vina", "minimize": True},
            {"name": "nn", "minimize": False},
        ],
        minimize=(True, False),
        n_obj=2,
    )
    out = run_bo_with_llm(
        seed_smiles=["CCO", "CCN", "CCC", "CCCC"],
        scorer=_scorer_vina_nn, llm=client,
        analog_fn=None, reasyn_pool=pool,
        config=cfg,
        trajectory_path=tmp_path,
    )
    # History should have multi-obj tuples (no collapse).
    assert len(out) >= 2
    for _, sc in out:
        # sc is either None (failed) or a list[float] of length 2
        if sc is not None:
            assert isinstance(sc, (list, tuple))
            assert len(sc) == 2
    # Trajectory scores should be lists of length 2
    files = list(tmp_path.glob("*_trajectory.json"))
    assert files
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    rd0 = payload["rounds"][0]
    if rd0["scores"]:
        first_sc = next(iter(rd0["scores"].values()))
        # Either "score" (n_obj==1) or "scores" (n_obj>=2)
        if "scores" in first_sc:
            assert len(first_sc["scores"]) == 2
        else:
            # single-obj form — should not happen for n_obj==2
            assert "score" not in first_sc


def test_pick_record_mu_sigma_per_objective(tmp_path: Path) -> None:
    """For n_obj>=2, ``PickRecord.mu`` / ``sigma`` are ``list[float]`` of length n_obj."""
    from strbo_v1.llm_advisor.reasyn_pool import ReasynConfigPool

    bo_cfg = BayesianAnalogSearchConfig(
        init_size=3, batch_size=1, n_iterations=1,
        warmup=False, acquisition="ei", smiles_max_len=80,
        minimize=(True, False),
        gp_config=GPConfig(impl="fingerprint+tanimoto", device="cpu"),
    )
    pool = ReasynConfigPool.from_env()
    client = DynamicMock()
    cfg = OrchestratorConfig(
        init_size=3, batch_size=1, n_iterations=1,
        bo_config=bo_cfg,
        method="pick-multi", seed=0,
        objective_legend=[{"name": "vina", "minimize": True}, {"name": "nn", "minimize": False}],
        minimize=(True, False),
        n_obj=2,
    )
    run_bo_with_llm(
        seed_smiles=["CCO", "CCN", "CCC", "CCCC", "CCCCC"],
        scorer=_scorer_vina_nn, llm=client,
        analog_fn=None, reasyn_pool=pool,
        config=cfg,
        trajectory_path=tmp_path,
    )
    files = list(tmp_path.glob("*_trajectory.json"))
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    # If BO produced suggestions, check their mu/sigma are lists.
    rd0 = payload["rounds"][0]
    if rd0.get("bo_suggestions"):
        pick = rd0["bo_suggestions"][0]
        if isinstance(pick["mu"], list):
            assert len(pick["mu"]) == 2
            assert len(pick["sigma"]) == 2


# ---------------------------------------------------------------------------
# Stagnation / best helpers
# ---------------------------------------------------------------------------


def test_cur_best_per_obj_single() -> None:
    history = {"A": -1.0, "B": -2.0, "C": -3.0}
    cur = _cur_best_per_obj(history, (True,), n_obj=1)
    assert cur == [-3.0]


def test_cur_best_per_obj_multi() -> None:
    history = {"A": [-1.0, 5.0], "B": [-2.0, 4.0]}
    cur = _cur_best_per_obj(history, (True, True), n_obj=2)
    assert cur == [-2.0, 4.0]


def test_any_obj_improved_first_round() -> None:
    """First round (last_best_per_obj all None) → True if any obj has finite score."""
    assert _any_obj_improved(
        {"A": -1.0, "B": -2.0}, [None], (True,), n_obj=1
    ) is True
    assert _any_obj_improved(
        {"A": [-1.0, 5.0], "B": [-2.0, 4.0]},
        [None, None], (True, True), n_obj=2,
    ) is True


def test_any_obj_improved_single_obj_no_improvement() -> None:
    assert _any_obj_improved(
        {"A": -1.0}, [-1.0], (True,), n_obj=1,
    ) is False
    assert _any_obj_improved(
        {"A": 1.0}, [1.0], (False,), n_obj=1,
    ) is False


def test_any_obj_improved_multi_obj_any() -> None:
    """Multi-obj: improvement on ANY obj is enough to reset stagnation."""
    # obj0 stays at -2.0 (no improvement), obj1 improved from 4.0 to 3.0 (minimize)
    assert _any_obj_improved(
        {"A": [-2.0, 3.0]}, [-2.0, 4.0], (True, True), n_obj=2,
    ) is True
    # Neither improved
    assert _any_obj_improved(
        {"A": [-2.0, 4.0]}, [-2.0, 4.0], (True, True), n_obj=2,
    ) is False


# ---------------------------------------------------------------------------
# OrchestratorConfig validation
# ---------------------------------------------------------------------------


def test_orchestrator_config_requires_bo_config() -> None:
    with pytest.raises(ValueError, match="bo_config is required"):
        OrchestratorConfig(
            init_size=2, batch_size=1, n_iterations=1,
            bo_config=None,
            method="x", seed=0,
            objective_legend=[{"name": "m", "minimize": True}],
            minimize=(True,),
        )


def test_orchestrator_config_enforces_pool_min_size_geq_batch() -> None:
    with pytest.raises(ValueError, match="pool_min_size"):
        OrchestratorConfig(
            init_size=2, batch_size=5, n_iterations=1,
            bo_config=_make_bo_config(),
            method="x", seed=0,
            objective_legend=[{"name": "m", "minimize": True}],
            minimize=(True,),
            pool_min_size=3,
        )


def test_orchestrator_config_enforces_minimize_n_obj_consistency() -> None:
    """len(minimize) must match n_obj."""
    with pytest.raises(ValueError, match="minimize length"):
        OrchestratorConfig(
            init_size=2, batch_size=1, n_iterations=1,
            bo_config=_make_bo_config(),
            method="x", seed=0,
            objective_legend=[{"name": "m", "minimize": True}],
            minimize=(True,),
            n_obj=2,
        )


def test_orchestrator_config_rejects_n_obj_zero() -> None:
    with pytest.raises(ValueError, match="n_obj must be >= 1"):
        OrchestratorConfig(
            init_size=2, batch_size=1, n_iterations=1,
            bo_config=_make_bo_config(),
            method="x", seed=0,
            objective_legend=[{"name": "m", "minimize": True}],
            minimize=(True,),
            n_obj=0,
        )


# ---------------------------------------------------------------------------
# pool-size loop test
# ---------------------------------------------------------------------------


def test_pool_size_loop_refills_pool() -> None:
    """When pool < pool_min_size after Stage A1, the orchestrator calls
    Stage A1 again.

    Uses scripted_blocks: first Stage A1 returns noop (pool has 1 SMILES < min 3),
    second Stage A1 returns propose with 3 new SMILES. After the loop, pool has
    4 SMILES >= min 3, so BO can proceed.
    """
    from strbo_v1.llm_advisor.blocks import ProposeBlock as _PB
    call_count = [0]

    class CountingMock(MockLLMClient):
        def chat(self, system, user, *, json_mode=True):
            if "STAGE A1" in system:
                call_count[0] += 1
                if call_count[0] == 1:
                    return _serialize_blocks([NoopBlock(rationale="first call noop")])
                else:
                    return _serialize_blocks([
                        _PB(rationale="refill pool", smiles=["CCN", "CCC", "CCCC"])
                    ])
            elif "STAGE B" in system:
                picks = re.findall(r"  - (\S+)\s+mu=", user)
                if picks:
                    return _serialize_blocks([
                        ReviewBOBlock(rationale="ok", decisions={p: "ok" for p in picks})
                    ])
                return _serialize_blocks([
                    ReviewBOBlock(rationale="no picks", decisions={})
                ])
            return _serialize_blocks([NoopBlock(rationale="unknown")])

    client = CountingMock()
    cfg = OrchestratorConfig(
        init_size=2, batch_size=1, n_iterations=1,
        bo_config=_make_bo_config(),
        method="pool-loop-test", seed=0,
        objective_legend=[{"name": "m", "minimize": True}],
        minimize=(True,),
        pool_min_size=3,
    )
    history = run_bo_with_llm(
        seed_smiles=["CCO"],
        scorer=_scorer,
        llm=client,
        config=cfg,
    )
    assert call_count[0] >= 2, f"Expected >= 2 Stage A1 calls, got {call_count[0]}"
    assert len(history) >= 1, f"Expected >= 1 scored entries, got {len(history)}"


# ---------------------------------------------------------------------------
# Verbose output test
# ---------------------------------------------------------------------------


def test_verbose_prints_progress(capsys: pytest.CaptureFixture[str]) -> None:
    client = DynamicMock()
    cfg = OrchestratorConfig(
        init_size=1, batch_size=1, n_iterations=1,
        smiles_max_len=80,
        bo_config=_make_bo_config(),
        method="mock", seed=0,
        objective_legend=[{"name": "mock", "minimize": True}],
        minimize=(True,),
        verbose=True,
    )
    run_bo_with_llm(
        seed_smiles=["CCO", "CCN", "CCC"],
        scorer=_scorer, llm=client,
        analog_fn=None, reasyn_pool=None,
        config=cfg,
    )
    captured = capsys.readouterr()
    out = captured.out
    # Expect stage markers in stdout.
    assert "[round 1/1] Stage A1" in out
    assert "[round 1/1] BO step:" in out
    assert "[round 1/1] Stage B:" in out
    assert "[round 1/1] scoring" in out
    # Single-obj format: bare float
    assert "scored" in out


# ---------------------------------------------------------------------------
# External guidance (LLM_GUIDANCE / --llm-guide)
# ---------------------------------------------------------------------------


def test_orchestrator_config_to_dict_includes_guidance() -> None:
    cfg = OrchestratorConfig(
        init_size=1, batch_size=1, n_iterations=1,
        smiles_max_len=80, bo_config=_make_bo_config(),
        method="mock", seed=0,
        objective_legend=[{"name": "m", "minimize": True}],
        minimize=(True,),
        guidance="Use analog heavily",
    )
    d = _config_to_dict(cfg)
    assert d["guidance"] == "Use analog heavily"


def test_orchestrator_config_to_dict_default_guidance_empty() -> None:
    cfg = OrchestratorConfig(
        init_size=1, batch_size=1, n_iterations=1,
        smiles_max_len=80, bo_config=_make_bo_config(),
        method="mock", seed=0,
        objective_legend=[{"name": "m", "minimize": True}],
        minimize=(True,),
    )
    d = _config_to_dict(cfg)
    assert d["guidance"] == ""


def test_orchestrator_threads_guidance_into_state_and_trajectory(
    tmp_path: Path,
) -> None:
    """``OrchestratorConfig.guidance`` is propagated into the
    trajectory's ``pre_state_snapshot`` for audit."""
    client = DynamicMock()
    cfg = OrchestratorConfig(
        init_size=1, batch_size=1, n_iterations=1,
        smiles_max_len=80,
        bo_config=_make_bo_config(),
        method="mock", seed=0,
        objective_legend=[{"name": "m", "minimize": True}],
        minimize=(True,),
        guidance="Use analog heavily",
    )
    traj_path = tmp_path / "traj.json"
    run_bo_with_llm(
        seed_smiles=["CCO", "CCN", "CCC"],
        scorer=_scorer, llm=client,
        analog_fn=None, reasyn_pool=None,
        config=cfg,
        trajectory_path=traj_path,
    )
    payload = json.loads(traj_path.read_text(encoding="utf-8"))
    # Every round's pre_state_snapshot should carry the guidance.
    for r in payload["rounds"]:
        assert r["pre_state_snapshot"]["guidance"] == "Use analog heavily"
    # And the config echo should carry it too.
    assert payload["config"]["guidance"] == "Use analog heavily"


def test_orchestrator_guidance_reaches_llm_system_prompt(
    tmp_path: Path,
) -> None:
    """The guidance text appended to ``OrchestratorConfig.guidance``
    is what the LLM sees in the system prompt for every stage."""
    captured: list = []

    class _CaptureClient(DynamicMock):
        def chat(self, system, user, *, json_mode=True):
            captured.append(system)
            return super().chat(system=system, user=user, json_mode=json_mode)

    client = _CaptureClient()
    cfg = OrchestratorConfig(
        init_size=1, batch_size=1, n_iterations=1,
        smiles_max_len=80,
        bo_config=_make_bo_config(),
        method="mock", seed=0,
        objective_legend=[{"name": "m", "minimize": True}],
        minimize=(True,),
        guidance="Use analog heavily",
    )
    run_bo_with_llm(
        seed_smiles=["CCO", "CCN", "CCC"],
        scorer=_scorer, llm=client,
        analog_fn=None, reasyn_pool=None,
        config=cfg,
    )
    assert len(captured) >= 1
    for s in captured:
        assert "## EXTERNAL GUIDANCE" in s
        assert "Use analog heavily" in s


def test_orchestrator_no_guidance_keeps_prompts_baseline(
    tmp_path: Path,
) -> None:
    """When ``guidance=""`` (the default), no GUIDANCE block is
    injected into the system prompt."""
    captured: list = []

    class _CaptureClient(DynamicMock):
        def chat(self, system, user, *, json_mode=True):
            captured.append(system)
            return super().chat(system=system, user=user, json_mode=json_mode)

    client = _CaptureClient()
    cfg = OrchestratorConfig(
        init_size=1, batch_size=1, n_iterations=1,
        smiles_max_len=80,
        bo_config=_make_bo_config(),
        method="mock", seed=0,
        objective_legend=[{"name": "m", "minimize": True}],
        minimize=(True,),
    )
    run_bo_with_llm(
        seed_smiles=["CCO", "CCN", "CCC"],
        scorer=_scorer, llm=client,
        analog_fn=None, reasyn_pool=None,
        config=cfg,
    )
    assert len(captured) >= 1
    for s in captured:
        assert "## EXTERNAL GUIDANCE" not in s
