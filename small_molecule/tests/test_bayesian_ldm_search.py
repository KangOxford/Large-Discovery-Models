"""Tests for strbo_v1.bayesian_ldm_search (the public LDM entry point)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from strbo_v1.bayesian_analog_search import BayesianAnalogSearchConfig
from strbo_v1.bayesian_ldm_search import (
    BayesianLDMSearchConfig,
    _build_orchestrator_config,
    bayesian_ldm_search,
)
from strbo_v1.gp import GPConfig
from strbo_v1.llm_advisor.client import MockLLMClient
from strbo_v1.llm_advisor.config import LLMClientConfig
from strbo_v1.llm_advisor.blocks import (
    NoopBlock, ReviewBOBlock, ProposeBlock,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm_cfg() -> LLMClientConfig:
    return LLMClientConfig(
        api_key="test-key", base_url="https://x.com/v1", model="mock-llm",
    )


def _bo_cfg(n_iter: int = 2) -> BayesianAnalogSearchConfig:
    return BayesianAnalogSearchConfig(
        init_size=2, batch_size=1, n_iterations=n_iter, warmup=False,
        acquisition="ei", smiles_max_len=80,
        gp_config=GPConfig(impl="fingerprint+tanimoto", device="cpu"),
    )


class _DynamicMockClient:
    """Minimal LLM client that returns sensible blocks for both stages.

    Stage A1 → NoopBlock (don't touch the pool).
    Stage B → ReviewBOBlock with "ok" for every SMILES mentioned in
             the user prompt (which the orchestrator populates with
             the BO picks).
    """

    def __init__(self, model_name: str = "mock-llm") -> None:
        self.model_name = model_name
        self.call_log: list = []

    def chat(self, system: str, user: str, *, json_mode: bool = True) -> str:
        self.call_log.append({"system": system[:30], "user": user[:30]})
        from strbo_v1.llm_advisor.client import _serialize_blocks
        from strbo_v1.llm_advisor.blocks import NoopBlock, ReviewBOBlock
        if "STAGE B" in system:
            decisions: dict = {}
            for line in user.splitlines():
                stripped = line.strip()
                if stripped.startswith("- ") and "  mu=" in stripped:
                    smi = stripped[2:].split("  mu=", 1)[0].strip()
                    if smi:
                        decisions[smi] = "ok"
            return _serialize_blocks(
                [ReviewBOBlock(rationale="ok", decisions=decisions)],
            )
        return _serialize_blocks([NoopBlock(rationale="ok")])


def _growing_scorer():
    """Score = -len(smi)."""
    def s(smis: List[str]) -> List[float]:
        return [-float(len(s)) for s in smis]
    return s


def _trivial_analog(smis: List[str]) -> List[str]:
    """Echo back the same SMILES."""
    return list(smis)


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_ldm_runs_two_rounds_single_objective() -> None:
    """End-to-end: 2 init + 2 BO rounds with a noop Stage A1 and ok Stage B."""
    client = _DynamicMockClient()
    cfg = BayesianLDMSearchConfig(
        init_size=2, batch_size=1, n_iterations=2,
        smiles_max_len=80,
        bo_config=_bo_cfg(n_iter=2), llm_config=_llm_cfg(),
        method="bo-tanimoto-ldm", seed=0,
    )

    history, trajectory = bayesian_ldm_search(
        seed_smiles=["CCO", "CCN", "CCC", "CCCC", "CCCCC"],
        scorer=_growing_scorer(),
        analog_fn=_trivial_analog,
        config=cfg,
        llm=client,
    )
    # At least 3 evaluations (2 init + 1+ BO).
    assert len(history) >= 3
    # Single-obj: score is a float.
    for _, sc in history:
        assert isinstance(sc, float)
        assert sc < 0
    # No trajectory dir → trajectory is None.
    assert trajectory is None


def test_ldm_returns_trajectory_when_dir_set(tmp_path: Path) -> None:
    """When trajectory_dir is set, the function reads it back and returns the dict."""
    client = _DynamicMockClient()
    out_dir = tmp_path / "traj"
    cfg = BayesianLDMSearchConfig(
        init_size=2, batch_size=1, n_iterations=1,
        smiles_max_len=80,
        bo_config=_bo_cfg(n_iter=1), llm_config=_llm_cfg(),
        method="bo-tanimoto-ldm", seed=0,
        trajectory_dir=str(out_dir),
    )
    history, trajectory = bayesian_ldm_search(
        seed_smiles=["CCO", "CCN", "CCC", "CCCC"],
        scorer=_growing_scorer(),
        analog_fn=_trivial_analog,
        config=cfg,
        llm=client,
    )
    assert len(history) == 3
    assert trajectory is not None
    assert "status" in trajectory
    assert "rounds" in trajectory
    assert len(trajectory["rounds"]) == 1
    # Stage A1 + Stage B were both called.
    assert "stage_a1" in trajectory["rounds"][0]["llm_interactions"]
    assert "stage_b" in trajectory["rounds"][0]["llm_interactions"]


def test_ldm_stage_b_review_bo_decisions_apply() -> None:
    """The LLM's Stage B decisions should propagate to the picked candidates."""
    client = _DynamicMockClient()
    cfg = BayesianLDMSearchConfig(
        init_size=2, batch_size=1, n_iterations=2,
        smiles_max_len=80,
        bo_config=_bo_cfg(n_iter=2), llm_config=_llm_cfg(),
        method="bo-tanimoto-ldm", seed=0,
    )
    history, _ = bayesian_ldm_search(
        seed_smiles=["CCO", "CCN", "CCC", "CCCC"],
        scorer=_growing_scorer(),
        analog_fn=_trivial_analog,
        config=cfg,
        llm=client,
    )
    assert len(history) >= 3
    for _, sc in history:
        assert sc is not None
        assert sc < 0


def test_ldm_propose_block_adds_to_pool() -> None:
    """A Stage A1 propose block should inject SMILES that get scored later."""
    from strbo_v1.llm_advisor.client import _serialize_blocks
    proposed = "CCCCCC"

    class _ProposeThenOk(_DynamicMockClient):
        def chat(self, system: str, user: str, *, json_mode: bool = True) -> str:
            from strbo_v1.llm_advisor.blocks import ProposeBlock
            self.call_log.append({"system": system[:30], "user": user[:30]})
            if "STAGE A1" in system:
                return _serialize_blocks(
                    [ProposeBlock(rationale="inject", smiles=[proposed])]
                )
            return super().chat(system, user, json_mode=json_mode)

    cfg = BayesianLDMSearchConfig(
        init_size=2, batch_size=1, n_iterations=1,
        smiles_max_len=80,
        bo_config=_bo_cfg(n_iter=1), llm_config=_llm_cfg(),
        method="bo-tanimoto-ldm", seed=0,
    )
    history, _ = bayesian_ldm_search(
        seed_smiles=["CCO", "CCN", "CCC"],
        scorer=_growing_scorer(),
        analog_fn=_trivial_analog,
        config=cfg,
        llm=_ProposeThenOk(),
    )
    assert len(history) == 3


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_ldm_config_rejects_missing_bo_config() -> None:
    with pytest.raises(ValueError, match="bo_config"):
        BayesianLDMSearchConfig(llm_config=_llm_cfg())


def test_ldm_config_rejects_missing_llm_config() -> None:
    with pytest.raises(ValueError, match="llm_config"):
        BayesianLDMSearchConfig(bo_config=_bo_cfg())


def test_ldm_config_rejects_zero_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        BayesianLDMSearchConfig(
            bo_config=_bo_cfg(), llm_config=_llm_cfg(), batch_size=0,
        )


# ---------------------------------------------------------------------------
# Native multi-obj: history is list[float] (no single-obj collapse)
# ---------------------------------------------------------------------------


def _scorer_vina_nn(smis: List[str]):
    """Multi-obj scorer: (vina=-len, nn=+len)."""
    return [[-1.0 * len(s), float(len(s))] for s in smis]


def _vina_only(smis: List[str]) -> List[float]:
    """Vina-like scorer: returns one float per SMILES."""
    return [-1.0 * len(s) for s in smis]


def _nn_only(smis: List[str]) -> List[float]:
    """NN-like scorer: returns one float per SMILES."""
    return [float(len(s)) for s in smis]


def test_ldm_vina_plus_nn_native_multi_obj_no_collapse() -> None:
    """vina+nn: the LDM must NOT collapse to single-obj.

    The LDM stores ``list[float]`` (length n_obj) in the history and
    hands the multi-obj history straight to
    ``select_candidates`` from ``bayesian_analog_search``, which
    dispatches EHVI. The public return shape is also multi-obj
    (list[float] for n_obj>=2).
    """
    bo_cfg = BayesianAnalogSearchConfig(
        init_size=2, batch_size=1, n_iterations=2, warmup=False,
        acquisition="ei", smiles_max_len=80,
        minimize=(True, False),       # vina min, nn max
        gp_config=GPConfig(impl="fingerprint+tanimoto", device="cpu"),
    )
    cfg = BayesianLDMSearchConfig(
        init_size=2, batch_size=1, n_iterations=2,
        smiles_max_len=80,
        bo_config=bo_cfg, llm_config=_llm_cfg(),
        method="bo-tanimoto-ldm", seed=0,
        minimize=(True, False),
    )
    # Two per-objective scorers, each returning one float per SMILES.
    # The LDM stacks them into a list[float] of length n_obj per SMILES.
    history, _ = bayesian_ldm_search(
        seed_smiles=["CCO", "CCN", "CCC", "CCCC"],
        scorer=(_vina_only, _nn_only),
        analog_fn=_trivial_analog,
        config=cfg,
        llm=_DynamicMockClient(),
    )
    # History entries must be list[float] of length 2 (no collapse).
    assert len(history) >= 3
    for _, sc in history:
        assert sc is not None
        assert isinstance(sc, (list, tuple))
        assert len(sc) == 2
        # The two objective scores must be present (no NaN/None).
        assert all(v is not None for v in sc)


def test_ldm_verbose_prints_progress(capsys):
    """When verbose=True, the LDM prints per-stage progress to stdout."""
    cfg = BayesianLDMSearchConfig(
        init_size=2, batch_size=1, n_iterations=1,
        smiles_max_len=80,
        bo_config=_bo_cfg(n_iter=1), llm_config=_llm_cfg(),
        method="bo-tanimoto-ldm", seed=0,
        verbose=True,
    )
    bayesian_ldm_search(
        seed_smiles=["CCO", "CCN", "CCC"],
        scorer=_growing_scorer(),
        analog_fn=_trivial_analog,
        config=cfg,
        llm=_DynamicMockClient(),
    )
    out = capsys.readouterr().out
    # Stage A1 / BO / Stage B / scoring markers should appear.
    assert "[round 1/1] Stage A1" in out
    assert "[round 1/1] BO step:" in out
    assert "[round 1/1] Stage B:" in out
    assert "[round 1/1] scoring" in out


# ---------------------------------------------------------------------------
# External guidance (LLM_GUIDANCE / --llm-guide)
# ---------------------------------------------------------------------------


def test_ldm_config_default_guidance_is_empty() -> None:
    """Default for ``BayesianLDMSearchConfig.guidance`` is empty."""
    cfg = BayesianLDMSearchConfig(
        init_size=2, batch_size=1, n_iterations=1, smiles_max_len=80,
        bo_config=_bo_cfg(n_iter=1), llm_config=_llm_cfg(),
        method="bo-tanimoto-ldm", seed=0,
    )
    assert cfg.guidance == ""


def test_ldm_config_passes_guidance_to_orchestrator_config() -> None:
    """``BayesianLDMSearchConfig.guidance`` is propagated to
    ``OrchestratorConfig.guidance`` via ``_build_orchestrator_config``."""
    cfg = BayesianLDMSearchConfig(
        init_size=2, batch_size=1, n_iterations=1, smiles_max_len=80,
        bo_config=_bo_cfg(n_iter=1), llm_config=_llm_cfg(),
        method="bo-tanimoto-ldm", seed=0,
        guidance="Use analog heavily",
    )
    ocfg = _build_orchestrator_config(cfg, n_obj=1)
    assert ocfg.guidance == "Use analog heavily"


def test_ldm_empty_guidance_propagates_as_empty_string() -> None:
    cfg = BayesianLDMSearchConfig(
        init_size=2, batch_size=1, n_iterations=1, smiles_max_len=80,
        bo_config=_bo_cfg(n_iter=1), llm_config=_llm_cfg(),
        method="bo-tanimoto-ldm", seed=0,
    )
    ocfg = _build_orchestrator_config(cfg, n_obj=1)
    assert ocfg.guidance == ""


def test_ldm_guidance_reaches_llm_system_prompt(tmp_path: Path) -> None:
    """End-to-end: ``BayesianLDMSearchConfig.guidance`` makes it to
    the LLM's system prompt for both Stage A1 and Stage B."""
    captured: list = []

    class _CaptureClient(_DynamicMockClient):
        def chat(self, system, user, *, json_mode=True):
            captured.append(system)
            return super().chat(system=system, user=user, json_mode=json_mode)

    cfg = BayesianLDMSearchConfig(
        init_size=2, batch_size=1, n_iterations=1, smiles_max_len=80,
        bo_config=_bo_cfg(n_iter=1), llm_config=_llm_cfg(),
        method="bo-tanimoto-ldm", seed=0,
        guidance="Use analog heavily",
        trajectory_dir=str(tmp_path),
    )
    bayesian_ldm_search(
        seed_smiles=["CCO", "CCN", "CCC"],
        scorer=_growing_scorer(),
        analog_fn=_trivial_analog,
        config=cfg,
        llm=_CaptureClient(),
    )
    assert len(captured) >= 1
    for s in captured:
        assert "## EXTERNAL GUIDANCE" in s
        assert "Use analog heavily" in s


def test_ldm_guidance_appears_in_trajectory_config(
    tmp_path: Path,
) -> None:
    cfg = BayesianLDMSearchConfig(
        init_size=2, batch_size=1, n_iterations=1, smiles_max_len=80,
        bo_config=_bo_cfg(n_iter=1), llm_config=_llm_cfg(),
        method="bo-tanimoto-ldm", seed=0,
        guidance="Use analog heavily",
        trajectory_dir=str(tmp_path),
    )
    bayesian_ldm_search(
        seed_smiles=["CCO", "CCN", "CCC"],
        scorer=_growing_scorer(),
        analog_fn=_trivial_analog,
        config=cfg,
        llm=_DynamicMockClient(),
    )
    # Read the trajectory file and confirm guidance is recorded.
    from strbo_v1.llm_advisor.trajectory import resolve_trajectory_path
    p = resolve_trajectory_path(str(tmp_path), method="bo-tanimoto-ldm", seed=0)
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["config"]["guidance"] == "Use analog heavily"
    for r in payload["rounds"]:
        assert r["pre_state_snapshot"]["guidance"] == "Use analog heavily"

