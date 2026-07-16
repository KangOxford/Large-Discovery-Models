from __future__ import annotations

import numpy as np
import torch

from bo.ldm.dsl.search_space import NeighborSampling, Or
from bo.ldm_reservoir import ReservoirAcquisitionSession, ReservoirLDMConfig


class FakePosterior:
    def __init__(self, x: torch.Tensor) -> None:
        self.mean = x[:, 0].float()
        self.stddev = torch.ones(x.shape[0], device=x.device)


class FakeGP:
    def __call__(self, x: torch.Tensor) -> FakePosterior:
        return FakePosterior(x)

    def likelihood(self, posterior: FakePosterior) -> FakePosterior:
        return posterior


def fake_acq(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0].float()


def make_five_anchor_dsl():
    anchors = ["AAAAAAAAAAA", "CAAAAAAAAAA", "DAAAAAAAAAA", "EAAAAAAAAAA", "FAAAAAAAAAA"]
    return Or(*(NeighborSampling(anchor, radius=0, budget=1) for anchor in anchors))


def run_session(cfg, strategies=None, fallback_center=None):
    session = ReservoirAcquisitionSession(cfg, acq_name="ei")
    selected = session.run(
        strategies=strategies or make_five_anchor_dsl(),
        bias_dsl=None,
        gp=FakeGP(),
        f_acq=fake_acq,
        batch_size=1,
        cat_config=np.array([20] * 11),
        cdr_constraints=False,
        device=torch.device("cpu"),
        fallback_center=fallback_center,
    )
    return selected, session


def test_argmax_selects_best_representative():
    cfg = ReservoirLDMConfig(
        n_strategies=5,
        per_strategy_budget=1,
        selection_mode="argmax",
        selection_score="acq",
    )
    selected, session = run_session(cfg)

    assert selected.shape == (1, 11)
    assert selected[0, 0] == 4  # F has index 4 in ACDEFGHIKLMNPQRSTVWY
    assert len(session.last_record["representatives"]) == 5
    assert session.last_record["selected_ids"] == [4]


def test_softmax_records_valid_probabilities():
    cfg = ReservoirLDMConfig(
        n_strategies=5,
        per_strategy_budget=1,
        selection_mode="softmax",
        selection_score="acq",
        softmax_eta=1.0,
        rng_seed=0,
    )
    selected, session = run_session(cfg)

    probs = session.last_record["probabilities"]
    assert selected.shape == (1, 11)
    assert len(probs) == 5
    assert abs(sum(probs) - 1.0) < 1e-8
    assert probs[-1] > probs[0]


def test_fills_missing_llm_strategies_from_fallback_center():
    cfg = ReservoirLDMConfig(
        n_strategies=5,
        per_strategy_budget=1,
        selection_mode="argmax",
        selection_score="acq",
    )
    one_strategy = NeighborSampling("AAAAAAAAAAA", radius=0, budget=1)
    selected, session = run_session(cfg, strategies=one_strategy, fallback_center=[0] * 11)

    assert selected.shape == (1, 11)
    assert session.last_record["n_strategies_executed"] == 5
    assert len(session.strategy_results) == 5


def test_large_strategy_budget_is_capped():
    cfg = ReservoirLDMConfig(
        n_strategies=5,
        per_strategy_budget=1,
        selection_mode="argmax",
        selection_score="acq",
    )
    too_large = Or(*(NeighborSampling("AAAAAAAAAAA", radius=0, budget=1000) for _ in range(5)))
    _, session = run_session(cfg, strategies=too_large)

    assert all("budget=1" in result.atom_repr for result in session.strategy_results)
