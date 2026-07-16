#!/usr/bin/env python
"""Standalone smoke test for the reservoir LDM prototype.

This uses a fake GP/acquisition so it does not need Absolut, an API key, or a
full BO run. It validates the discrete algorithmic loop:
5 strategies -> 5 representatives -> argmax/softmax selected candidate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from bo.ldm.dsl.alphabet import idx_to_aa
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
    # Higher first amino-acid index is better; deterministic and easy to verify.
    return x[:, 0].float()


def seq_to_str(seq) -> str:
    return "".join(idx_to_aa(int(i)) for i in seq)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", choices=["argmax", "softmax"], default="argmax")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    anchors = ["AAAAAAAAAAA", "CAAAAAAAAAA", "DAAAAAAAAAA", "EAAAAAAAAAA", "FAAAAAAAAAA"]
    dsl = Or(*(NeighborSampling(anchor, radius=0, budget=1) for anchor in anchors))
    cfg = ReservoirLDMConfig(
        n_strategies=5,
        per_strategy_budget=1,
        selection_mode=args.selection,
        selection_score="acq",
        pool_score="combined",
        softmax_eta=5.0,
        rng_seed=0,
    )
    session = ReservoirAcquisitionSession(cfg, acq_name="ei")
    selected = session.run(
        strategies=dsl,
        bias_dsl=None,
        gp=FakeGP(),
        f_acq=fake_acq,
        batch_size=1,
        cat_config=np.array([20] * 11),
        cdr_constraints=False,
        device=torch.device("cpu"),
    )

    print("selected:", seq_to_str(selected[0]), selected.tolist())
    print("probabilities:", session.last_record["probabilities"])
    print("representatives:")
    for rec in session.last_record["representatives"]:
        print(f"  strategy={rec.get('strategy_idx')} seq={rec['seq_str']} ei={rec['ei']:.3f}")

    if args.plot:
        import matplotlib.pyplot as plt

        out_dir = Path("outputs/ldm_reservoir_smoke")
        out_dir.mkdir(parents=True, exist_ok=True)
        reps = session.last_record["representatives"]
        labels = [r["seq_str"] for r in reps]
        scores = [r["ei"] for r in reps]
        probs = session.last_record["probabilities"]
        fig, ax1 = plt.subplots(figsize=(7, 3.5))
        ax1.bar(labels, scores, color="#4C78A8", alpha=0.8, label="EI")
        ax1.set_ylabel("acquisition")
        ax1.tick_params(axis="x", rotation=30)
        ax2 = ax1.twinx()
        ax2.plot(labels, probs, color="#F58518", marker="o", label="selection prob")
        ax2.set_ylabel("probability")
        fig.tight_layout()
        out = out_dir / "candidate_scores.png"
        fig.savefig(out, dpi=160)
        print("plot:", out)


if __name__ == "__main__":
    main()
