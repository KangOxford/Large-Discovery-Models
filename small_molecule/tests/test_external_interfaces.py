"""Tests for public external scorer/acquisition interfaces."""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

from extract_and_dock import DockingResult
from strbo_v1.gp import GPConfig


class _FakeScoreScorer:
    def __init__(self, values: dict[str, float]) -> None:
        self.values = values

    def __call__(self, smiles: Sequence[str]) -> list[float]:
        return [self.values.get(s, float("nan")) for s in smiles]


class _FakeVinaScorer(_FakeScoreScorer):
    def _dock_smiles(self, smiles: Sequence[str]) -> list[DockingResult]:
        out = []
        for smi in smiles:
            value = self.values.get(smi)
            if value is None or math.isnan(float(value)):
                out.append(DockingResult(
                    compound_id=smi,
                    canonical_smiles=smi,
                    score=None,
                    pose_ref=None,
                    status="dock_failed",
                    message="bad molecule",
                ))
                continue
            out.append(DockingResult(
                compound_id=smi,
                canonical_smiles=smi,
                score=value,
                pose_ref=f"/poses/{smi}.pdbqt",
                status="ok",
            ))
        return out


class _FakeSurrogate:
    fit_calls = 0

    def __init__(self, config: GPConfig) -> None:
        self.config = config

    def fit(self, smiles: list[str], scores: list[float]) -> "_FakeSurrogate":
        type(self).fit_calls += 1
        return self

    def predict(self, smiles: list[str], *, return_tensor: bool = False):
        means = {"CCCO": -6.8, "CCCN": -6.1}
        return [means[str(s)] for s in smiles], [0.5 for _ in smiles]


@dataclass
class _ExternalTestConfig:
    acquisition: object = "ei"
    minimize: object = True
    xi: float = 0.01
    kappa: float = 2.0
    gp_config: GPConfig = field(
        default_factory=lambda: GPConfig(device="cpu", fit_n_itersteps=1)
    )
    ref_point: tuple[float, ...] | None = None
    ehvi_n_samples: int = 8
    che_alpha: float = 1.0


class ExternalInterfacesTests(unittest.TestCase):
    def test_score_vina_returns_structured_per_item_results(self) -> None:
        from strbo_v1.external_interfaces import score_vina

        result = score_vina(
            ["CCO", "BAD"],
            scorer_factory=lambda _cfg: _FakeVinaScorer({"CCO": -7.5}),
        )

        self.assertTrue(result["ok"])
        self.assertEqual(len(result["items"]), 2)
        self.assertEqual(result["items"][0]["smiles"], "CCO")
        self.assertTrue(result["items"][0]["ok"])
        self.assertEqual(result["items"][0]["value"], -7.5)
        self.assertEqual(result["items"][0]["details"]["status"], "ok")
        self.assertFalse(result["items"][1]["ok"])
        self.assertIn("bad molecule", result["items"][1]["error"])

    def test_score_nn_returns_structured_per_item_results(self) -> None:
        from strbo_v1.external_interfaces import score_nn

        result = score_nn(
            ["CCO", "BAD"],
            scorer_factory=lambda _cfg: _FakeScoreScorer({"CCO": 6.2}),
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["items"][0]["value"], 6.2)
        self.assertTrue(math.isnan(result["items"][1]["details"]["raw_value"]))
        self.assertFalse(result["items"][1]["ok"])
        self.assertIn("non-finite", result["items"][1]["error"])

    def test_score_nn_default_config_uses_g12d_model(self) -> None:
        from strbo_v1.experiment_defaults import DEFAULT_NN_MODEL_PATH
        from strbo_v1.external_interfaces import build_nn_config

        cfg = build_nn_config()

        self.assertEqual(cfg.model_path, DEFAULT_NN_MODEL_PATH)

    def test_evaluate_acquisition_single_objective_is_structured(self) -> None:
        from strbo_v1.external_interfaces import evaluate_acquisition

        _FakeSurrogate.fit_calls = 0
        result = evaluate_acquisition(
            history=[{"smiles": "CCO", "score": -7.1}, {"smiles": "CCN", "score": -6.4}],
            query_smiles=["CCCO", "CCCN"],
            config=_ExternalTestConfig(acquisition=("ei", "pi")),
            surrogate_factory=_FakeSurrogate,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(_FakeSurrogate.fit_calls, 1)
        first = result["items"][0]
        self.assertEqual(first["smiles"], "CCCO")
        self.assertTrue(first["ok"])
        self.assertIn("acquisition_ei", first["details"])
        self.assertIn("mean", first["details"])
        self.assertEqual(first["value"], first["details"]["acquisition_ei"])

    def test_evaluate_acquisition_two_objective_returns_ehvi(self) -> None:
        from strbo_v1.external_interfaces import evaluate_acquisition

        result = evaluate_acquisition(
            history=[
                {"smiles": "CCO", "scores": [-7.1, 5.1]},
                {"smiles": "CCN", "scores": [-6.4, 5.8]},
            ],
            query_smiles=["CCCO", "CCCN"],
            config=_ExternalTestConfig(
                minimize=(True, False),
                ref_point=(0.0, 4.0),
                ehvi_n_samples=4,
            ),
            surrogate_factory=_FakeSurrogate,
        )

        self.assertTrue(result["ok"])
        item = result["items"][0]
        self.assertIn("acquisition_ehvi", item["details"])
        self.assertIn("objectives", item["details"])
        self.assertEqual(len(item["details"]["objectives"]), 2)

    def test_score_vina_uses_config_cache_dir(self) -> None:
        from strbo_v1.external_interfaces import build_vina_config

        with TemporaryDirectory() as tmp:
            cfg = build_vina_config({"vina_pdb_id": "8UN5"}, cache_dir=tmp)

        self.assertEqual(Path(cfg.cache_dir), Path(tmp))
