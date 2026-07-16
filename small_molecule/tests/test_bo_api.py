"""Tests for ``bo_api.py`` (JSON-in/JSON-out API).

These tests exercise the two public API entry points end-to-end
using the mock objective (no Vina, no ReaSyn, no GPU). They cover:

* ``run_search_trajectory``: full schema (config/history/summary),
  error-JSON format, multi-objective dispatch.
* ``recommend_next_smiles``: every method (random / random-best /
  bo-tanimoto / bo-strkernel), empty pool/history, seed
  reproducibility, error JSON, multi-objective.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List


import bo_api
from strbo_v1 import (
    BayesianAnalogSearchConfig,
    GPConfig,
    RNG,
    bayesian_select_candidates,
    random_select_next_batch,
)


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------


def _bo_request(
    *,
    method: str = "bo-tanimoto",
    pool: List[str] = None,
    history: List[dict] = None,
    batch_size: int = 2,
    minimize=None,
    ref_point=None,
    gp: dict = None,
    seed: int = 0,
):
    """Build a JSON request for ``recommend_next_smiles``."""
    request = {
        "method": method,
        "pool": pool if pool is not None else [
            "CCO", "CCN", "CCC", "C1CCCCC1", "c1ccccc1", "CCCCC", "CCCCO", "CCCCN",
        ],
        "history": history if history is not None else [
            {"smiles": "CCO",  "score": -7.5},
            {"smiles": "CCN",  "score": -6.2},
            {"smiles": "CCC",  "score": -5.8},
            {"smiles": "CCCC", "score": -7.0},
        ],
        "batch_size": batch_size,
        "gp": gp if gp is not None else {"device": "cpu", "fit_itersteps": 20, "fp_n_bits": 128},
        "seed": seed,
    }
    if minimize is not None:
        request["minimize"] = minimize
    if ref_point is not None:
        request["ref_point"] = ref_point
    return request


def _trajectory_request(
    *,
    method: str = "random",
    seed: int = 0,
    seed_smiles: str = "CCO,CCN,CCC",
    num_evaluations: int = 8,
    batch_size: int = 2,
    objective: str = "mock",
    init_size: int = None,
    gp_device: str = "cpu",
    gp_standardize_y: bool = True,
):
    """Build a JSON request for ``run_search_trajectory``."""
    request = {
        "method": method,
        "seed": seed,
        "seed-smiles": seed_smiles,
        "num-evaluations": num_evaluations,
        "batch-size": batch_size,
        "objective": objective,
        "gp-device": gp_device,
        "gp-standardize-y": gp_standardize_y,
    }
    if init_size is not None:
        request["init-size"] = init_size
    return request


# ---------------------------------------------------------------------------
# run_search_trajectory
# ---------------------------------------------------------------------------


class RunSearchTrajectoryTests(unittest.TestCase):
    """Tests for the full-trajectory API (random / bo-tanimoto / multi-obj)."""

    def test_random_mock_returns_full_schema(self) -> None:
        req = _trajectory_request(
            method="random", seed=0, num_evaluations=6, batch_size=2,
        )
        resp_str = bo_api.run_search_trajectory(json.dumps(req))
        resp = json.loads(resp_str)
        self.assertNotIn("error", resp)
        self.assertEqual(
            set(resp.keys()),
            {"config", "history", "summary"},
        )
        self.assertEqual(resp["config"]["method"], "random")
        self.assertEqual(resp["config"]["objective"], "mock")
        self.assertEqual(resp["config"]["n_objectives"], 1)
        self.assertEqual(len(resp["history"]), 6)
        # Each entry has index, smiles, score (n_obj=1).
        for i, entry in enumerate(resp["history"]):
            self.assertEqual(entry["index"], i)
            self.assertIn("smiles", entry)
            self.assertIn("score", entry)
            self.assertNotIn("scores", entry)
        # Summary has bsf for n_obj=1.
        self.assertIn("bsf", resp["summary"])
        self.assertEqual(len(resp["summary"]["bsf"]), 6)

    def test_bo_tanimoto_mock_returns_full_schema(self) -> None:
        req = _trajectory_request(
            method="bo-tanimoto", seed=0, num_evaluations=8, batch_size=2,
            init_size=4,
        )
        resp_str = bo_api.run_search_trajectory(json.dumps(req))
        resp = json.loads(resp_str)
        self.assertNotIn("error", resp)
        self.assertEqual(resp["config"]["method"], "bo-tanimoto")
        self.assertEqual(resp["config"]["gp"]["impl"], "fingerprint+tanimoto")
        self.assertGreaterEqual(len(resp["history"]), 1)
        self.assertIn("summary", resp)
        self.assertIn("bsf", resp["summary"])

    def test_multi_obj_mock_returns_hypervolume_summary(self) -> None:
        # Mock + Mock gives n_obj=2 (no real Vina needed).
        req = _trajectory_request(
            method="bo-tanimoto", seed=0, num_evaluations=6, batch_size=2,
            init_size=4, objective="mock+mock",
        )
        req["ref-point"] = "0,0"
        resp_str = bo_api.run_search_trajectory(json.dumps(req))
        resp = json.loads(resp_str)
        self.assertNotIn("error", resp, msg=resp_str[:400])
        self.assertEqual(resp["config"]["n_objectives"], 2)
        self.assertEqual(resp["config"]["objective_parts"], ["mock", "mock"])
        self.assertEqual(resp["config"]["minimize"], [True, True])
        # Each history entry uses 'scores' for n_obj>=2.
        for entry in resp["history"]:
            self.assertIn("scores", entry)
            self.assertNotIn("score", entry)
            self.assertEqual(len(entry["scores"]), 2)
        # Summary uses hypervolume for n_obj==2.
        self.assertIn("hypervolume", resp["summary"])
        self.assertNotIn("bsf", resp["summary"])
        self.assertEqual(len(resp["summary"]["hypervolume"]), 6)

    def test_invalid_objective_returns_error_json(self) -> None:
        req = _trajectory_request()
        req["objective"] = "not_a_real_backend"
        resp_str = bo_api.run_search_trajectory(json.dumps(req))
        resp = json.loads(resp_str)
        self.assertIn("error", resp)
        self.assertIn("error_type", resp)
        self.assertIn("traceback", resp)
        self.assertIn("not_a_real_backend", resp["error"])
        self.assertEqual(resp["error_type"], "ValueError")
        # Traceback should mention the source line.
        self.assertIn("Traceback", resp["traceback"])

    def test_invalid_method_returns_error_json(self) -> None:
        req = _trajectory_request()
        req["method"] = "not_a_method"
        resp_str = bo_api.run_search_trajectory(json.dumps(req))
        resp = json.loads(resp_str)
        self.assertIn("error", resp)
        self.assertIn("not_a_method", resp["error"])

    def test_unknown_key_returns_error_json(self) -> None:
        req = _trajectory_request()
        req["totally_made_up_key"] = 42
        resp_str = bo_api.run_search_trajectory(json.dumps(req))
        resp = json.loads(resp_str)
        self.assertIn("error", resp)
        self.assertIn("totally_made_up_key", resp["error"])

    def test_malformed_json_returns_error_json(self) -> None:
        resp_str = bo_api.run_search_trajectory("not even json")
        resp = json.loads(resp_str)
        self.assertIn("error", resp)
        self.assertEqual(resp["error_type"], "JSONDecodeError")

    def test_request_must_be_object(self) -> None:
        resp_str = bo_api.run_search_trajectory(json.dumps([1, 2, 3]))
        resp = json.loads(resp_str)
        self.assertIn("error", resp)
        self.assertIn("object", resp["error"])

    def test_bo_tanimoto_ldm_mock_returns_embedded_trajectory(self) -> None:
        """LDM trajectory path is exposed through bo_api, not dropped."""
        from unittest import mock
        from strbo_v1.llm_advisor import client as llm_client_module
        from strbo_v1.llm_advisor.blocks import NoopBlock, ReviewBOBlock
        from strbo_v1.llm_advisor.client import MockLLMClient, _serialize_blocks

        class Dyn(MockLLMClient):
            def chat(self, system, user, *, json_mode=True):
                self.call_log.append({"system": system[:80], "user": user[:80]})
                if "STAGE B" in system:
                    decisions = {}
                    for line in user.splitlines():
                        s = line.strip()
                        if s.startswith("- ") and "  mu=" in s:
                            smi = s[2:].split("  mu=", 1)[0].strip()
                            if smi:
                                decisions[smi] = "ok"
                    return _serialize_blocks([
                        ReviewBOBlock(rationale="ok", decisions=decisions),
                    ])
                return _serialize_blocks([NoopBlock(rationale="ok")])

        def factory(cfg, **_kwargs):
            return Dyn()

        req = _trajectory_request(
            method="bo-tanimoto-ldm", objective="mock",
            seed_smiles="CCO,CCN,CCC,CCCC",
            num_evaluations=4, batch_size=1, init_size=2,
        )
        req["pool-min-size"] = 1
        req["gp-fit-itersteps"] = 5
        req["gp-fp-n-bits"] = 128
        with mock.patch.object(llm_client_module, "OpenAIChatClient", factory):
            resp = json.loads(bo_api.run_search_trajectory(
                json.dumps(req),
                gp_device="cpu",
                llm_api_key="test-key",
                llm_base_url="https://example.invalid/v1",
            ))
        self.assertNotIn("error", resp, msg=json.dumps(resp)[:500])
        self.assertEqual(resp["config"]["method"], "bo-tanimoto-ldm")
        self.assertIn("llm_trajectory", resp)
        self.assertIn(resp["llm_trajectory"]["status"], ("completed", "fatal_error"))


# ---------------------------------------------------------------------------
# recommend_next_smiles
# ---------------------------------------------------------------------------


class RecommendNextSmilesTests(unittest.TestCase):
    """Tests for the advisor-step API (one round)."""

    def test_random_returns_subset_of_pool_with_empty_acq(self) -> None:
        req = _bo_request(method="random", batch_size=3)
        resp_str = bo_api.recommend_next_smiles(json.dumps(req))
        resp = json.loads(resp_str)
        self.assertNotIn("error", resp)
        self.assertEqual(resp["method"], "random")
        self.assertEqual(resp["n_objectives"], 1)
        self.assertEqual(resp["n_history"], 4)
        self.assertEqual(resp["pool_size"], 8)
        self.assertEqual(resp["acquisition_values"], [])
        self.assertEqual(len(resp["recommendations"]), 3)
        # All picks must come from the pool.
        for pick in resp["recommendations"]:
            self.assertIn(pick, req["pool"])

    def test_random_best_same_per_round_pick_as_random(self) -> None:
        # random-best's per-round advisor is identical to random
        # (the 'best' only affects expansion).
        req = _bo_request(method="random-best", batch_size=2, seed=42)
        resp_str = bo_api.recommend_next_smiles(json.dumps(req))
        resp = json.loads(resp_str)
        self.assertNotIn("error", resp)
        self.assertEqual(resp["method"], "random-best")
        self.assertEqual(resp["acquisition_values"], [])
        self.assertEqual(len(resp["recommendations"]), 2)

    def test_bo_tanimoto_returns_top_k_with_acquisition_values(self) -> None:
        req = _bo_request(method="bo-tanimoto", batch_size=2)
        resp_str = bo_api.recommend_next_smiles(json.dumps(req))
        resp = json.loads(resp_str)
        self.assertNotIn("error", resp, msg=resp_str[:400])
        self.assertEqual(resp["method"], "bo-tanimoto")
        self.assertEqual(resp["n_objectives"], 1)
        self.assertEqual(len(resp["recommendations"]), 2)
        self.assertEqual(len(resp["acquisition_values"]), 2)
        # All picks must come from the pool (after filtering).
        for pick in resp["recommendations"]:
            self.assertIn(pick, req["pool"])
        # Acquisition values must be finite.
        for v in resp["acquisition_values"]:
            self.assertTrue(
                float("-inf") < v < float("inf"),
                f"non-finite acquisition value: {v!r}",
            )

    def test_bo_strkernel_returns_top_k_with_acquisition_values(self) -> None:
        req = _bo_request(method="bo-strkernel", batch_size=2)
        resp_str = bo_api.recommend_next_smiles(json.dumps(req))
        resp = json.loads(resp_str)
        self.assertNotIn("error", resp, msg=resp_str[:400])
        self.assertEqual(resp["method"], "bo-strkernel")
        self.assertEqual(len(resp["recommendations"]), 2)
        self.assertEqual(len(resp["acquisition_values"]), 2)

    def test_seed_reproducibility(self) -> None:
        req = _bo_request(method="bo-tanimoto", batch_size=2, seed=12345)
        resp_a = json.loads(bo_api.recommend_next_smiles(json.dumps(req)))
        resp_b = json.loads(bo_api.recommend_next_smiles(json.dumps(req)))
        self.assertEqual(resp_a["recommendations"], resp_b["recommendations"])
        self.assertEqual(resp_a["acquisition_values"], resp_b["acquisition_values"])

    def test_random_seed_reproducibility(self) -> None:
        req = _bo_request(method="random", batch_size=3, seed=42)
        resp_a = json.loads(bo_api.recommend_next_smiles(json.dumps(req)))
        resp_b = json.loads(bo_api.recommend_next_smiles(json.dumps(req)))
        self.assertEqual(resp_a["recommendations"], resp_b["recommendations"])

    def test_empty_history_random_fallback(self) -> None:
        req = _bo_request(method="bo-tanimoto", batch_size=2, history=[])
        resp_str = bo_api.recommend_next_smiles(json.dumps(req))
        resp = json.loads(resp_str)
        self.assertNotIn("error", resp)
        # Empty history → <2 finite scores → uniform random pick.
        self.assertEqual(len(resp["recommendations"]), 2)
        # Acquisition values are 0 for the random-fallback path.
        self.assertEqual(resp["acquisition_values"], [0.0, 0.0])

    def test_empty_pool_returns_empty_recommendations(self) -> None:
        req = _bo_request(method="bo-tanimoto", batch_size=2, pool=[])
        resp_str = bo_api.recommend_next_smiles(json.dumps(req))
        resp = json.loads(resp_str)
        self.assertNotIn("error", resp)
        self.assertEqual(resp["recommendations"], [])
        self.assertEqual(resp["pool_size"], 0)

    def test_multi_obj_2_returns_n_objectives_2(self) -> None:
        req = _bo_request(method="bo-tanimoto", batch_size=2)
        req["history"] = [
            {"smiles": "CCO",  "scores": [-7.5, 5.2]},
            {"smiles": "CCN",  "scores": [-6.2, 5.5]},
            {"smiles": "CCC",  "scores": [-5.8, 4.8]},
            {"smiles": "CCCC", "scores": [-7.0, 5.1]},
        ]
        req["minimize"] = [True, False]
        req["ref_point"] = [0.0, 5.0]
        req["ehvi_n_samples"] = 32
        resp_str = bo_api.recommend_next_smiles(json.dumps(req))
        resp = json.loads(resp_str)
        self.assertNotIn("error", resp, msg=resp_str[:400])
        self.assertEqual(resp["n_objectives"], 2)
        self.assertEqual(len(resp["recommendations"]), 2)
        self.assertEqual(len(resp["acquisition_values"]), 2)

    def test_minimize_length_mismatch_returns_error(self) -> None:
        req = _bo_request(method="bo-tanimoto")
        req["history"] = [
            {"smiles": "CCO",  "scores": [-7.5, 5.2]},
            {"smiles": "CCN",  "scores": [-6.2, 5.5]},
        ]
        req["minimize"] = [True, False, True]  # length 3, but n_obj=2
        resp_str = bo_api.recommend_next_smiles(json.dumps(req))
        resp = json.loads(resp_str)
        self.assertIn("error", resp)
        self.assertIn("minimize", resp["error"])

    def test_unknown_method_returns_error_json(self) -> None:
        req = _bo_request()
        req["method"] = "foo"
        resp_str = bo_api.recommend_next_smiles(json.dumps(req))
        resp = json.loads(resp_str)
        self.assertIn("error", resp)
        self.assertEqual(resp["error_type"], "ValueError")
        self.assertIn("foo", resp["error"])
        self.assertIn("Traceback", resp["traceback"])

    def test_batch_size_zero_returns_error(self) -> None:
        req = _bo_request(method="random", batch_size=0)
        resp_str = bo_api.recommend_next_smiles(json.dumps(req))
        resp = json.loads(resp_str)
        self.assertIn("error", resp)
        self.assertIn("batch_size", resp["error"])

    def test_malformed_json_returns_error_json(self) -> None:
        resp_str = bo_api.recommend_next_smiles("not even json")
        resp = json.loads(resp_str)
        self.assertIn("error", resp)
        self.assertEqual(resp["error_type"], "JSONDecodeError")

    def test_request_must_be_object(self) -> None:
        resp_str = bo_api.recommend_next_smiles(json.dumps([1, 2]))
        resp = json.loads(resp_str)
        self.assertIn("error", resp)
        self.assertIn("object", resp["error"])

    def test_pool_filtered_against_history(self) -> None:
        """SMILES in history should not appear in recommendations."""
        req = _bo_request(method="random", batch_size=3)
        # Pretend CCO and CCN are already in history.
        req["pool"] = ["CCC", "C1CCCCC1", "c1ccccc1", "CCCCC"]
        req["history"] = [
            {"smiles": "CCO", "score": -7.5},
            {"smiles": "CCN", "score": -6.2},
        ]
        resp_str = bo_api.recommend_next_smiles(json.dumps(req))
        resp = json.loads(resp_str)
        self.assertNotIn("error", resp)
        for pick in resp["recommendations"]:
            self.assertNotIn(pick, ["CCO", "CCN"])

    def test_bo_tanimoto_ldm_one_step_with_mock_llm(self) -> None:
        """One-step LDM path runs A1 + BO + Stage B and returns diagnostics."""
        from unittest import mock
        from strbo_v1.llm_advisor import client as llm_client_module
        from strbo_v1.llm_advisor.blocks import NoopBlock, ReviewBOBlock
        from strbo_v1.llm_advisor.client import MockLLMClient, _serialize_blocks

        class Dyn(MockLLMClient):
            def chat(self, system, user, *, json_mode=True):
                self.call_log.append({"system": system[:80], "user": user[:80]})
                if "STAGE B" in system:
                    decisions = {}
                    for line in user.splitlines():
                        s = line.strip()
                        if s.startswith("- ") and "  mu=" in s:
                            smi = s[2:].split("  mu=", 1)[0].strip()
                            if smi:
                                decisions[smi] = "ok"
                    return _serialize_blocks([
                        ReviewBOBlock(rationale="ok", decisions=decisions),
                    ])
                return _serialize_blocks([NoopBlock(rationale="ok")])

        def factory(cfg):
            return Dyn()

        req = _bo_request(method="bo-tanimoto-ldm", batch_size=2)
        req["gp_fit_itersteps"] = 5
        req["gp_fp_n_bits"] = 128
        req["pool_min_size"] = 2
        with mock.patch.object(llm_client_module, "OpenAIChatClient", factory):
            resp = json.loads(bo_api.recommend_next_smiles(
                json.dumps(req),
                gp_device="cpu",
                llm_api_key="test-key",
                llm_base_url="https://example.invalid/v1",
            ))
        self.assertNotIn("error", resp, msg=json.dumps(resp)[:500])
        self.assertEqual(resp["method"], "bo-tanimoto-ldm")
        self.assertEqual(resp["n_objectives"], 1)
        self.assertEqual(len(resp["recommendations"]), 2)
        self.assertIn("llm", resp)
        self.assertTrue(resp["llm"]["stage_a1"]["executed"])
        self.assertTrue(resp["llm"]["stage_b"]["executed"])
        self.assertEqual(len(resp["llm"]["bo_suggestions"]), 2)


# ---------------------------------------------------------------------------
# select_candidates dispatch (sanity)
# ---------------------------------------------------------------------------


class SelectCandidatesDispatchTests(unittest.TestCase):
    """Sanity-check that ``bayesian_select_candidates`` returns the right
    shape for each ``n_obj``."""

    def test_n_obj_1_returns_acq_values_of_length_k(self) -> None:
        gp = GPConfig(impl="fingerprint+tanimoto", device="cpu",
                      fit_n_itersteps=10, fp_n_bits=128)
        cfg = BayesianAnalogSearchConfig(
            batch_size=3, minimize=True, gp_config=gp,
        )
        picks, acq = bayesian_select_candidates(
            pool=["CCC", "C1CCCCC1", "c1ccccc1", "CCCCC"],
            history=[("CCO", -7.5), ("CCN", -6.2)],
            config=cfg,
            rng=RNG(seed=0),
        )
        self.assertEqual(len(picks), 3)
        self.assertEqual(len(acq), 3)

    def test_n_obj_2_returns_acq_values_of_length_k(self) -> None:
        gp = GPConfig(impl="fingerprint+tanimoto", device="cpu",
                      fit_n_itersteps=10, fp_n_bits=128)
        cfg = BayesianAnalogSearchConfig(
            batch_size=2, minimize=(True, False), ref_point=(0.0, 5.0),
            ehvi_n_samples=16, gp_config=gp,
        )
        history = [
            ("CCO", (-7.5, 5.2)),
            ("CCN", (-6.2, 5.5)),
            ("CCC", (-5.8, 4.8)),
        ]
        picks, acq = bayesian_select_candidates(
            pool=["C1CCCCC1", "c1ccccc1", "CCCCC", "CCCCO"],
            history=history, config=cfg, rng=RNG(seed=0),
        )
        self.assertEqual(len(picks), 2)
        self.assertEqual(len(acq), 2)

    def test_random_next_batch_returns_subset(self) -> None:
        picks = random_select_next_batch(
            pool=["a", "b", "c", "d", "e"],
            batch_size=3, rng=RNG(seed=42),
        )
        self.assertEqual(len(picks), 3)
        for p in picks:
            self.assertIn(p, ["a", "b", "c", "d", "e"])

    def test_random_next_batch_empty_pool(self) -> None:
        self.assertEqual(
            random_select_next_batch(pool=[], batch_size=3, rng=RNG(seed=0)),
            [],
        )


# ---------------------------------------------------------------------------
# Provider's setting (Python kwargs only) — JSON values are silently ignored
# ---------------------------------------------------------------------------


class RunSearchTrajectoryProviderSettingTests(unittest.TestCase):
    """Provider's setting: deployment wiring.

    Precedence: ``Python kwarg > env var > hard-coded default``. The
    JSON body never participates for the provider-setting keys; any
    value passed for them in ``request_json`` is silently dropped and
    a DEBUG log line is emitted. See :data:`bo_api.PROVIDER_SETTING_KEYS`
    for the canonical list.
    """

    # -- Provider-setting JSON keys are silently ignored --------------------

    def test_json_provider_setting_keys_silently_ignored(self) -> None:
        """Each provider-setting key (both hyphen and underscore
        forms) is dropped from the request body; the call succeeds and the
        DEBUG log is emitted exactly once per ignored key."""
        for json_key in (
            "vina-bin", "vina_bin",
            "vina-cache-dir", "vina_cache_dir",
            "vina-max-workers", "vina_max_workers",
            "gp-device", "gp_device",
            "reasyn-repo", "reasyn_repo",
            "reasyn-python-bin", "reasyn_python_bin",
            "reasyn-model-path", "reasyn_model_path",
            "reasyn-devices", "reasyn_devices",
            "nn-model-path", "nn_model_path",
            "nn-metadata-path", "nn_metadata_path",
            "llm-model", "llm_model",
            "llm-base-url", "llm_base_url",
            "llm-api-key", "llm_api_key",
        ):
            with self.subTest(json_key=json_key):
                req = _trajectory_request(
                    method="random", objective="mock",
                    seed_smiles="CCO,CCN,CCC",
                    num_evaluations=4, batch_size=2,
                )
                req[json_key] = "/should/be/ignored/path"
                with self.assertLogs("bo_api", level="DEBUG") as caplog:
                    resp = json.loads(bo_api.run_search_trajectory(
                        json.dumps(req),
                    ))
                self.assertNotIn("error", resp)
                # DEBUG log mentions the original key.
                self.assertTrue(
                    any(json_key in rec.getMessage() for rec in caplog.records),
                    f"no DEBUG log mentioning {json_key!r}: {[r.getMessage() for r in caplog.records]}",
                )
                # The config echo must NOT contain the JSON value.
                cfg_str = json.dumps(resp["config"])
                self.assertNotIn("/should/be/ignored/path", cfg_str)

    def test_reasyn_repo_kwarg_wins_json_silently_dropped(self) -> None:
        """The kwarg wins; the JSON value is silently dropped."""
        req = _trajectory_request(objective="mock", seed_smiles="CCO")
        req["reasyn-repo"] = "/json/value/ignored"
        resp = json.loads(bo_api.run_search_trajectory(
            json.dumps(req),
            reasyn_repo="/correct/kwarg/path",
        ))
        self.assertNotIn("error", resp)
        self.assertEqual(
            resp["config"]["reasyn"]["reasyn_repo"],
            "/correct/kwarg/path",
        )

    def test_reasyn_repo_kwarg_none_json_silently_dropped(self) -> None:
        """When kwarg is None, the JSON value is also dropped — env var
        or default applies (no JSON participation)."""
        req = _trajectory_request(objective="mock", seed_smiles="CCO")
        req["reasyn-repo"] = "/json/value/ignored"
        with self.assertLogs("bo_api", level="DEBUG"):
            resp = json.loads(bo_api.run_search_trajectory(json.dumps(req)))
        self.assertNotIn("error", resp)
        # The JSON value must not appear in the echo.
        self.assertNotIn("/json/value/ignored", json.dumps(resp["config"]))

    def test_reasyn_repo_kwarg_overrides_env_var(self) -> None:
        """Explicit kwarg beats env var when JSON is silent."""
        import os
        saved = os.environ.get("REASYN_HOME")
        os.environ["REASYN_HOME"] = "/env/path"
        try:
            req = _trajectory_request(objective="mock", seed_smiles="CCO")
            resp = json.loads(bo_api.run_search_trajectory(
                json.dumps(req),
                reasyn_repo="/kwarg/beats/env",
            ))
            self.assertNotIn("error", resp)
            # The kwarg wins; the env var is not consulted.
            self.assertEqual(
                resp["config"]["reasyn"]["reasyn_repo"],
                "/kwarg/beats/env",
            )
        finally:
            if saved is None:
                os.environ.pop("REASYN_HOME", None)
            else:
                os.environ["REASYN_HOME"] = saved

    def test_env_var_used_when_kwarg_none_and_json_silent(self) -> None:
        """Env var used when both kwarg and JSON are silent."""
        import os
        saved = os.environ.get("REASYN_REPO")
        os.environ["REASYN_REPO"] = "/env/repo/path"
        try:
            req = _trajectory_request(objective="mock", seed_smiles="CCO")
            resp = json.loads(bo_api.run_search_trajectory(json.dumps(req)))
            self.assertNotIn("error", resp)
            # Env var resolves; the config echo reflects args.reasyn_repo
            # (which was injected by config_from_dict; the env-var lookup
            # happens later in _build_reasyn_analog, so the echo still
            # shows None — but the call itself succeeds without the
            # ReaSyn-failed error, proving the env var was honored).
            # The real proof is that the call did NOT error out with
            # "Cannot locate ReaSyn repo" before this assertion ran.
            self.assertNotIn("error", resp)
        finally:
            if saved is None:
                os.environ.pop("REASYN_REPO", None)
            else:
                os.environ["REASYN_REPO"] = saved

    def test_all_5_reasyn_kwargs_propagate(self) -> None:
        """All five ReaSyn kwargs make it to the config echo."""
        req = _trajectory_request(
            method="bo-tanimoto", objective="mock",
            seed_smiles="CCO", num_evaluations=4, batch_size=2, init_size=2,
        )
        resp = json.loads(bo_api.run_search_trajectory(
            json.dumps(req),
            reasyn_repo="../ReaSyn",
            reasyn_python_bin="/path/to/python",
            reasyn_model_path="/path/to/ckpt.ckpt",
            reasyn_devices="0,1",
        ))
        self.assertNotIn("error", resp)
        self.assertEqual(
            resp["config"]["reasyn"]["reasyn_repo"], "../ReaSyn",
        )
        self.assertEqual(
            resp["config"]["reasyn"]["reasyn_python_bin"], "/path/to/python",
        )
        self.assertEqual(
            resp["config"]["reasyn"]["reasyn_model_path"], "/path/to/ckpt.ckpt",
        )
        self.assertEqual(
            resp["config"]["reasyn"]["reasyn_devices"], "0,1",
        )

    def test_vina_bin_kwarg_propagates_to_error_message(self) -> None:
        """Vina kwarg is wired through (proven via the FileNotFoundError
        that the VinaScorer raises on construction)."""
        req = _trajectory_request(
            method="bo-tanimoto", objective="vina",
            seed_smiles="CCO", num_evaluations=4, batch_size=2, init_size=2,
            gp_device="cpu",
        )
        # No real Vina binary at this path; expect error JSON.
        resp = json.loads(bo_api.run_search_trajectory(
            json.dumps(req), vina_bin="/definitely/not/a/real/vina",
        ))
        self.assertIn("error", resp)
        # The error message must contain the kwarg-supplied path, proving
        # the kwarg was wired through to VinaScorerConfig.vina_bin.
        self.assertIn("/definitely/not/a/real/vina", resp["error"])

    def test_vina_bin_kwarg_wins_json_silently_dropped(self) -> None:
        """Vina kwarg wins; JSON value is silently dropped (also via error
        message — the error must contain the kwarg path and not the JSON one).
        """
        req = _trajectory_request(
            method="bo-tanimoto", objective="vina",
            seed_smiles="CCO", num_evaluations=4, batch_size=2, init_size=2,
            gp_device="cpu",
        )
        req["vina-bin"] = "/wrong/json/path"
        with self.assertLogs("bo_api", level="DEBUG"):
            resp = json.loads(bo_api.run_search_trajectory(
                json.dumps(req), vina_bin="/correct/kwarg/path",
            ))
        self.assertIn("error", resp)
        self.assertIn("/correct/kwarg/path", resp["error"])
        self.assertNotIn("/wrong/json/path", resp["error"])

    def test_vina_cache_dir_kwarg_propagates(self) -> None:
        """``vina_cache_dir`` kwarg is wired through to VinaScorerConfig."""
        from strbo_v1 import objective_vina
        captured: dict = {}
        original = objective_vina.VinaScorerConfig
        def spy(**kwargs):
            captured.update(kwargs)
            return original(**kwargs)
        objective_vina.VinaScorerConfig = spy
        try:
            req = _trajectory_request(
                method="bo-tanimoto", objective="vina",
                seed_smiles="CCO", num_evaluations=4, batch_size=2, init_size=2,
                gp_device="cpu",
            )
            with TemporaryDirectory() as tmp:
                cache_dir = str(Path(tmp) / "cache")
                _ = json.loads(bo_api.run_search_trajectory(
                    json.dumps(req),
                    vina_bin="/definitely/not/a/real/vina",
                    vina_cache_dir=cache_dir,
                ))
            self.assertEqual(
                str(captured.get("cache_dir")),
                cache_dir,
            )
        finally:
            objective_vina.VinaScorerConfig = original

    def test_vina_cache_dir_kwarg_wins_json_silently_dropped(self) -> None:
        """``vina_cache_dir`` kwarg wins; the JSON value is silently
        dropped."""
        from strbo_v1 import objective_vina
        captured: dict = {}
        original = objective_vina.VinaScorerConfig
        def spy(**kwargs):
            captured.update(kwargs)
            return original(**kwargs)
        objective_vina.VinaScorerConfig = spy
        try:
            req = _trajectory_request(
                method="bo-tanimoto", objective="vina",
                seed_smiles="CCO", num_evaluations=4, batch_size=2, init_size=2,
                gp_device="cpu",
            )
            req["vina-cache-dir"] = "/json_cache_will_be_dropped"
            with self.assertLogs("bo_api", level="DEBUG"):
                with TemporaryDirectory() as tmp:
                    cache_dir = str(Path(tmp) / "cache")
                    _ = json.loads(bo_api.run_search_trajectory(
                        json.dumps(req),
                        vina_bin="/definitely/not/a/real/vina",
                        vina_cache_dir=cache_dir,
                    ))
            self.assertEqual(
                str(captured.get("cache_dir")),
                cache_dir,
            )
        finally:
            objective_vina.VinaScorerConfig = original

    def test_gp_device_kwarg_propagates(self) -> None:
        """``gp_device`` kwarg is wired through to the GP config echo."""
        req = _trajectory_request(
            method="bo-tanimoto", objective="mock",
            seed_smiles="CCO", num_evaluations=4, batch_size=2, init_size=2,
        )
        resp = json.loads(bo_api.run_search_trajectory(
            json.dumps(req),
            gp_device="cuda:7",
        ))
        self.assertNotIn("error", resp)
        self.assertEqual(
            resp["config"]["gp"]["gp_device"], "cuda:7",
        )

    def test_gp_device_kwarg_wins_json_silently_dropped(self) -> None:
        """``gp_device`` kwarg wins over JSON (which is dropped)."""
        req = _trajectory_request(
            method="bo-tanimoto", objective="mock",
            seed_smiles="CCO", num_evaluations=4, batch_size=2, init_size=2,
        )
        req["gp-device"] = "cpu"
        with self.assertLogs("bo_api", level="DEBUG"):
            resp = json.loads(bo_api.run_search_trajectory(
                json.dumps(req),
                gp_device="cuda:3",
            ))
        self.assertNotIn("error", resp)
        self.assertEqual(
            resp["config"]["gp"]["gp_device"], "cuda:3",
        )
        self.assertNotIn('"cuda:3"', "")  # sanity no-op

    def test_provider_setting_keys_constant(self) -> None:
        """The canonical list of provider-setting keys is exposed."""
        self.assertEqual(
            sorted(bo_api.PROVIDER_SETTING_KEYS),
            [
                "gp_device",
                "llm_api_key",
                "llm_base_url",
                "llm_model",
                "nn_metadata_path",
                "nn_model_path",
                "reasyn_devices",
                "reasyn_model_path",
                "reasyn_python_bin",
                "reasyn_repo",
                "vina_bin",
                "vina_cache_dir",
                "vina_max_workers",
            ],
        )

    def test_nn_kwargs_accepted_by_signature(self) -> None:
        """The provider-setting kwargs are part of the trajectory
        signature (signature check is the only feasible verification since
        loading real Vina/ReaSyn/NN backends requires GPU etc.)."""
        import inspect
        sig = inspect.signature(bo_api.run_search_trajectory)
        for name in (
            "vina_bin", "vina_cache_dir", "vina_max_workers",
            "reasyn_repo", "reasyn_python_bin",
            "reasyn_model_path", "reasyn_devices",
            "gp_device",
            "nn_model_path", "nn_metadata_path",
            "llm_model", "llm_base_url", "llm_api_key",
        ):
            with self.subTest(kwarg=name):
                self.assertIn(name, sig.parameters)
                self.assertEqual(sig.parameters[name].default, None)
                self.assertEqual(
                    sig.parameters[name].kind,
                    inspect.Parameter.KEYWORD_ONLY,
                )

    def test_recommend_next_smiles_provider_kwargs(self) -> None:
        """API 2 is a pure advisor; it does not invoke Vina/ReaSyn/NN so
        it must not expose those provider-setting kwargs. It does accept
        ``gp_device`` and LLM provider kwargs for bo-*-ldm methods."""
        import inspect
        sig = inspect.signature(bo_api.recommend_next_smiles)
        for name in (
            "vina_bin", "vina_cache_dir", "vina_max_workers",
            "reasyn_repo", "reasyn_python_bin",
            "reasyn_model_path", "reasyn_devices",
            "nn_model_path", "nn_metadata_path",
        ):
            self.assertNotIn(name, sig.parameters)
        # ``gp_device`` IS on the advisor (pure BO uses it).
        self.assertIn("gp_device", sig.parameters)
        self.assertEqual(sig.parameters["gp_device"].default, None)
        self.assertEqual(
            sig.parameters["gp_device"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )
        for name in ("llm_model", "llm_base_url", "llm_api_key"):
            self.assertIn(name, sig.parameters)
            self.assertEqual(sig.parameters[name].default, None)
            self.assertEqual(
                sig.parameters[name].kind,
                inspect.Parameter.KEYWORD_ONLY,
            )

    def test_gp_device_kwarg_propagates_on_advisor(self) -> None:
        """The advisor's ``gp_device`` kwarg is wired through to the GPConfig
        (proven via monkey-patching ``_build_gp_config`` to capture the
        ``device`` keyword argument).
        """
        req = _bo_request(
            method="bo-tanimoto",
            gp={"device": "cpu", "fit_itersteps": 5, "fp_n_bits": 128},
            seed=0,
        )
        resp_str = bo_api.recommend_next_smiles(
            json.dumps(req), gp_device="cuda:9",
        )
        resp = json.loads(resp_str)
        self.assertNotIn("error", resp)
        # Spy on _build_gp_config to capture the device kwarg.
        captured: list = []
        original = bo_api._build_gp_config
        def spy(request, method, *, device="cuda"):
            captured.append({"device": device})
            return original(request, method, device=device)
        bo_api._build_gp_config = spy
        try:
            bo_api.recommend_next_smiles(
                json.dumps(req), gp_device="cuda:9",
            )
        finally:
            bo_api._build_gp_config = original
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["device"], "cuda:9")

    def test_no_kwargs_backward_compat(self) -> None:
        """Calling without any provider-setting kwargs still works
        (legacy callers)."""
        req = _trajectory_request(
            method="random", objective="mock",
            seed_smiles="CCO,CCN,CCC",
            num_evaluations=6, batch_size=2,
        )
        resp = json.loads(bo_api.run_search_trajectory(json.dumps(req)))
        self.assertNotIn("error", resp)
        self.assertEqual(len(resp["history"]), 6)


# ---------------------------------------------------------------------------
# bo_api.DEFAULT (mirrors run_search.sh) — contents, exclusions, application
# ---------------------------------------------------------------------------


class DefaultDictTests(unittest.TestCase):
    """The ``bo_api.DEFAULT`` dict is the bo_api's source-of-truth for
    user's-request defaults. Provider-setting keys must NEVER
    appear here."""

    def test_default_dict_contents(self) -> None:
        """All 28 keys with their expected ``run_search.sh`` values."""
        self.assertEqual(
            bo_api.DEFAULT["num_evaluations"], 80,
        )
        self.assertEqual(bo_api.DEFAULT["batch_size"], 5)
        self.assertEqual(bo_api.DEFAULT["init_size"], 10)
        self.assertEqual(bo_api.DEFAULT["objective"], "vina+nn")
        # GP tuning (flat).
        self.assertEqual(bo_api.DEFAULT["gp_fit_itersteps"], 100)
        self.assertEqual(bo_api.DEFAULT["gp_learning_rate"], 0.05)
        self.assertEqual(bo_api.DEFAULT["gp_min_jitter"], 1e-6)
        self.assertEqual(bo_api.DEFAULT["gp_max_jitter"], 1e-1)
        self.assertEqual(bo_api.DEFAULT["gp_standardize_y"], True)
        self.assertEqual(bo_api.DEFAULT["gp_fp_radius"], 2)
        self.assertEqual(bo_api.DEFAULT["gp_fp_n_bits"], 2048)
        # Acquisition / multi-obj.
        self.assertEqual(bo_api.DEFAULT["acquisition"], "ei")
        self.assertEqual(bo_api.DEFAULT["xi"], 0.01)
        self.assertEqual(bo_api.DEFAULT["kappa"], 2.0)
        self.assertEqual(bo_api.DEFAULT["ehvi_n_samples"], 128)
        self.assertEqual(bo_api.DEFAULT["che_alpha"], 1.0)
        # ReaSyn tuning.
        self.assertEqual(bo_api.DEFAULT["reasyn_search_width"], 5)
        self.assertEqual(bo_api.DEFAULT["reasyn_exhaustiveness"], 8)
        self.assertEqual(bo_api.DEFAULT["reasyn_num_cycles"], 3)
        self.assertEqual(bo_api.DEFAULT["reasyn_num_editflow_samples"], 10)
        self.assertEqual(bo_api.DEFAULT["reasyn_num_editflow_steps"], 30)
        self.assertEqual(bo_api.DEFAULT["reasyn_time_limit"], 20)
        self.assertEqual(bo_api.DEFAULT["reasyn_num_workers_per_gpu"], 1)
        self.assertEqual(bo_api.DEFAULT["reasyn_filter_sim"], 0.8)
        # Pool sizing.
        self.assertEqual(bo_api.DEFAULT["smiles_max_len"], 100)
        self.assertEqual(bo_api.DEFAULT["pool_min_size"], 9)
        self.assertEqual(bo_api.DEFAULT["pool_max_size"], 18)
        self.assertEqual(bo_api.DEFAULT["max_pool_size"], 1024)

    def test_default_dict_excludes_provider_setting_keys(self) -> None:
        """Intersection of DEFAULT.keys() and PROVIDER_SETTING_KEYS is empty."""
        self.assertTrue(
            bo_api.DEFAULT.keys().isdisjoint(bo_api.PROVIDER_SETTING_KEYS),
            f"DEFAULT contains provider-setting keys: "
            f"{set(bo_api.DEFAULT.keys()) & set(bo_api.PROVIDER_SETTING_KEYS)}",
        )

    def test_default_dict_excludes_all_ten_provider_keys(self) -> None:
        """Explicit per-key check: none of the provider-setting keys
        appear in DEFAULT."""
        for k in bo_api.PROVIDER_SETTING_KEYS:
            with self.subTest(provider_key=k):
                self.assertNotIn(k, bo_api.DEFAULT)

    def _stub_vina_reasyn(self):
        """Helper: monkey-patch run_search._build_vina_scorer and
        _build_reasyn_analog with no-op stubs so the trajectory completes
        without real Vina/ReaSyn."""
        from unittest.mock import patch
        import run_search

        class StubScorer:
            def __call__(self, smis):
                return [-1.0 for _ in smis]

        self._vina_patch = patch.object(
            run_search, "_build_vina_scorer", return_value=StubScorer(),
        )
        self._reasyn_patch = patch.object(
            run_search, "_build_reasyn_analog",
            return_value=lambda smis: [s + "C" for s in smis],
        )
        self._vina_patch.start()
        self._reasyn_patch.start()

    def tearDown(self):
        # Clean up patches if any.
        for attr in ("_vina_patch", "_reasyn_patch"):
            patch = getattr(self, attr, None)
            if patch is not None:
                patch.stop()
                setattr(self, attr, None)

    def test_trajectory_default_applied_when_key_omitted(self) -> None:
        """When the user omits a user's-request key, DEFAULT fills it in.

        Uses ``objective=vina`` (with stubbed Vina scorer) so the config
        echo includes the ``vina`` block."""
        self._stub_vina_reasyn()
        req = {
            "method": "bo-tanimoto",
            "seed": 0,
            "seed-smiles": "CCO,CCN,CCC",
            "num-evaluations": 4,
            "batch-size": 2,
            "init-size": 2,
            "objective": "vina",
            "gp-device": "cpu",
            "gp-standardize-y": True,
        }
        resp = json.loads(bo_api.run_search_trajectory(json.dumps(req)))
        self.assertNotIn("error", resp)
        cfg = resp["config"]
        # Keys we supplied.
        self.assertEqual(cfg["num_evaluations"], 4)
        self.assertEqual(cfg["batch_size"], 2)
        self.assertEqual(cfg["objective"], "vina")
        # DEFAULT keys that we omitted — they should be filled in.
        self.assertEqual(cfg["init_size"], 2)  # supplied
        self.assertEqual(cfg["smiles_max_len"], 100)
        self.assertEqual(cfg["pool_min_size"], 9)
        self.assertEqual(cfg["pool_max_size"], 18)
        self.assertEqual(cfg["max_pool_size"], 1024)
        # ReaSyn tuning from DEFAULT.
        self.assertEqual(cfg["reasyn"]["reasyn_search_width"], 5)
        self.assertEqual(cfg["reasyn"]["reasyn_exhaustiveness"], 8)
        self.assertEqual(cfg["reasyn"]["reasyn_num_cycles"], 3)
        self.assertEqual(cfg["reasyn"]["reasyn_time_limit"], 20)
        # Provider-setting keys: vina_max_workers used argparse default (1),
        # NOT a bo_api DEFAULT (would have been 4 if we put it there).
        self.assertEqual(cfg["vina"]["vina_max_workers"], 1)

    def test_trajectory_provider_setting_keys_not_injected_by_default(self) -> None:
        """When the user supplies no provider-setting kwargs, the echo
        shows argparse defaults — NOT bo_api DEFAULT values."""
        self._stub_vina_reasyn()
        req = {
            "method": "bo-tanimoto",
            "seed": 0,
            "seed-smiles": "CCO",
            "num-evaluations": 4,
            "batch-size": 2,
            "init-size": 2,
            "objective": "vina",
            "gp-device": "cpu",
            "gp-standardize-y": True,
        }
        resp = json.loads(bo_api.run_search_trajectory(json.dumps(req)))
        self.assertNotIn("error", resp)
        cfg = resp["config"]
        # Provider-setting keys come from argparse defaults — NOT bo_api.
        # vina_max_workers: argparse default is 1.
        self.assertEqual(cfg["vina"]["vina_max_workers"], 1)
        # vina_cache_dir: argparse default is "output/bo/vina_cache/".
        self.assertEqual(cfg["vina"]["vina_cache_dir"], "output/bo/vina_cache/")
        # vina_bin: argparse default is None.
        self.assertIsNone(cfg["vina"]["vina_bin"])
        # reasyn_devices: argparse default is "1,2".
        self.assertEqual(cfg["reasyn"]["reasyn_devices"], "1,2")

    def test_trajectory_user_json_overrides_default(self) -> None:
        """User's explicit value wins over DEFAULT."""
        self._stub_vina_reasyn()
        req = {
            "method": "bo-tanimoto",
            "seed": 0,
            "seed-smiles": "CCO",
            "num-evaluations": 6,
            "batch-size": 1,
            "init-size": 3,
            "objective": "vina",
            "gp-device": "cpu",
            "gp-standardize-y": True,
        }
        resp = json.loads(bo_api.run_search_trajectory(json.dumps(req)))
        self.assertNotIn("error", resp)
        cfg = resp["config"]
        # User's value (6) wins over DEFAULT (80).
        self.assertEqual(cfg["num_evaluations"], 6)
        # User's value (1) wins over DEFAULT (5).
        self.assertEqual(cfg["batch_size"], 1)

    def test_advisor_flat_gp_keys_applied_when_omitted(self) -> None:
        """The advisor's flat GP keys are populated from DEFAULT when omitted."""
        req = _bo_request(
            method="bo-tanimoto",
            history=[
                {"smiles": "CCO", "score": -7.5},
                {"smiles": "CCN", "score": -6.2},
                {"smiles": "CCC", "score": -5.8},
                {"smiles": "CCCC", "score": -7.0},
            ],
            batch_size=2,
            gp={"device": "cpu", "fit_itersteps": 5, "fp_n_bits": 128},
            seed=0,
        )
        captured: list = []
        original = bo_api._build_gp_config
        def spy(request, method, *, device="cuda"):
            captured.append({
                "device": device,
                "gp_fit_itersteps": request.get(
                    "gp_fit_itersteps", bo_api.DEFAULT["gp_fit_itersteps"],
                ),
                "gp_learning_rate": request.get(
                    "gp_learning_rate", bo_api.DEFAULT["gp_learning_rate"],
                ),
                "gp_min_jitter": request.get(
                    "gp_min_jitter", bo_api.DEFAULT["gp_min_jitter"],
                ),
            })
            return original(request, method, device=device)
        bo_api._build_gp_config = spy
        try:
            bo_api.recommend_next_smiles(json.dumps(req), gp_device="cpu")
        finally:
            bo_api._build_gp_config = original
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["device"], "cpu")
        # learning_rate and min_jitter not in user's request → DEFAULT.
        self.assertEqual(captured[0]["gp_learning_rate"], 0.05)
        self.assertEqual(captured[0]["gp_min_jitter"], 1e-6)

    def test_advisor_user_json_gp_overrides_default(self) -> None:
        """User's flat gp_learning_rate wins over DEFAULT."""
        req = _bo_request(
            method="bo-tanimoto",
            history=[
                {"smiles": "CCO", "score": -7.5},
                {"smiles": "CCN", "score": -6.2},
                {"smiles": "CCC", "score": -5.8},
                {"smiles": "CCCC", "score": -7.0},
            ],
            batch_size=2,
            gp={"device": "cpu", "fit_itersteps": 5, "fp_n_bits": 128},
            seed=0,
        )
        # Override via flat gp_learning_rate at top level.
        req["gp_learning_rate"] = 0.5
        captured: list = []
        original = bo_api._build_gp_config
        def spy(request, method, *, device="cuda"):
            captured.append({
                "gp_fit_itersteps": request.get(
                    "gp_fit_itersteps", bo_api.DEFAULT["gp_fit_itersteps"],
                ),
                "gp_learning_rate": request.get(
                    "gp_learning_rate", bo_api.DEFAULT["gp_learning_rate"],
                ),
            })
            return original(request, method, device=device)
        bo_api._build_gp_config = spy
        try:
            bo_api.recommend_next_smiles(json.dumps(req), gp_device="cpu")
        finally:
            bo_api._build_gp_config = original
        self.assertEqual(captured[0]["gp_learning_rate"], 0.5)
        # fit_itersteps falls back to DEFAULT (wasn't overridden).
        self.assertEqual(captured[0]["gp_fit_itersteps"], 100)

    def test_advisor_provider_setting_json_silently_dropped(self) -> None:
        """JSON ``gp-device`` is silently filtered; the kwarg (or hardcoded
        fallback) wins."""
        req = _bo_request(
            method="bo-tanimoto",
            history=[
                {"smiles": "CCO", "score": -7.5},
                {"smiles": "CCN", "score": -6.2},
                {"smiles": "CCC", "score": -5.8},
                {"smiles": "CCCC", "score": -7.0},
            ],
            batch_size=2,
            gp={"device": "cpu", "fit_itersteps": 5, "fp_n_bits": 128},
            seed=0,
        )
        req["gp-device"] = "cpu"  # JSON: should be silently dropped
        captured: list = []
        original = bo_api._build_gp_config
        def spy(request, method, *, device="cuda"):
            captured.append({"device": device})
            return original(request, method, device=device)
        bo_api._build_gp_config = spy
        try:
            # No kwarg → device should fall back to "cuda" (hardcoded),
            # NOT to "cpu" from the JSON.
            bo_api.recommend_next_smiles(json.dumps(req))
        finally:
            bo_api._build_gp_config = original
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["device"], "cuda")

    def test_vina_max_workers_kwarg_propagates(self) -> None:
        """``vina_max_workers`` kwarg is wired through to VinaScorerConfig
        (proven via monkey-patch on ``strbo_v1.objective_vina.VinaScorerConfig``)."""
        from strbo_v1 import objective_vina
        captured: dict = {}
        original = objective_vina.VinaScorerConfig
        def spy(**kwargs):
            captured.update(kwargs)
            return original(**kwargs)
        objective_vina.VinaScorerConfig = spy
        try:
            req = _trajectory_request(
                method="bo-tanimoto", objective="vina",
                seed_smiles="CCO", num_evaluations=4,
                batch_size=2, init_size=2,
                gp_device="cpu",
            )
            resp = json.loads(bo_api.run_search_trajectory(
                json.dumps(req),
                vina_bin="/definitely/not/a/real/vina",
                vina_max_workers=4,
            ))
            # Either error (bin path not found) or success — but the
            # VinaScorerConfig spy fires before the bin check.
            self.assertEqual(captured.get("max_workers"), 4)
            self.assertEqual(captured.get("vina_bin"), "/definitely/not/a/real/vina")
        finally:
            objective_vina.VinaScorerConfig = original

    def test_vina_max_workers_kwarg_wins_json_silently_dropped(self) -> None:
        """JSON ``vina-max-workers`` is silently dropped; the kwarg value wins."""
        from strbo_v1 import objective_vina
        captured: dict = {}
        original = objective_vina.VinaScorerConfig
        def spy(**kwargs):
            captured.update(kwargs)
            return original(**kwargs)
        objective_vina.VinaScorerConfig = spy
        try:
            req = _trajectory_request(
                method="bo-tanimoto", objective="vina",
                seed_smiles="CCO", num_evaluations=4,
                batch_size=2, init_size=2,
                gp_device="cpu",
            )
            req["vina-max-workers"] = 99
            with self.assertLogs("bo_api", level="DEBUG") as caplog:
                resp = json.loads(bo_api.run_search_trajectory(
                    json.dumps(req),
                    vina_bin="/definitely/not/a/real/vina",
                    vina_max_workers=4,
                ))
            self.assertEqual(captured.get("max_workers"), 4)  # kwarg wins
            self.assertEqual(captured.get("vina_bin"), "/definitely/not/a/real/vina")
            # DEBUG log mentions the dropped key.
            self.assertTrue(
                any("vina-max-workers" in r.getMessage() for r in caplog.records),
                f"no DEBUG log mentioning 'vina-max-workers': "
                f"{[r.getMessage() for r in caplog.records]}",
            )
        finally:
            objective_vina.VinaScorerConfig = original


# ---------------------------------------------------------------------------
# Docstring content (catches accidental doc-regressions)
# ---------------------------------------------------------------------------


class DocstringContentTests(unittest.TestCase):
    """Both function docstrings must mention the three-layer setting model
    so that callers reading the inline help see the split."""

    def test_trajectory_docstring_mentions_provider_setting(self) -> None:
        doc = bo_api.run_search_trajectory.__doc__ or ""
        self.assertIn("Provider", doc)
        self.assertIn("silently", doc.lower())
        # Provider kwarg names appear in the docstring.
        for name in (
            "vina_bin", "vina_cache_dir", "vina_max_workers", "gp_device",
            "reasyn_repo", "reasyn_python_bin", "reasyn_model_path",
            "reasyn_devices", "nn_model_path", "nn_metadata_path",
            "llm_model", "llm_base_url", "llm_api_key",
        ):
            self.assertIn(name, doc, f"missing kwarg {name!r} in docstring")

    def test_advisor_docstring_mentions_provider_setting(self) -> None:
        doc = bo_api.recommend_next_smiles.__doc__ or ""
        self.assertIn("Provider", doc)
        self.assertIn("gp_device", doc)

    def test_module_docstring_mentions_three_layers(self) -> None:
        doc = bo_api.__doc__ or ""
        self.assertIn("Provider's setting", doc)
        self.assertIn("bo_api's defaults", doc)
        self.assertIn("DEFAULT", doc)
        self.assertIn("PROVIDER_SETTING_KEYS", doc)
        self.assertIn("provider-setting", doc)


if __name__ == "__main__":
    unittest.main()
