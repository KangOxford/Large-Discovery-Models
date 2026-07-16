"""Tests for external JSON and HTTP-facing interfaces."""

from __future__ import annotations

import json
import unittest
from unittest import mock

import bo_api


class BoApiExternalJsonTests(unittest.TestCase):
    def test_score_vina_json_drops_json_provider_settings(self) -> None:
        request = {
            "smiles": ["CCO"],
            "vina_bin": "/client/should/not/win",
            "vina_cache_dir": "/client/cache",
        }
        fake = {"ok": True, "items": [], "errors": []}
        with mock.patch.object(bo_api, "score_vina", return_value=fake) as patched:
            response = json.loads(bo_api.score_vina_json(
                json.dumps(request),
                vina_bin="/provider/vina",
                vina_cache_dir="/provider/cache",
            ))

        self.assertTrue(response["ok"])
        _, kwargs = patched.call_args
        self.assertEqual(kwargs["vina_bin"], "/provider/vina")
        self.assertEqual(kwargs["vina_cache_dir"], "/provider/cache")
        self.assertNotIn("vina_bin", kwargs["request"])
        self.assertNotIn("vina_cache_dir", kwargs["request"])

    def test_score_nn_json_wraps_structured_response(self) -> None:
        fake = {"ok": True, "items": [{"smiles": "CCO", "ok": True}], "errors": []}
        with mock.patch.object(bo_api, "score_nn", return_value=fake):
            response = json.loads(bo_api.score_nn_json(
                json.dumps({"smiles": ["CCO"], "nn_model_path": "/ignored"}),
                nn_model_path="/provider/model.joblib",
            ))

        self.assertTrue(response["ok"])
        self.assertEqual(response["items"][0]["smiles"], "CCO")

    def test_evaluate_acquisition_json_uses_gp_device_kwarg(self) -> None:
        fake = {"ok": True, "items": [], "errors": []}
        with mock.patch.object(bo_api, "evaluate_acquisition", return_value=fake) as patched:
            response = json.loads(bo_api.evaluate_acquisition_json(
                json.dumps({
                    "history": [{"smiles": "CCO", "score": -7.1}],
                    "query_smiles": ["CCN"],
                    "gp_device": "cuda:client",
                }),
                gp_device="cpu",
            ))

        self.assertTrue(response["ok"])
        _, kwargs = patched.call_args
        self.assertEqual(kwargs["gp_device"], "cpu")
        self.assertNotIn("gp_device", kwargs["request"])

    def test_bad_request_returns_ok_false_error(self) -> None:
        response = json.loads(bo_api.score_nn_json("[1, 2, 3]"))

        self.assertFalse(response["ok"])
        self.assertEqual(response["items"], [])
        self.assertIn("error", response)
        self.assertEqual(response["error_type"], "ValueError")


class BoApiHttpRoutesTests(unittest.TestCase):
    def test_score_vina_route(self) -> None:
        import bo_api_http

        with mock.patch.object(bo_api_http.bo_api, "score_vina_json", return_value='{"ok": true}'):
            status, body = bo_api_http.handle_request("/score/vina", "{}")

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"ok": True})

    def test_error_response_maps_to_502(self) -> None:
        import bo_api_http

        with mock.patch.object(
            bo_api_http.bo_api,
            "evaluate_acquisition_json",
            return_value='{"ok": false, "error": "bad"}',
        ):
            status, body = bo_api_http.handle_request("/acquisition/evaluate", "{}")

        self.assertEqual(status, 502)
        self.assertEqual(json.loads(body)["error"], "bad")

    def test_unknown_route_returns_404(self) -> None:
        import bo_api_http

        status, body = bo_api_http.handle_request("/missing", "{}")

        self.assertEqual(status, 404)
        self.assertIn("unknown route", json.loads(body)["error"])
