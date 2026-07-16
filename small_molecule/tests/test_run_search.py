"""Tests for ``run_search.py`` (single-method, single-seed JSON output)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_SEARCH = REPO_ROOT / "run_search.py"


def _run(argv: list[str]) -> int:
    """Invoke ``run_search.main`` with the given argv; returns exit code."""
    # Imported lazily so tests work regardless of cwd.
    sys.path.insert(0, str(REPO_ROOT))
    import run_search  # type: ignore
    # Force logger re-init between runs.
    return run_search.main(argv)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


class OutputPathResolutionTests(unittest.TestCase):
    def test_directory_output_auto_names(self) -> None:
        from run_search import resolve_output_path  # type: ignore

        ns = mock.Mock()
        ns.output = "output/bo"
        path = resolve_output_path(ns, method="bo-tanimoto", seed=7)
        self.assertEqual(path, Path("output/bo/bo-tanimoto_seed=7.json"))

    def test_explicit_json_path(self) -> None:
        from run_search import resolve_output_path  # type: ignore

        ns = mock.Mock()
        ns.output = "/tmp/foo/bar.json"
        path = resolve_output_path(ns, method="random", seed=0)
        self.assertEqual(path, Path("/tmp/foo/bar.json"))


class ArgparseDefaultTests(unittest.TestCase):
    def test_nn_model_path_defaults_to_g12d_model(self) -> None:
        from run_search import _build_argparser  # type: ignore
        from strbo_v1.experiment_defaults import DEFAULT_NN_MODEL_PATH

        args = _build_argparser().parse_args([])

        self.assertEqual(args.nn_model_path, DEFAULT_NN_MODEL_PATH)


class ConfigEchoTests(unittest.TestCase):
    def test_config_echo_contains_required_keys(self) -> None:
        from run_search import _config_echo  # type: ignore

        ns = mock.Mock()
        ns._seed_smiles_list = ["CCO", "CCN"]
        ns.num_evaluations = 30
        ns.batch_size = 1
        ns.init_size = 10
        ns.acquisition = "ei"
        ns.xi = 0.01
        ns.kappa = 2.0
        ns.minimize = True
        ns.acq_budget = 500
        ns.pool_min_size = 1
        ns.pool_max_size = None
        ns.smiles_max_len = 50
        ns.mock = False
        ns.gp_device = "cpu"
        ns.gp_fit_itersteps = 100
        ns.gp_learning_rate = 0.05
        ns.gp_min_jitter = 1e-6
        ns.gp_max_jitter = 1e-1
        ns.gp_standardize_y = True
        ns.gp_fp_radius = 2
        ns.gp_fp_n_bits = 2048
        ns.vina_bin = "x"
        ns.vina_cache_dir = "y"
        ns.vina_pdb_id = "z"
        ns.vina_chain_id = "A"
        ns.vina_ligand_resname = None
        ns.vina_exhaustiveness = 4
        ns.vina_n_poses = 3
        ns.vina_seed = 42
        ns.vina_max_workers = 1
        ns.vina_allow_debug_receptor = False
        ns.vina_no_cache = False
        ns.reasyn_model_path = "m"
        ns.reasyn_devices = "1,2"
        ns.reasyn_repo = "r"
        ns.reasyn_python_bin = None
        ns.reasyn_search_width = 5
        ns.reasyn_exhaustiveness = 8
        ns.reasyn_num_cycles = 3
        ns.reasyn_num_editflow_samples = 15
        ns.reasyn_num_editflow_steps = 40
        ns.reasyn_time_limit = 30
        ns.reasyn_num_workers_per_gpu = 1
        ns.reasyn_filter_sim = 0.75
        ns.reasyn_no_canonicalize = False

        cfg = _config_echo(
            ns, method="bo-tanimoto", seed=0,
            parts=["vina"], minimize=(True,), ref_point=None,
        )

        self.assertEqual(cfg["method"], "bo-tanimoto")
        self.assertEqual(cfg["seed"], 0)
        self.assertEqual(cfg["acq_budget"], 500)
        self.assertEqual(cfg["gp"]["impl"], "fingerprint+tanimoto")
        self.assertEqual(cfg["n_objectives"], 1)
        self.assertEqual(cfg["objective_parts"], ["vina"])
        self.assertIn("vina", cfg)
        self.assertIn("reasyn", cfg)


class WriteJsonTests(unittest.TestCase):
    def test_write_json_schema(self) -> None:
        from run_search import write_json  # type: ignore

        cfg = {"method": "random", "seed": 0, "minimize": True}
        history = [("CCO", -7.5), ("CCN", -7.2), ("CCC", None)]
        path = Path("/tmp/test_run_search_write.json")
        if path.exists():
            path.unlink()
        write_json(cfg, history, path)
        try:
            payload = _read_json(path)
            self.assertIn("config", payload)
            self.assertIn("history", payload)
            self.assertEqual(payload["config"], cfg)
            self.assertEqual(len(payload["history"]), 3)
            # Three keys per entry.
            for entry in payload["history"]:
                self.assertEqual(set(entry.keys()), {"index", "smiles", "score"})
            # Indices are zero-based and consecutive.
            self.assertEqual([e["index"] for e in payload["history"]], [0, 1, 2])
            # None → JSON null.
            self.assertIsNone(payload["history"][2]["score"])
        finally:
            path.unlink(missing_ok=True)


class EndToEndMockTests(unittest.TestCase):
    """Smoke test: actually run run_search.main with --objective mock."""

    def setUp(self) -> None:
        self.tmpdir = Path("/tmp/test_run_search_e2e")
        self.tmpdir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        for f in self.tmpdir.iterdir():
            f.unlink(missing_ok=True)
        self.tmpdir.rmdir()

    def test_random_method_writes_json(self) -> None:
        out = self.tmpdir / "out.json"
        rc = _run([
            "--objective", "mock",
            "--method", "random",
            "--seed", "0",
            "--num-evaluations", "5",
            "--batch-size", "1",
            "--seed-smiles", "CCO,CCN",
            "--pool-min-size", "1",
            "--output", str(out),
            "--log-level", "ERROR",
        ])
        self.assertEqual(rc, 0)
        self.assertTrue(out.exists())
        payload = _read_json(out)
        self.assertEqual(payload["config"]["method"], "random")
        self.assertEqual(payload["config"]["seed"], 0)
        self.assertGreater(len(payload["history"]), 0)
        for entry in payload["history"]:
            self.assertEqual(set(entry.keys()), {"index", "smiles", "score"})

    def test_random_best_writes_history(self) -> None:
        out = self.tmpdir / "rb.json"
        rc = _run([
            "--objective", "mock",
            "--method", "random-best",
            "--seed", "0",
            "--num-evaluations", "5",
            "--batch-size", "1",
            "--seed-smiles", "CCO,CCN",
            "--pool-min-size", "1",
            "--output", str(out),
            "--log-level", "ERROR",
        ])
        self.assertEqual(rc, 0)
        payload = _read_json(out)
        self.assertEqual(payload["config"]["method"], "random-best")

    def test_bo_tanimoto_writes_history(self) -> None:
        out = self.tmpdir / "bot.json"
        rc = _run([
            "--objective", "mock",
            "--method", "bo-tanimoto",
            "--seed", "0",
            "--num-evaluations", "5",
            "--batch-size", "1",
            "--init-size", "2",
            "--seed-smiles", "CCO,CCN",
            "--output", str(out),
            "--gp-device", "cpu",
            "--gp-fit-itersteps", "5",
            "--log-level", "ERROR",
        ])
        self.assertEqual(rc, 0)
        payload = _read_json(out)
        self.assertEqual(payload["config"]["method"], "bo-tanimoto")
        self.assertEqual(payload["config"]["gp"]["impl"], "fingerprint+tanimoto")
        self.assertGreater(len(payload["history"]), 0)

    def test_bo_strkernel_writes_history(self) -> None:
        out = self.tmpdir / "bos.json"
        rc = _run([
            "--objective", "mock",
            "--method", "bo-strkernel",
            "--seed", "0",
            "--num-evaluations", "5",
            "--batch-size", "1",
            "--init-size", "2",
            "--seed-smiles", "CCO,CCN",
            "--output", str(out),
            "--gp-device", "cpu",
            "--gp-fit-itersteps", "5",
            "--smiles-max-len", "50",
            "--log-level", "ERROR",
        ])
        self.assertEqual(rc, 0)
        payload = _read_json(out)
        self.assertEqual(payload["config"]["method"], "bo-strkernel")
        self.assertEqual(payload["config"]["gp"]["impl"], "smiles-strkernel")

    def test_acq_budget_is_propagated(self) -> None:
        out = self.tmpdir / "budget.json"
        rc = _run([
            "--objective", "mock",
            "--method", "bo-tanimoto",
            "--seed", "0",
            "--num-evaluations", "5",
            "--batch-size", "1",
            "--init-size", "2",
            "--seed-smiles", "CCO,CCN",
            "--acq-budget", "3",
            "--output", str(out),
            "--gp-device", "cpu",
            "--gp-fit-itersteps", "5",
            "--log-level", "ERROR",
        ])
        self.assertEqual(rc, 0)
        payload = _read_json(out)
        self.assertEqual(payload["config"]["acq_budget"], 3)

    def test_directory_output_uses_method_seed_filename(self) -> None:
        rc = _run([
            "--objective", "mock",
            "--method", "random",
            "--seed", "42",
            "--num-evaluations", "3",
            "--batch-size", "1",
            "--seed-smiles", "CCO",
            "--pool-min-size", "1",
            "--output", str(self.tmpdir),
            "--log-level", "ERROR",
        ])
        self.assertEqual(rc, 0)
        expected = self.tmpdir / "random_seed=42.json"
        self.assertTrue(expected.exists())


class MaxPoolSizeTests(unittest.TestCase):
    """``--max-pool-size`` propagates to BO config; random methods ignore it."""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="run_search_max_pool_"))

    def tearDown(self) -> None:
        for f in self.tmpdir.iterdir():
            f.unlink(missing_ok=True)
        self.tmpdir.rmdir()

    def test_max_pool_size_arg_propagates_to_bo_config(self) -> None:
        out = self.tmpdir / "bot.json"
        rc = _run([
            "--objective", "mock",
            "--method", "bo-tanimoto",
            "--seed", "0",
            "--num-evaluations", "4",
            "--batch-size", "1",
            "--init-size", "2",
            "--max-pool-size", "7",
            "--seed-smiles", "CCO,CCN",
            "--output", str(out),
            "--gp-device", "cpu",
            "--gp-fit-itersteps", "5",
            "--log-level", "ERROR",
        ])
        self.assertEqual(rc, 0)
        import json
        payload = json.loads(out.read_text())
        self.assertEqual(payload["config"]["max_pool_size"], 7)

    def test_max_pool_size_in_random_ignored(self) -> None:
        """``--method random`` doesn't take max-pool-size; the run still succeeds
        (the flag is silently ignored for non-BO methods)."""
        out = self.tmpdir / "r.json"
        rc = _run([
            "--objective", "mock",
            "--method", "random",
            "--seed", "0",
            "--num-evaluations", "3",
            "--batch-size", "1",
            "--seed-smiles", "CCO",
            "--max-pool-size", "7",
            "--output", str(out),
            "--log-level", "ERROR",
        ])
        self.assertEqual(rc, 0)
        # max_pool_size is still echoed in the JSON (for transparency), but
        # it has no effect on random methods.
        import json
        payload = json.loads(out.read_text())
        self.assertEqual(payload["config"]["max_pool_size"], 7)


class MaxLenFilterTests(unittest.TestCase):
    """``--smiles-max-len`` propagates to BO + GP config and to random config."""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="run_search_max_len_"))

    def tearDown(self) -> None:
        for f in self.tmpdir.iterdir():
            f.unlink(missing_ok=True)
        self.tmpdir.rmdir()

    def test_smiles_max_len_arg_propagates_to_bo_config(self) -> None:
        """--smiles-max-len drives both BayesianAnalogSearchConfig.smiles_max_len
        and GPConfig.smiles_maxlen (single source of truth)."""
        out = self.tmpdir / "bot.json"
        rc = _run([
            "--objective", "mock",
            "--method", "bo-tanimoto",
            "--seed", "0",
            "--num-evaluations", "4",
            "--batch-size", "1",
            "--init-size", "2",
            "--smiles-max-len", "25",
            "--seed-smiles", "CCO,CCN",
            "--output", str(out),
            "--gp-device", "cpu",
            "--gp-fit-itersteps", "5",
            "--log-level", "ERROR",
        ])
        self.assertEqual(rc, 0)
        import json
        payload = json.loads(out.read_text())
        # Loop-level filter
        self.assertEqual(payload["config"]["smiles_max_len"], 25)
        # GP-level featurization cap (same value)
        self.assertEqual(payload["config"]["gp"]["smiles_maxlen"], 25)

    def test_smiles_max_len_arg_propagates_to_random_config(self) -> None:
        """--smiles-max-len is echoed in the random-method JSON config too."""
        out = self.tmpdir / "r.json"
        rc = _run([
            "--objective", "mock",
            "--method", "random",
            "--seed", "0",
            "--num-evaluations", "3",
            "--batch-size", "1",
            "--smiles-max-len", "30",
            "--seed-smiles", "CCO",
            "--output", str(out),
            "--log-level", "ERROR",
        ])
        self.assertEqual(rc, 0)
        import json
        payload = json.loads(out.read_text())
        self.assertEqual(payload["config"]["smiles_max_len"], 30)


class SeedSmilesParseTests(unittest.TestCase):
    """Tests for the ``_parse_seed_smiles`` helper.

    Covers file-vs-comma detection, RDKit validation, auto-canonicalization,
    blank-line filtering, and error-message context (file + line number
    for file mode; 1-based position for comma mode).
    """

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="run_search_seed_smiles_"))

    def tearDown(self) -> None:
        for f in self.tmpdir.iterdir():
            f.unlink(missing_ok=True)
        self.tmpdir.rmdir()

    def _parse(self, text: str) -> list[str]:
        from run_search import _parse_seed_smiles  # type: ignore
        return _parse_seed_smiles(text)

    # ---- comma mode -------------------------------------------------------

    def test_comma_mode_happy_path(self) -> None:
        self.assertEqual(
            self._parse("CCO,CCN,CCC"),
            ["CCO", "CCN", "CCC"],
        )

    def test_comma_mode_canonicalizes(self) -> None:
        # OCC / CCO / C(C)O / C(O)C all canonicalize to "CCO".
        self.assertEqual(
            self._parse("OCC,CCO,C(C)O,C(O)C"),
            ["CCO", "CCO", "CCO", "CCO"],
        )

    def test_comma_mode_invalid_raises_with_position(self) -> None:
        with self.assertRaises(ValueError) as cm:
            self._parse("CCO,not_a_smiles,CCN")
        msg = str(cm.exception)
        self.assertIn("position 2", msg)
        self.assertIn("not_a_smiles", msg)

    def test_comma_mode_filters_blanks(self) -> None:
        # Comma-only and empty tokens are filtered; only valid SMILES pass.
        self.assertEqual(
            self._parse("CCO,,,CCN,"),
            ["CCO", "CCN"],
        )

    def test_comma_mode_empty_input_returns_empty(self) -> None:
        self.assertEqual(self._parse(""), [])
        self.assertEqual(self._parse("   "), [])

    # ---- file mode --------------------------------------------------------

    def _write_seed_file(self, name: str, body: str) -> Path:
        path = self.tmpdir / name
        path.write_text(body, encoding="utf-8")
        return path

    def test_file_mode_happy_path(self) -> None:
        path = self._write_seed_file("seeds.smi", "CCO\nCCN\nCCC\n")
        self.assertEqual(
            self._parse(str(path)),
            ["CCO", "CCN", "CCC"],
        )

    def test_file_mode_canonicalizes_per_line(self) -> None:
        path = self._write_seed_file("seeds.smi", "OCC\nC(C)O\nCCN\n")
        self.assertEqual(self._parse(str(path)), ["CCO", "CCO", "CCN"])

    def test_file_mode_invalid_raises_with_line_number(self) -> None:
        path = self._write_seed_file("seeds.smi", "CCO\nnot_a_smiles\nCCN\n")
        with self.assertRaises(ValueError) as cm:
            self._parse(str(path))
        msg = str(cm.exception)
        self.assertIn("line 2", msg)
        self.assertIn("not_a_smiles", msg)
        self.assertIn(str(path), msg)

    def test_file_mode_filters_blank_lines(self) -> None:
        path = self._write_seed_file("seeds.smi", "CCO\n\n   \n\t\nCCN\n")
        self.assertEqual(self._parse(str(path)), ["CCO", "CCN"])

    def test_file_mode_invalid_on_first_nonblank_line_reports_line_1(self) -> None:
        # Leading blank line is filtered; first non-blank is line 2 and
        # is the bad one.
        path = self._write_seed_file("seeds.smi", "\nnot_a_smiles\nCCN\n")
        with self.assertRaises(ValueError) as cm:
            self._parse(str(path))
        self.assertIn("line 2", str(cm.exception))

    def test_file_mode_empty_file_returns_empty(self) -> None:
        path = self._write_seed_file("empty.smi", "")
        self.assertEqual(self._parse(str(path)), [])

    def test_file_mode_only_blank_lines_returns_empty(self) -> None:
        path = self._write_seed_file("blank.smi", "\n   \n\t\n")
        self.assertEqual(self._parse(str(path)), [])

    # ---- fallback behaviour ----------------------------------------------

    def test_non_existent_path_falls_through_to_comma_mode(self) -> None:
        # "not_a_file.smi" does not exist in tmpdir; should be parsed as
        # the first token of a comma-separated list, then validated as
        # SMILES and rejected.
        fake = str(self.tmpdir / "not_a_file.smi")
        self.assertFalse(Path(fake).exists())
        with self.assertRaises(ValueError) as cm:
            self._parse(f"{fake},CCO")
        msg = str(cm.exception)
        self.assertIn("position 1", msg)
        self.assertIn("not_a_file.smi", msg)


# ---------------------------------------------------------------------------
# Multi-objective tests (--objective vina+nn)
# ---------------------------------------------------------------------------


class ParseObjectiveTests(unittest.TestCase):
    def test_single_objective(self) -> None:
        from run_search import _parse_objective
        self.assertEqual(_parse_objective("vina"), ["vina"])
        self.assertEqual(_parse_objective("nn"), ["nn"])
        self.assertEqual(_parse_objective("mock"), ["mock"])

    def test_two_objectives(self) -> None:
        from run_search import _parse_objective
        self.assertEqual(_parse_objective("vina+nn"), ["vina", "nn"])
        self.assertEqual(_parse_objective("nn+vina"), ["nn", "vina"])

    def test_three_objectives(self) -> None:
        from run_search import _parse_objective
        self.assertEqual(
            _parse_objective("vina+nn+mock"), ["vina", "nn", "mock"]
        )

    def test_empty_raises(self) -> None:
        from run_search import _parse_objective
        with self.assertRaises(ValueError):
            _parse_objective("")
        with self.assertRaises(ValueError):
            _parse_objective("+")

    def test_unknown_part_raises(self) -> None:
        from run_search import _parse_objective
        with self.assertRaises(ValueError):
            _parse_objective("vina+unknown")

    def test_whitespace_tolerated(self) -> None:
        from run_search import _parse_objective
        self.assertEqual(_parse_objective("vina + nn"), ["vina", "nn"])


class ParseRefPointTests(unittest.TestCase):
    def test_none_yields_none(self) -> None:
        from run_search import _parse_ref_point
        self.assertIsNone(_parse_ref_point(None))
        self.assertIsNone(_parse_ref_point(""))
        self.assertIsNone(_parse_ref_point("   "))

    def test_comma_split(self) -> None:
        from run_search import _parse_ref_point
        self.assertEqual(_parse_ref_point("0,5"), (0.0, 5.0))
        self.assertEqual(_parse_ref_point("1.0, 2.5, 3.14"), (1.0, 2.5, 3.14))


class SummarizeHistoryTests(unittest.TestCase):
    def test_single_obj_returns_bsf(self) -> None:
        from run_search import summarize_history
        history = [("CCO", -5.0), ("CCN", -7.0), ("CCC", -6.0)]
        out = summarize_history(
            history, ref_point=None, num_evaluations=3, minimize=(True,),
        )
        self.assertIn("bsf", out)
        # bsf should be non-increasing for minimize.
        bsf = out["bsf"]
        self.assertEqual(len(bsf), 3)
        for i in range(1, len(bsf)):
            self.assertLessEqual(bsf[i], bsf[i - 1])

    def test_two_obj_returns_hypervolume(self) -> None:
        from run_search import summarize_history
        history = [
            ("CCO", (1.0, 5.0)),
            ("CCN", (2.0, 6.0)),
        ]
        out = summarize_history(
            history, ref_point=(5.0, 10.0), num_evaluations=2, minimize=(True, False),
        )
        self.assertIn("hypervolume", out)
        # HV is non-decreasing.
        hv = out["hypervolume"]
        self.assertEqual(len(hv), 2)
        self.assertLessEqual(hv[0], hv[1])

    def test_three_obj_graceful_degradation(self) -> None:
        from run_search import summarize_history
        history = [
            ("CCO", (1.0, 5.0, 2.0)),
            ("CCN", (2.0, 6.0, 3.0)),
        ]
        out = summarize_history(
            history, ref_point=(5.0, 10.0, 4.0), num_evaluations=2,
            minimize=(True, False, False),
        )
        self.assertIn("bsf_per_objective", out)
        # shape (n_obj, num_evaluations) = (3, 2).
        self.assertEqual(out["bsf_per_objective"].shape, (3, 2))


class BuildScorersAndMinimizeTests(unittest.TestCase):
    def test_single_objective_returns_single_scorer(self) -> None:
        from run_search import _build_scorers_and_minimize
        from unittest import mock
        args = mock.Mock()
        args.objective = "mock"
        scorer, minimize = _build_scorers_and_minimize(args)
        self.assertFalse(isinstance(scorer, tuple))
        self.assertEqual(minimize, (True,))

    def test_two_objectives_returns_tuple(self) -> None:
        from run_search import _build_scorers_and_minimize
        from unittest import mock
        args = mock.Mock()
        args.objective = "vina+nn"
        # We can't actually build the real Vina / NN scorers in tests
        # without their binaries / model file, so monkey-patch the
        # one-scorer factory.
        import run_search
        with mock.patch.object(run_search, "_build_one_scorer", side_effect=lambda p, a: p):
            scorer, minimize = _build_scorers_and_minimize(args)
        self.assertEqual(scorer, ("vina", "nn"))
        # vina → True, nn → False.
        self.assertEqual(minimize, (True, False))

    def test_three_objectives(self) -> None:
        from run_search import _build_scorers_and_minimize
        from unittest import mock
        args = mock.Mock()
        args.objective = "vina+nn+mock"
        import run_search
        with mock.patch.object(run_search, "_build_one_scorer", side_effect=lambda p, a: p):
            scorer, minimize = _build_scorers_and_minimize(args)
        self.assertEqual(scorer, ("vina", "nn", "mock"))
        self.assertEqual(minimize, (True, False, True))


class ConfigEchoMultiObjTests(unittest.TestCase):
    def test_minimize_is_list_for_multi_obj(self) -> None:
        from run_search import _config_echo
        from unittest import mock
        ns = mock.Mock()
        ns._seed_smiles_list = ["CCO", "CCN"]
        ns.num_evaluations = 10
        ns.batch_size = 1
        ns.init_size = 2
        ns.acquisition = "ei"
        ns.xi = 0.01
        ns.kappa = 2.0
        ns.acq_budget = None
        ns.pool_min_size = 1
        ns.pool_max_size = None
        ns.smiles_max_len = 50
        ns.objective = "vina+nn"
        ns.gp_device = "cpu"
        ns.gp_fit_itersteps = 10
        ns.gp_learning_rate = 0.1
        ns.gp_min_jitter = 1e-6
        ns.gp_max_jitter = 1e-1
        ns.gp_standardize_y = True
        ns.gp_fp_radius = 2
        ns.gp_fp_n_bits = 2048
        ns.vina_bin = "x"; ns.vina_cache_dir = "y"
        ns.vina_pdb_id = "z"; ns.vina_chain_id = "A"
        ns.vina_ligand_resname = None
        ns.vina_exhaustiveness = 4; ns.vina_n_poses = 3
        ns.vina_seed = 42; ns.vina_max_workers = 1
        ns.vina_allow_debug_receptor = False; ns.vina_no_cache = False
        ns.reasyn_model_path = "m"; ns.reasyn_devices = "1,2"
        ns.reasyn_repo = "r"; ns.reasyn_python_bin = None
        ns.reasyn_search_width = 5; ns.reasyn_exhaustiveness = 8
        ns.reasyn_num_cycles = 3; ns.reasyn_num_editflow_samples = 10
        ns.reasyn_num_editflow_steps = 30; ns.reasyn_time_limit = 20
        ns.reasyn_num_workers_per_gpu = 1; ns.reasyn_filter_sim = 0.8
        ns.reasyn_no_canonicalize = False
        ns.ehvi_n_samples = 128
        ns.che_alpha = 1.0

        cfg = _config_echo(
            ns, method="bo-tanimoto", seed=0,
            parts=["vina", "nn"], minimize=(True, False),
            ref_point=(0.0, 5.0),
        )
        self.assertEqual(cfg["n_objectives"], 2)
        self.assertEqual(cfg["objective_parts"], ["vina", "nn"])
        self.assertEqual(cfg["minimize"], [True, False])
        self.assertEqual(cfg["ref_point"], [0.0, 5.0])
        self.assertEqual(cfg["ehvi_n_samples"], 128)
        self.assertIn("vina", cfg)  # echoed because one part is vina

    def test_minimize_is_bool_for_single_obj(self) -> None:
        from run_search import _config_echo
        from unittest import mock
        ns = mock.Mock()
        ns._seed_smiles_list = ["CCO"]
        ns.num_evaluations = 10
        ns.batch_size = 1
        ns.init_size = 2
        ns.acquisition = "ei"
        ns.xi = 0.01
        ns.kappa = 2.0
        ns.acq_budget = None
        ns.pool_min_size = 1
        ns.pool_max_size = None
        ns.smiles_max_len = 50
        ns.objective = "nn"
        ns.gp_device = "cpu"
        ns.gp_fit_itersteps = 10
        ns.gp_learning_rate = 0.1
        ns.gp_min_jitter = 1e-6
        ns.gp_max_jitter = 1e-1
        ns.gp_standardize_y = True
        ns.gp_fp_radius = 2
        ns.gp_fp_n_bits = 2048
        ns.vina_bin = "x"; ns.vina_cache_dir = "y"
        ns.vina_pdb_id = "z"; ns.vina_chain_id = "A"
        ns.vina_ligand_resname = None
        ns.vina_exhaustiveness = 4; ns.vina_n_poses = 3
        ns.vina_seed = 42; ns.vina_max_workers = 1
        ns.vina_allow_debug_receptor = False; ns.vina_no_cache = False
        ns.reasyn_model_path = "m"; ns.reasyn_devices = "1,2"
        ns.reasyn_repo = "r"; ns.reasyn_python_bin = None
        ns.reasyn_search_width = 5; ns.reasyn_exhaustiveness = 8
        ns.reasyn_num_cycles = 3; ns.reasyn_num_editflow_samples = 10
        ns.reasyn_num_editflow_steps = 30; ns.reasyn_time_limit = 20
        ns.reasyn_num_workers_per_gpu = 1; ns.reasyn_filter_sim = 0.8
        ns.reasyn_no_canonicalize = False
        ns.ehvi_n_samples = 128
        ns.che_alpha = 1.0

        cfg = _config_echo(
            ns, method="bo-tanimoto", seed=0,
            parts=["nn"], minimize=(False,), ref_point=None,
        )
        # n_obj==1 → minimize is bare bool, ref_point omitted.
        self.assertEqual(cfg["n_objectives"], 1)
        self.assertEqual(cfg["minimize"], False)
        self.assertNotIn("ref_point", cfg)
        self.assertNotIn("vina", cfg)  # no vina part → no vina echo


class VinaNnObjectiveEndToEndTests(unittest.TestCase):
    """End-to-end CLI invocations of ``--objective vina+nn`` (mocked
    backends to avoid needing vina binary / NN model)."""

    def test_2obj_cli_random(self) -> None:
        """`--objective vina+nn` with mocked backends, --method random."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            output_dir.mkdir()
            argv = [
                "--method", "random",
                "--seed", "0",
                "--objective", "vina+nn",
                "--num-evaluations", "3",
                "--batch-size", "1",
                "--pool-min-size", "1",
                "--pool-max-size", "10",
                "--smiles-max-len", "50",
                "--ref-point", "0,5",
                "--output", str(output_dir / "r.json"),
            ]
            import run_search
            import unittest.mock as mock_mod
            with mock_mod.patch.object(run_search, "_build_vina_scorer",
                                       return_value=lambda smis: [-float(s.count("C")) for s in smis]), \
                 mock_mod.patch.object(run_search, "_build_nn_scorer",
                                       return_value=lambda smis: [5.0 + 0.5 * float(s.count("N")) for s in smis]), \
                 mock_mod.patch.object(run_search, "_build_reasyn_analog",
                                       return_value=lambda smis: [s + "C" for s in smis]):
                rc = _run(argv)
            self.assertEqual(rc, 0)
            payload = json.loads((output_dir / "r.json").read_text())
            self.assertEqual(payload["config"]["n_objectives"], 2)
            self.assertEqual(payload["config"]["objective_parts"], ["vina", "nn"])
            self.assertEqual(payload["config"]["minimize"], [True, False])
            self.assertEqual(payload["config"]["ref_point"], [0.0, 5.0])
            for entry in payload["history"]:
                self.assertIn("scores", entry)
                self.assertEqual(len(entry["scores"]), 2)
                self.assertNotIn("score", entry)

    def test_3obj_cli_random(self) -> None:
        """`--objective vina+nn+mock` exercises the n_obj=3 Chebyshev path."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            output_dir.mkdir()
            argv = [
                "--method", "random",
                "--seed", "0",
                "--objective", "vina+nn+mock",
                "--num-evaluations", "3",
                "--batch-size", "1",
                "--pool-min-size", "1",
                "--pool-max-size", "10",
                "--smiles-max-len", "50",
                "--output", str(output_dir / "r.json"),
            ]
            import run_search
            import unittest.mock as mock_mod
            with mock_mod.patch.object(run_search, "_build_vina_scorer",
                                       return_value=lambda smis: [-float(s.count("C")) for s in smis]), \
                 mock_mod.patch.object(run_search, "_build_nn_scorer",
                                       return_value=lambda smis: [5.0 + 0.5 * float(s.count("N")) for s in smis]), \
                 mock_mod.patch.object(run_search, "_build_reasyn_analog",
                                       return_value=lambda smis: [s + "C" for s in smis]):
                rc = _run(argv)
            self.assertEqual(rc, 0)
            payload = json.loads((output_dir / "r.json").read_text())
            self.assertEqual(payload["config"]["n_objectives"], 3)
            self.assertEqual(payload["config"]["objective_parts"], ["vina", "nn", "mock"])
            self.assertEqual(payload["config"]["minimize"], [True, False, True])
            for entry in payload["history"]:
                self.assertEqual(len(entry["scores"]), 3)

    def test_ref_point_length_mismatch_raises(self) -> None:
        """--ref-point length must match n_obj."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            output_dir.mkdir()
            argv = [
                "--method", "random",
                "--seed", "0",
                "--objective", "vina+nn",
                "--num-evaluations", "1",
                "--ref-point", "0",  # length 1, n_obj=2
                "--output", str(output_dir / "r.json"),
            ]
            with self.assertRaises(SystemExit):
                _run(argv)

    def test_unknown_objective_part_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            output_dir.mkdir()
            argv = [
                "--method", "random",
                "--seed", "0",
                "--objective", "vina+unknown",
                "--num-evaluations", "1",
                "--output", str(output_dir / "r.json"),
            ]
            with self.assertRaises(SystemExit):
                _run(argv)

    def test_single_obj_ref_point_ignored(self) -> None:
        """--ref-point is silently ignored for single-objective."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            output_dir.mkdir()
            argv = [
                "--method", "random",
                "--seed", "0",
                "--objective", "mock",
                "--num-evaluations", "2",
                "--ref-point", "0,5",  # ignored for single-obj
                "--output", str(output_dir / "r.json"),
            ]
            rc = _run(argv)
            self.assertEqual(rc, 0)
            payload = json.loads((output_dir / "r.json").read_text())
            self.assertNotIn("ref_point", payload["config"])


if __name__ == "__main__":
    unittest.main()
