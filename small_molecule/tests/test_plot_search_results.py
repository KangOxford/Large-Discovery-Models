"""Tests for ``plot_search_results.py``."""

from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(argv: list) -> int:
    sys.path.insert(0, str(REPO_ROOT))
    import plot_search_results  # type: ignore
    try:
        return plot_search_results.main(argv)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 1
        if isinstance(code, int):
            return code
        return 1


def _write_run(
    path: Path, method: str, seed: int,
    scores, *,
    minimize: object = True,
    n_obj: int = 1,
    ref_point=None,
) -> None:
    """Write a run JSON. ``scores`` is a list of floats (n_obj=1) or
    a list of (s0, s1, ...) tuples (n_obj>=2)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "method": method,
        "seed": seed,
        "minimize": minimize,
        "num_evaluations": len(scores),
        "n_objectives": n_obj,
        "objective_parts": ["vina"] * n_obj if isinstance(minimize, list) and minimize
            else (["vina"] if n_obj == 1 else [f"obj{i}" for i in range(n_obj)]),
    }
    if ref_point is not None:
        cfg["ref_point"] = list(ref_point)
    if n_obj == 1:
        history = [
            {"index": i, "smiles": f"S{i}", "score": sc}
            for i, sc in enumerate(scores)
        ]
    else:
        history = [
            {"index": i, "smiles": f"S{i}", "scores": list(sc)}
            for i, sc in enumerate(scores)
        ]
    payload = {"config": cfg, "history": history}
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _write_run_with_none(
    path: Path, method: str, seed: int,
    entries, *,
    n_obj: int = 1,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "method": method, "seed": seed, "minimize": True,
        "num_evaluations": len(entries), "n_objectives": n_obj,
    }
    if n_obj == 1:
        history = [
            {"index": i, "smiles": f"S{i}", "score": sc}
            for i, sc in enumerate(entries)
        ]
    else:
        history = [
            {"index": i, "smiles": f"S{i}", "scores": list(sc)}
            for i, sc in enumerate(entries)
        ]
    with path.open("w", encoding="utf-8") as fh:
        json.dump({"config": cfg, "history": history}, fh)


class ParseFilenameTests(unittest.TestCase):
    def test_standard(self) -> None:
        from plot_search_results import parse_filename  # type: ignore
        self.assertEqual(
            parse_filename(Path("output/bo/random_seed=0.json")),
            ("random", 0),
        )

    def test_method_with_dash(self) -> None:
        from plot_search_results import parse_filename  # type: ignore
        self.assertEqual(
            parse_filename(Path("output/bo/bo-strkernel_seed=12.json")),
            ("bo-strkernel", 12),
        )

    def test_no_match(self) -> None:
        from plot_search_results import parse_filename  # type: ignore
        self.assertIsNone(parse_filename(Path("foo/bar.json")))


class BestSoFarSingleTests(unittest.TestCase):
    def test_minimize(self) -> None:
        from plot_search_results import best_so_far_single  # type: ignore
        history = [("a", -3.0), ("b", -5.0), ("c", -4.0)]
        out = best_so_far_single(history, num_evaluations=5, minimize=True)
        self.assertEqual(len(out), 5)
        self.assertAlmostEqual(out[0], -3.0)
        self.assertAlmostEqual(out[1], -5.0)
        self.assertAlmostEqual(out[2], -5.0)
        self.assertAlmostEqual(out[3], -5.0)
        self.assertAlmostEqual(out[4], -5.0)

    def test_maximize(self) -> None:
        from plot_search_results import best_so_far_single  # type: ignore
        history = [("a", 3.0), ("b", 5.0), ("c", 4.0)]
        out = best_so_far_single(history, num_evaluations=3, minimize=False)
        self.assertAlmostEqual(out[0], 3.0)
        self.assertAlmostEqual(out[1], 5.0)
        self.assertAlmostEqual(out[2], 5.0)

    def test_none_score_padded(self) -> None:
        from plot_search_results import best_so_far_single  # type: ignore
        history = [("a", None), ("b", -2.0), ("c", None)]
        out = best_so_far_single(history, num_evaluations=4, minimize=True)
        # None → bsf stays at +inf; finite at index 1 → bsf = -2.0.
        import math
        self.assertTrue(math.isinf(out[0]))
        self.assertAlmostEqual(out[1], -2.0)
        self.assertAlmostEqual(out[2], -2.0)
        self.assertAlmostEqual(out[3], -2.0)

    def test_tuple_score_extracted(self) -> None:
        from plot_search_results import best_so_far_single  # type: ignore
        # If the score happens to be a length-1 tuple (e.g. after
        # upgrading a JSON to multi-obj), we still extract the value.
        history = [("a", (-3.0,)), ("b", (-5.0,)), ("c", (-4.0,))]
        out = best_so_far_single(history, num_evaluations=3, minimize=True)
        self.assertAlmostEqual(out[0], -3.0)
        self.assertAlmostEqual(out[1], -5.0)


class LoadHistoryTests(unittest.TestCase):
    def test_skip_none_smiles(self) -> None:
        from plot_search_results import load_history  # type: ignore
        path = Path("/tmp/test_load_history_skip.json")
        path.write_text(json.dumps({
            "config": {"method": "x", "seed": 0, "minimize": True, "n_objectives": 1},
            "history": [{"index": 0, "smiles": None, "score": -1.0}],
        }))
        try:
            self.assertEqual(load_history(path, n_obj=1), [])
        finally:
            path.unlink(missing_ok=True)

    def test_none_score_passed_through(self) -> None:
        from plot_search_results import load_history  # type: ignore
        path = Path("/tmp/test_load_history_none_score.json")
        path.write_text(json.dumps({
            "config": {"method": "x", "seed": 0, "minimize": True, "n_objectives": 1},
            "history": [{"index": 0, "smiles": "CCO", "score": None}],
        }))
        try:
            out = load_history(path, n_obj=1)
            self.assertEqual(out, [("CCO", None)])
        finally:
            path.unlink(missing_ok=True)

    def test_n_obj_2_load(self) -> None:
        from plot_search_results import load_history  # type: ignore
        path = Path("/tmp/test_load_history_2obj.json")
        path.write_text(json.dumps({
            "config": {"method": "x", "seed": 0, "minimize": [True, False],
                       "n_objectives": 2, "ref_point": [0.0, 5.0]},
            "history": [
                {"index": 0, "smiles": "CCO", "scores": [1.0, 5.0]},
                {"index": 1, "smiles": "CCN", "scores": [2.0, 6.0]},
            ],
        }))
        try:
            out = load_history(path, n_obj=2)
            self.assertEqual(out, [("CCO", (1.0, 5.0)), ("CCN", (2.0, 6.0))])
        finally:
            path.unlink(missing_ok=True)


class LoadInputsTests(unittest.TestCase):
    def test_aggregates_across_seeds(self) -> None:
        from plot_search_results import load_inputs  # type: ignore
        tmp = Path("/tmp/test_plot_load_inputs")
        if tmp.exists():
            for f in tmp.iterdir():
                f.unlink()
        tmp.mkdir()
        try:
            _write_run(tmp / "random_seed=0.json", "random", 0, [-3.0, -4.0, -5.0])
            _write_run(tmp / "random_seed=1.json", "random", 1, [-2.0, -3.0, -3.5])
            results, meta = load_inputs(tmp, methods_filter=None)
            self.assertIn("random", results)
            self.assertEqual(set(results["random"].keys()), {0, 1})
            self.assertEqual(meta["random"]["n_obj"], 1)
        finally:
            for f in tmp.iterdir():
                f.unlink()
            tmp.rmdir()

    def test_method_filter(self) -> None:
        from plot_search_results import load_inputs  # type: ignore
        tmp = Path("/tmp/test_plot_method_filter")
        if tmp.exists():
            for f in tmp.iterdir():
                f.unlink()
        tmp.mkdir()
        try:
            _write_run(tmp / "random_seed=0.json", "random", 0, [-1.0])
            _write_run(tmp / "bo-tanimoto_seed=0.json", "bo-tanimoto", 0, [-2.0])
            results, _ = load_inputs(tmp, methods_filter={"bo-tanimoto"})
            self.assertIn("bo-tanimoto", results)
            self.assertNotIn("random", results)
        finally:
            for f in tmp.iterdir():
                f.unlink()
            tmp.rmdir()

    def test_skip_non_matching_filenames(self) -> None:
        from plot_search_results import load_inputs  # type: ignore
        tmp = Path("/tmp/test_plot_skip")
        if tmp.exists():
            for f in tmp.iterdir():
                f.unlink()
        tmp.mkdir()
        try:
            _write_run(tmp / "random_seed=0.json", "random", 0, [-1.0])
            (tmp / "notes.txt").write_text("ignore me")
            results, _ = load_inputs(tmp, methods_filter=None)
            self.assertEqual(list(results.keys()), ["random"])
        finally:
            for f in tmp.iterdir():
                f.unlink()
            tmp.rmdir()

    def test_loads_2obj_meta(self) -> None:
        from plot_search_results import load_inputs  # type: ignore
        tmp = Path("/tmp/test_plot_load_2obj")
        if tmp.exists():
            for f in tmp.iterdir():
                f.unlink()
        tmp.mkdir()
        try:
            _write_run(
                tmp / "mo_seed=0.json", "mo", 0,
                [(1.0, 5.0), (2.0, 6.0)],
                minimize=[True, False], n_obj=2, ref_point=(0.0, 5.0),
            )
            results, meta = load_inputs(tmp, methods_filter=None)
            self.assertIn("mo", results)
            self.assertEqual(meta["mo"]["n_obj"], 2)
            self.assertEqual(meta["mo"]["minimize"], (True, False))
            self.assertEqual(meta["mo"]["ref_point"], (0.0, 5.0))
        finally:
            for f in tmp.iterdir():
                f.unlink()
            tmp.rmdir()


class AggregateTests(unittest.TestCase):
    def test_mean_std_single_obj(self) -> None:
        from plot_search_results import aggregate, load_inputs  # type: ignore
        tmp = Path("/tmp/test_plot_aggregate")
        if tmp.exists():
            for f in tmp.iterdir():
                f.unlink()
        tmp.mkdir()
        try:
            _write_run(tmp / "x_seed=0.json", "x", 0, [-1.0, -2.0, -3.0])
            _write_run(tmp / "x_seed=1.json", "x", 1, [-1.5, -2.5, -3.5])
            results, meta = load_inputs(tmp, methods_filter=None)
            agg = aggregate(results, meta, num_evaluations=3)
            self.assertIn("x", agg)
            mean, std, n_obj = agg["x"]
            self.assertEqual(n_obj, 1)
            self.assertAlmostEqual(mean[0], -1.25, places=6)
            self.assertAlmostEqual(mean[1], -2.25, places=6)
            self.assertAlmostEqual(mean[2], -3.25, places=6)
            self.assertGreater(std[0], 0.0)
        finally:
            for f in tmp.iterdir():
                f.unlink()
            tmp.rmdir()

    def test_aggregate_2obj(self) -> None:
        from plot_search_results import aggregate, load_inputs  # type: ignore
        tmp = Path("/tmp/test_plot_aggregate_2obj")
        if tmp.exists():
            for f in tmp.iterdir():
                f.unlink()
        tmp.mkdir()
        try:
            _write_run(
                tmp / "x_seed=0.json", "x", 0,
                [(1.0, 5.0), (2.0, 6.0)],
                minimize=[True, False], n_obj=2, ref_point=(5.0, 10.0),
            )
            _write_run(
                tmp / "x_seed=1.json", "x", 1,
                [(1.5, 5.5), (2.5, 6.5)],
                minimize=[True, False], n_obj=2, ref_point=(5.0, 10.0),
            )
            results, meta = load_inputs(tmp, methods_filter=None)
            agg = aggregate(results, meta, num_evaluations=2)
            mean, std, n_obj = agg["x"]
            self.assertEqual(n_obj, 2)
            self.assertEqual(len(mean), 2)
        finally:
            for f in tmp.iterdir():
                f.unlink()
            tmp.rmdir()

    def test_aggregate_3obj_graceful(self) -> None:
        from plot_search_results import aggregate, load_inputs  # type: ignore
        tmp = Path("/tmp/test_plot_aggregate_3obj")
        if tmp.exists():
            for f in tmp.iterdir():
                f.unlink()
        tmp.mkdir()
        try:
            _write_run(
                tmp / "x_seed=0.json", "x", 0,
                [(1.0, 5.0, 2.0), (2.0, 6.0, 3.0)],
                minimize=[True, False, False], n_obj=3,
            )
            _write_run(
                tmp / "x_seed=1.json", "x", 1,
                [(1.5, 5.5, 2.5), (2.5, 6.5, 3.5)],
                minimize=[True, False, False], n_obj=3,
            )
            results, meta = load_inputs(tmp, methods_filter=None)
            agg = aggregate(results, meta, num_evaluations=2)
            mean, std, n_obj = agg["x"]
            self.assertEqual(n_obj, 3)
            self.assertEqual(mean.shape, (3, 2))
        finally:
            for f in tmp.iterdir():
                f.unlink()
            tmp.rmdir()

    def test_2obj_requires_ref_point(self) -> None:
        """An n_obj=2 method without a JSON-embedded ref_point and
        no --ref-point CLI override must raise a clear error."""
        from plot_search_results import aggregate, load_inputs  # type: ignore
        tmp = Path("/tmp/test_plot_2obj_no_ref")
        if tmp.exists():
            for f in tmp.iterdir():
                f.unlink()
        tmp.mkdir()
        try:
            _write_run(
                tmp / "x_seed=0.json", "x", 0,
                [(1.0, 5.0), (2.0, 6.0)],
                minimize=[True, False], n_obj=2,  # no ref_point
            )
            results, meta = load_inputs(tmp, methods_filter=None)
            with self.assertRaises(ValueError):
                aggregate(results, meta, num_evaluations=2)
        finally:
            for f in tmp.iterdir():
                f.unlink()
            tmp.rmdir()


class HypervolumeCurveTests(unittest.TestCase):
    def test_basic_2obj(self) -> None:
        from plot_search_results import hypervolume_curve  # type: ignore
        history = [
            ("CCO", (1.0, 5.0)),
            ("CCN", (2.0, 6.0)),
        ]
        out = hypervolume_curve(
            history, num_evaluations=2, ref_point=(5.0, 10.0),
            minimize=(True, False),
        )
        self.assertEqual(len(out), 2)
        # HV is non-decreasing.
        self.assertLessEqual(out[0], out[1])

    def test_length_padding(self) -> None:
        from plot_search_results import hypervolume_curve  # type: ignore
        history = [("CCO", (1.0, 5.0))]
        out = hypervolume_curve(
            history, num_evaluations=5, ref_point=(5.0, 10.0),
            minimize=(True, False),
        )
        self.assertEqual(len(out), 5)


class PerObjBestSoFarTests(unittest.TestCase):
    def test_3obj_shape(self) -> None:
        from plot_search_results import per_obj_best_so_far  # type: ignore
        history = [
            ("CCO", (1.0, 5.0, 2.0)),
            ("CCN", (2.0, 6.0, 3.0)),
        ]
        out = per_obj_best_so_far(history, num_evaluations=3, n_obj=3)
        self.assertEqual(out.shape, (3, 3))


class EndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path("/tmp/test_plot_e2e")
        if self.tmp.exists():
            for f in self.tmp.iterdir():
                f.unlink()
        self.tmp.mkdir()
        _write_run(self.tmp / "random_seed=0.json", "random", 0, [-1.0, -2.0, -3.0])
        _write_run(self.tmp / "random_seed=1.json", "random", 1, [-1.5, -2.5, -3.5])
        _write_run(self.tmp / "bo-tanimoto_seed=0.json", "bo-tanimoto", 0, [-0.5, -1.5, -4.0])
        _write_run(self.tmp / "bo-tanimoto_seed=1.json", "bo-tanimoto", 1, [-0.8, -2.0, -5.0])

    def tearDown(self) -> None:
        for f in self.tmp.iterdir():
            f.unlink(missing_ok=True)
        self.tmp.rmdir()

    def test_produces_csv_and_png(self) -> None:
        out_base = self.tmp / "summary"
        rc = _run([
            "--input-dir", str(self.tmp),
            "--output", str(out_base),
            "--figure-format", "png",
            "--log-level", "ERROR",
        ])
        self.assertEqual(rc, 0)
        self.assertTrue(out_base.with_suffix(".png").exists())
        self.assertTrue(out_base.with_suffix(".csv").exists())
        with out_base.with_suffix(".csv").open() as fh:
            reader = csv.reader(fh)
            header = next(reader)
            self.assertEqual(header[0], "evaluation")
            self.assertIn("random_mean", header)
            self.assertIn("bo-tanimoto_mean", header)

    def test_method_filter(self) -> None:
        out_base = self.tmp / "filtered"
        rc = _run([
            "--input-dir", str(self.tmp),
            "--output", str(out_base),
            "--methods", "random",
            "--figure-format", "png",
            "--log-level", "ERROR",
        ])
        self.assertEqual(rc, 0)
        with out_base.with_suffix(".csv").open() as fh:
            reader = csv.reader(fh)
            header = next(reader)
            self.assertIn("random_mean", header)
            self.assertNotIn("bo-tanimoto_mean", header)

    def test_no_csv_flag(self) -> None:
        out_base = self.tmp / "no_csv"
        rc = _run([
            "--input-dir", str(self.tmp),
            "--output", str(out_base),
            "--no-csv",
            "--figure-format", "png",
            "--log-level", "ERROR",
        ])
        self.assertEqual(rc, 0)
        self.assertTrue(out_base.with_suffix(".png").exists())
        self.assertFalse(out_base.with_suffix(".csv").exists())


class EndToEnd2ObjTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path("/tmp/test_plot_e2e_2obj")
        if self.tmp.exists():
            for f in self.tmp.iterdir():
                f.unlink()
        self.tmp.mkdir()
        _write_run(
            self.tmp / "mo_seed=0.json", "mo", 0,
            [(1.0, 5.0), (2.0, 6.0), (1.5, 5.5)],
            minimize=[True, False], n_obj=2, ref_point=(5.0, 10.0),
        )
        _write_run(
            self.tmp / "mo_seed=1.json", "mo", 1,
            [(0.5, 5.5), (1.5, 6.0), (1.0, 5.0)],
            minimize=[True, False], n_obj=2, ref_point=(5.0, 10.0),
        )

    def tearDown(self) -> None:
        for f in self.tmp.iterdir():
            f.unlink(missing_ok=True)
        self.tmp.rmdir()

    def test_2obj_produces_png(self) -> None:
        out_base = self.tmp / "summary"
        rc = _run([
            "--input-dir", str(self.tmp),
            "--output", str(out_base),
            "--figure-format", "png",
            "--log-level", "ERROR",
        ])
        self.assertEqual(rc, 0)
        self.assertTrue(out_base.with_suffix(".png").exists())
        with out_base.with_suffix(".csv").open() as fh:
            reader = csv.reader(fh)
            header = next(reader)
            self.assertIn("mo_mean", header)
            self.assertIn("mo_std", header)

    def test_2obj_with_cli_ref_point(self) -> None:
        """No embedded ref_point, but --ref-point on the CLI works."""
        out_base = self.tmp / "summary"
        # Rewrite the runs without embedded ref_point.
        _write_run(
            self.tmp / "mo_seed=0.json", "mo", 0,
            [(1.0, 5.0), (2.0, 6.0), (1.5, 5.5)],
            minimize=[True, False], n_obj=2,
        )
        _write_run(
            self.tmp / "mo_seed=1.json", "mo", 1,
            [(0.5, 5.5), (1.5, 6.0), (1.0, 5.0)],
            minimize=[True, False], n_obj=2,
        )
        rc = _run([
            "--input-dir", str(self.tmp),
            "--output", str(out_base),
            "--ref-point", "5,10",
            "--figure-format", "png",
            "--log-level", "ERROR",
        ])
        self.assertEqual(rc, 0)
        self.assertTrue(out_base.with_suffix(".png").exists())

    def test_2obj_no_pareto_placeholder_text(self) -> None:
        """The 2-obj plot must be a single HV-curve subplot; the
        legacy 'See HV curve for ...' placeholder is removed."""
        out_base = self.tmp / "single_subplot"
        rc = _run([
            "--input-dir", str(self.tmp),
            "--output", str(out_base),
            "--figure-format", "png",
            "--log-level", "ERROR",
        ])
        self.assertEqual(rc, 0)
        png_path = out_base.with_suffix(".png")
        self.assertTrue(png_path.exists())
        # Inspect figure structure.
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
        img = mpimg.imread(png_path)
        # (n_obj=2 path produces a single HV-curve subplot; the legacy
        # placeholder would imply a 2-subplot layout. Assert that the
        # figure size is the single-subplot 8x5, not the 14x5 dual.)
        # The exact size is internal; what we really care about is
        # that the figure has at most one set of axes.
        fig = plt.imread(png_path)  # re-read to keep ref
        # Re-render the figure for a definitive axes count check:
        from plot_search_results import _plot_2obj  # type: ignore
        from pathlib import Path as _P
        from PIL import Image
        with Image.open(png_path) as im:
            w, h = im.size
        # Single-subplot 8x5: w/h ~ 1.6. Dual 14x5: w/h ~ 2.8.
        ratio = w / h
        self.assertLess(
            ratio, 2.0,
            f"expected single-subplot ratio (<2.0), got {ratio:.2f} (size {w}x{h}); "
            "the legacy 2-subplot layout may have been re-introduced.",
        )


class EndToEnd3ObjTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path("/tmp/test_plot_e2e_3obj")
        if self.tmp.exists():
            for f in self.tmp.iterdir():
                f.unlink()
        self.tmp.mkdir()
        _write_run(
            self.tmp / "mo_seed=0.json", "mo", 0,
            [(1.0, 5.0, 2.0), (2.0, 6.0, 3.0)],
            minimize=[True, False, False], n_obj=3,
        )
        _write_run(
            self.tmp / "mo_seed=1.json", "mo", 1,
            [(0.5, 5.5, 2.5), (1.5, 6.5, 3.5)],
            minimize=[True, False, False], n_obj=3,
        )

    def tearDown(self) -> None:
        for f in self.tmp.iterdir():
            f.unlink(missing_ok=True)
        self.tmp.rmdir()

    def test_3obj_graceful_no_raise(self) -> None:
        """n_obj>=3 must NOT raise in plot_search_results; it
        gracefully degrades to per-objective BSF curves."""
        out_base = self.tmp / "summary"
        rc = _run([
            "--input-dir", str(self.tmp),
            "--output", str(out_base),
            "--figure-format", "png",
            "--log-level", "ERROR",
        ])
        self.assertEqual(rc, 0)
        self.assertTrue(out_base.with_suffix(".png").exists())
        with out_base.with_suffix(".csv").open() as fh:
            reader = csv.reader(fh)
            header = next(reader)
            # Per-objective columns.
            self.assertIn("mo_obj0_mean", header)
            self.assertIn("mo_obj1_mean", header)
            self.assertIn("mo_obj2_mean", header)


if __name__ == "__main__":
    unittest.main()
