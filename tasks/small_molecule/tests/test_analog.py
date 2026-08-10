"""Tests for ``tasks.small_molecule.core.analog``.

Convention follows the rest of the suite: ``unittest`` + ``tempfile``, with
fake ReaSyn checkouts built per-test so the GPU / model code paths are
exercised without real checkpoints. Real-model smoke tests are guarded
behind ``@unittest.skipUnless`` so CI can stay offline.
"""

import logging
import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock

# Make sure project root is importable when tests are run directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tasks.small_molecule.core import analog  # noqa: E402
from tasks.small_molecule.core.analog import (  # noqa: E402
    ReasynConfig,
    _build_launcher_source,
    _canonicalize_dedup,
    _make_mols,
    _parse_model_paths,
    _query_visible_gpu_ids,
    _resolve_devices,
    _resolve_python_bin,
    _resolve_reasyn_repo,
)


FAKE_PARALLEL_SRC = textwrap.dedent(
    """
    \"\"\"Stand-in for reasyn.sampler.parallel used by tests.\"\"\"
    import os
    import sys
    import pandas as pd


    print(
        f"[fake_reasyn] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        file=sys.stderr,
    )


    def run_sampling_one(input, model_path, device, **kwargs):
        return pd.DataFrame(
            [
                {
                    "target": input.smiles,
                    "smiles": input.smiles + "_mod",
                    "score": 0.91,
                    "synthesis": "fake_rxn_single",
                    "num_steps": 1,
                    "time": 0.01,
                }
            ]
        )


    def run_parallel_sampling_return_smiles(input, model_path, **kwargs):
        rows = [
            {
                "target": m.smiles,
                "smiles": m.smiles + "_mod",
                "score": 0.9 + 0.01 * i,
                "synthesis": f"fake_rxn_{i}",
                "num_steps": 1,
                "time": 0.01,
            }
            for i, m in enumerate(input)
        ]
        return pd.DataFrame(rows)
    """
).lstrip()


# Self-contained stub for reasyn.chem.mol used inside the fake repo. The real
# mol.py transitively imports featurize/fpindex/matrix/..., which would drag
# the entire reasyn package into the test fixture. We only need enough surface
# for the auto-generated launcher to construct Molecule instances.
#
# When rdkit is available (in the ReaSyn venv subprocess) we use it for proper
# canonicalization. When rdkit is missing (in the calling test Python) we
# degrade to identity SMILES so canonicalize/dedup in the main process does
# not crash the test.
FAKE_MOL_SRC = textwrap.dedent(
    """
    \"\"\"Stand-in for reasyn.chem.mol used in tests.\"\"\"
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        _HAS_RDKIT = True
    except ImportError:
        _HAS_RDKIT = False


    class Molecule:
        def __init__(self, smiles):
            self._smiles = smiles
            self._rdmol = Chem.MolFromSmiles(smiles) if _HAS_RDKIT else None

        @property
        def smiles(self):
            return self._smiles

        @property
        def csmiles(self):
            if _HAS_RDKIT and self._rdmol is not None:
                return Chem.MolToSmiles(self._rdmol)
            return self._smiles

        @property
        def is_valid(self):
            if _HAS_RDKIT:
                return self._rdmol is not None
            return bool(self._smiles)


    class FingerprintOption:
        @staticmethod
        def morgan_for_tanimoto_similarity():
            return FingerprintOption()

        @staticmethod
        def rdkit():
            return FingerprintOption()
    """
).lstrip()


def _build_fake_reasyn_repo(real_reasyn: Path) -> Path:
    """Build a minimal ReaSyn checkout with stub ``parallel.py`` and stub ``chem/mol.py``.

    The real ``chem/mol.py`` transitively pulls in numpy/torch/rdkit and a
    half-dozen sibling ReaSyn modules. For the fake repo we ship a 30-line
    stub that still uses rdkit (so SMILES canonicalization behaves the same
    as the real implementation) but does not require numpy/torch/etc.
    """
    if not (real_reasyn / "reasyn" / "sampler" / "parallel.py").exists():
        raise unittest.SkipTest(f"Real ReaSyn not available at {real_reasyn}")

    tmp = Path(tempfile.mkdtemp(prefix="fake_reasyn_"))
    (tmp / "reasyn" / "chem").mkdir(parents=True)
    (tmp / "reasyn" / "sampler").mkdir(parents=True)
    (tmp / "data" / "trained_model").mkdir(parents=True)

    (tmp / "reasyn" / "chem" / "mol.py").write_text(FAKE_MOL_SRC, encoding="utf-8")
    (tmp / "reasyn" / "chem" / "__init__.py").write_text("")
    (tmp / "reasyn" / "sampler" / "parallel.py").write_text(
        FAKE_PARALLEL_SRC, encoding="utf-8"
    )
    (tmp / "reasyn" / "sampler" / "__init__.py").write_text("")
    (tmp / "reasyn" / "__init__.py").write_text("")
    (tmp / "data" / "trained_model" / "fake-ar.ckpt").write_text("fake")
    (tmp / "data" / "trained_model" / "fake-eb.ckpt").write_text("fake")
    return tmp


def _real_reasyn_path() -> Path:
    """Locate the real ReaSyn checkout via REASYN_HOME / REASYN_REPO / sibling convention."""
    for env_name in ("REASYN_HOME", "REASYN_REPO"):
        val = os.environ.get(env_name)
        if val:
            return Path(val).expanduser()
    # Sibling convention as a last resort (not used by analog.py itself).
    return (_PROJECT_ROOT.parents[1] / "ReaSyn").resolve()


def _reasyn_venv_python() -> Optional[str]:
    """Return ReaSyn venv's python if it exists, else None."""
    p = _real_reasyn_path() / ".venv" / "bin" / "python"
    return str(p) if p.exists() else None


def _has_rdkit() -> bool:
    try:
        import rdkit  # noqa: F401
        return True
    except ImportError:
        return False


def _fake_config(**overrides) -> ReasynConfig:
    base = dict(
        model_path=["/abs/ar.ckpt", "/abs/eb.ckpt"],
        reasyn_repo="/abs/ReaSyn",
        devices=[1, 2],
    )
    base.update(overrides)
    return ReasynConfig(**base)


# ---------------------------------------------------------------------------
# Class 1: _resolve_reasyn_repo
# ---------------------------------------------------------------------------


class ResolveReasynRepoTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env = os.environ.copy()
        for var in ("REASYN_HOME", "REASYN_REPO"):
            os.environ.pop(var, None)

    def tearDown(self) -> None:
        for var in ("REASYN_HOME", "REASYN_REPO"):
            if var in self._env:
                os.environ[var] = self._env[var]
            else:
                os.environ.pop(var, None)

    def _make_repo(self) -> Path:
        root = Path(tempfile.mkdtemp(prefix="reasyn_repo_"))
        (root / "reasyn" / "sampler").mkdir(parents=True)
        (root / "reasyn" / "sampler" / "parallel.py").write_text("")
        return root

    def test_explicit_arg_wins_over_env(self) -> None:
        explicit = self._make_repo()
        os.environ["REASYN_HOME"] = str(self._make_repo())
        resolved = _resolve_reasyn_repo(str(explicit))
        self.assertEqual(resolved, explicit.resolve())

    def test_uses_reasyn_home_env_when_no_explicit(self) -> None:
        env_repo = self._make_repo()
        os.environ["REASYN_HOME"] = str(env_repo)
        self.assertEqual(_resolve_reasyn_repo(None), env_repo.resolve())

    def test_falls_back_to_reasyn_repo_env(self) -> None:
        env_repo = self._make_repo()
        os.environ["REASYN_REPO"] = str(env_repo)
        self.assertEqual(_resolve_reasyn_repo(None), env_repo.resolve())

    def test_raises_when_no_candidate(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            _resolve_reasyn_repo(None)
        self.assertIn("Cannot locate ReaSyn repo", str(ctx.exception))

    def test_error_message_lists_all_tried_paths(self) -> None:
        bad1 = Path(tempfile.mkdtemp(prefix="bad_"))
        bad2 = Path(tempfile.mkdtemp(prefix="bad_"))
        with self.assertRaises(RuntimeError) as ctx:
            _resolve_reasyn_repo(str(bad1))
        msg = str(ctx.exception)
        self.assertIn(str(bad1), msg)
        os.environ["REASYN_HOME"] = str(bad2)
        with self.assertRaises(RuntimeError) as ctx:
            _resolve_reasyn_repo(None)
        msg = str(ctx.exception)
        self.assertIn(str(bad2), msg)


# ---------------------------------------------------------------------------
# Class 2: _resolve_python_bin
# ---------------------------------------------------------------------------


class ResolvePythonBinTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env = os.environ.copy()
        for var in ("REASYN_PYTHON", "REASYN_BIN"):
            os.environ.pop(var, None)
        self.tmp_repo = Path(tempfile.mkdtemp(prefix="repo_"))
        (self.tmp_repo / ".venv" / "bin").mkdir(parents=True)
        self.venv_python = self.tmp_repo / ".venv" / "bin" / "python"
        self.venv_python.write_text("")

    def tearDown(self) -> None:
        for var in ("REASYN_PYTHON", "REASYN_BIN"):
            if var in self._env:
                os.environ[var] = self._env[var]
            else:
                os.environ.pop(var, None)

    def test_explicit_arg_wins_over_env(self) -> None:
        os.environ["REASYN_PYTHON"] = str(self.venv_python)
        # _resolve_python_bin must NOT call .resolve() on the result, otherwise
        # symlinked venv interpreters lose their pyvenv.cfg context in
        # subprocess.run. The returned path should be the input verbatim (modulo
        # expanduser) so venv detection keeps working.
        self.assertEqual(
            _resolve_python_bin(sys.executable, self.tmp_repo),
            str(Path(sys.executable).expanduser().absolute()),
        )

    def test_explicit_arg_raises_if_missing(self) -> None:
        with self.assertRaises(RuntimeError):
            _resolve_python_bin("/nonexistent/python_xyz", self.tmp_repo)

    def test_explicit_arg_accepts_venv_bin_directory(self) -> None:
        self.assertEqual(
            _resolve_python_bin(str(self.venv_python.parent), self.tmp_repo),
            str(self.venv_python.absolute()),
        )

    def test_venv_bin_directory_falls_back_to_python3(self) -> None:
        self.venv_python.unlink()
        python3 = self.venv_python.parent / "python3"
        python3.write_text("")
        self.assertEqual(
            _resolve_python_bin(str(self.venv_python.parent), self.tmp_repo),
            str(python3.absolute()),
        )

    def test_missing_python_path_falls_back_to_versioned_python(self) -> None:
        self.venv_python.unlink()
        python310 = self.venv_python.parent / "python3.10"
        python310.write_text("")
        self.assertEqual(
            _resolve_python_bin(str(self.venv_python), self.tmp_repo),
            str(python310.absolute()),
        )

    def test_uses_reasyn_python_env(self) -> None:
        os.environ["REASYN_PYTHON"] = sys.executable
        self.assertEqual(
            _resolve_python_bin(None, self.tmp_repo),
            str(Path(sys.executable).expanduser().absolute()),
        )

    def test_falls_back_to_reasyn_bin_env(self) -> None:
        os.environ["REASYN_BIN"] = sys.executable
        self.assertEqual(
            _resolve_python_bin(None, self.tmp_repo),
            str(Path(sys.executable).expanduser().absolute()),
        )

    def test_reasyn_bin_env_accepts_venv_bin_directory(self) -> None:
        os.environ["REASYN_BIN"] = str(self.venv_python.parent)
        self.assertEqual(
            _resolve_python_bin(None, self.tmp_repo),
            str(self.venv_python.absolute()),
        )

    def test_directory_without_python_raises_clear_error(self) -> None:
        empty_bin = Path(tempfile.mkdtemp(prefix="empty_bin_"))
        with self.assertRaises(RuntimeError) as ctx:
            _resolve_python_bin(str(empty_bin), self.tmp_repo)
        self.assertIn("points to a directory", str(ctx.exception))
        self.assertIn("python3.*", str(ctx.exception))

    def test_falls_back_to_repo_venv_python(self) -> None:
        # No env vars, but a .venv/bin/python exists under repo.
        self.assertEqual(
            _resolve_python_bin(None, self.tmp_repo), str(self.venv_python.absolute())
        )

    def test_returns_none_when_nothing_available(self) -> None:
        bare_repo = Path(tempfile.mkdtemp(prefix="bare_repo_"))
        self.assertIsNone(_resolve_python_bin(None, bare_repo))


# ---------------------------------------------------------------------------
# Class 3: _parse_model_paths
# ---------------------------------------------------------------------------


class ParseModelPathsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = Path(tempfile.mkdtemp(prefix="repo_"))
        (self.repo / "data" / "trained_model").mkdir(parents=True)
        self.ar = self.repo / "data" / "trained_model" / "ar.ckpt"
        self.eb = self.repo / "data" / "trained_model" / "eb.ckpt"
        self.ar.write_text("")
        self.eb.write_text("")

    def test_accepts_csv_string(self) -> None:
        paths = _parse_model_paths(f"{self.ar},{self.eb}", self.repo)
        self.assertEqual(paths, [self.ar.resolve(), self.eb.resolve()])

    def test_accepts_list_of_strings(self) -> None:
        paths = _parse_model_paths([str(self.ar), str(self.eb)], self.repo)
        self.assertEqual(paths, [self.ar.resolve(), self.eb.resolve()])

    def test_resolves_relative_under_repo(self) -> None:
        paths = _parse_model_paths(
            "data/trained_model/ar.ckpt,data/trained_model/eb.ckpt", self.repo
        )
        self.assertEqual(paths, [self.ar.resolve(), self.eb.resolve()])

    def test_rejects_wrong_count(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            _parse_model_paths(f"{self.ar}", self.repo)
        self.assertIn("exactly 2 checkpoints", str(ctx.exception))
        third = self.repo / "third.ckpt"
        third.write_text("")
        with self.assertRaises(RuntimeError):
            _parse_model_paths(f"{self.ar},{self.eb},{third}", self.repo)

    def test_missing_ckpt_raises(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            _parse_model_paths("/nope/ar.ckpt,/nope/eb.ckpt", self.repo)
        self.assertIn("Missing files", str(ctx.exception))


# ---------------------------------------------------------------------------
# Class 4: _resolve_devices (CUDA constraint)
# ---------------------------------------------------------------------------


class ResolveDevicesTests(unittest.TestCase):
    def test_default_devices_is_one_and_two(self) -> None:
        cfg = ReasynConfig(model_path=["/a", "/b"])
        self.assertEqual(cfg.devices, [1, 2])

    def test_explicit_list_passes_through(self) -> None:
        with mock.patch.object(analog, "_query_visible_gpu_ids", return_value={0, 1, 2, 3}):
            self.assertEqual(_resolve_devices([1, 3]), [1, 3])

    def test_validates_against_nvidia_smi(self) -> None:
        with mock.patch.object(analog, "_query_visible_gpu_ids", return_value={0, 1, 2}):
            self.assertEqual(_resolve_devices([1, 2]), [1, 2])

    def test_raises_when_device_not_visible(self) -> None:
        with mock.patch.object(analog, "_query_visible_gpu_ids", return_value={0, 1}):
            with self.assertRaises(RuntimeError) as ctx:
                _resolve_devices([1, 2])
        self.assertIn("not visible", str(ctx.exception))
        self.assertIn("[0, 1]", str(ctx.exception))

    def test_empty_list_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            _resolve_devices([])

    def test_non_list_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            _resolve_devices(1)  # type: ignore[arg-type]

    def test_warns_when_nvidia_smi_unavailable(self) -> None:
        with mock.patch.object(analog, "_query_visible_gpu_ids", return_value=set()):
            with self.assertLogs("tasks.small_molecule.core.analog", level="WARNING") as ctx:
                self.assertEqual(_resolve_devices([1, 2]), [1, 2])
            self.assertTrue(any("nvidia-smi unavailable" in m for m in ctx.output))


# ---------------------------------------------------------------------------
# Class 5: input processing / post-processing
# ---------------------------------------------------------------------------


class InputProcessingTests(unittest.TestCase):
    def test_make_mols_skips_invalid_smiles(self) -> None:
        # Stub Molecule so this unit test does not need the real ReaSyn
        # checkout on sys.path.
        class _StubMol:
            def __init__(self, s: str) -> None:
                self.smiles = s
                self._valid = "INVALID" not in s

            @property
            def is_valid(self) -> bool:
                return self._valid

        with mock.patch.dict(sys.modules, {"reasyn": mock.MagicMock(), "reasyn.chem": mock.MagicMock(), "reasyn.chem.mol": mock.MagicMock(Molecule=_StubMol)}):
            captured: list[str] = []
            with mock.patch("sys.stderr.write", side_effect=captured.append):
                mols = _make_mols(["CCO", "INVALID@@@", "CCN"])
        self.assertEqual(len(mols), 2)
        joined = "".join(captured)
        self.assertIn("INVALID@@@", joined)

    def test_make_mols_empty_input_returns_empty_list(self) -> None:
        class _StubMol:
            def __init__(self, s: str) -> None:
                self.smiles = s

            @property
            def is_valid(self) -> bool:
                return True

        with mock.patch.dict(sys.modules, {"reasyn": mock.MagicMock(), "reasyn.chem": mock.MagicMock(), "reasyn.chem.mol": mock.MagicMock(Molecule=_StubMol)}):
            self.assertEqual(_make_mols([]), [])

    def test_canonicalize_dedup_drops_duplicate_csmiles(self) -> None:
        # Stub Molecule.csmiles to make "OCC" and "CCO" collide.
        class _StubMol:
            def __init__(self, s: str) -> None:
                self._raw = s

            @property
            def csmiles(self) -> str:
                return "CCO" if "C" in self._raw and "O" in self._raw else self._raw

        import pandas as pd

        df = pd.DataFrame(
            [
                {"target": "CCO", "smiles": "OCC", "score": 0.5,
                 "synthesis": "a", "num_steps": 1},
                {"target": "CCO", "smiles": "CCO", "score": 0.9,
                 "synthesis": "b", "num_steps": 1},
            ]
        )
        with mock.patch.dict(sys.modules, {"reasyn": mock.MagicMock(), "reasyn.chem": mock.MagicMock(), "reasyn.chem.mol": mock.MagicMock(Molecule=_StubMol)}):
            out = _canonicalize_dedup(df)
        self.assertEqual(len(out), 1)
        # Both rows collapse to (target="CCO", smiles="CCO") and drop_duplicates
        # keeps the first occurrence.
        self.assertEqual(out.iloc[0]["smiles"], "CCO")
        self.assertAlmostEqual(out.iloc[0]["score"], 0.5)

    def test_canonicalize_dedup_sorts_by_score_descending(self) -> None:
        class _StubMol:
            def __init__(self, s: str) -> None:
                pass

            @property
            def csmiles(self) -> str:
                return "x"

        import pandas as pd

        df = pd.DataFrame(
            [
                {"target": "CCN", "smiles": "CCN", "score": 0.3,
                 "synthesis": "x", "num_steps": 1},
                {"target": "CCN", "smiles": "NCC", "score": 0.8,
                 "synthesis": "y", "num_steps": 1},
                {"target": "CCN", "smiles": "CNC", "score": 0.5,
                 "synthesis": "z", "num_steps": 1},
            ]
        )
        with mock.patch.dict(sys.modules, {"reasyn": mock.MagicMock(), "reasyn.chem": mock.MagicMock(), "reasyn.chem.mol": mock.MagicMock(Molecule=_StubMol)}):
            out = _canonicalize_dedup(df)
        scores = out["score"].tolist()
        self.assertEqual(scores, sorted(scores, reverse=True))


# ---------------------------------------------------------------------------
# Class 6: _build_launcher_source
# ---------------------------------------------------------------------------


class BuildLauncherSourceTests(unittest.TestCase):
    def _kwargs(self, **overrides):
        base = dict(
            python_bin="/abs/python",
            reasyn_repo="/abs/ReaSyn",
            ar_ckpt="/abs/ar.ckpt",
            eb_ckpt="/abs/eb.ckpt",
            in_path="/tmp/in.smi",
            out_path="/tmp/out.pkl",
            devices=[1, 2],
            cuda_visible_devices="1,2",
            search_width=12,
            exhaustiveness=128,
            num_cycles=12,
            num_editflow_samples=100,
            num_editflow_steps=100,
            time_limit=10000,
            num_workers_per_gpu=8,
            task_qsize=0,
            result_qsize=0,
            filter_sim=0.8,
            add_bb_path=None,
            mols_to_filter=None,
            no_exact_break=True,
        )
        base.update(overrides)
        return base

    def test_launcher_is_valid_python(self) -> None:
        src = _build_launcher_source(**self._kwargs())
        compile(src, "<launcher>", "exec")  # raises SyntaxError if invalid

    def test_launcher_omits_exact_break_for_inmemory_api(self) -> None:
        # ReaSyn's run_sampling_one / run_parallel_sampling_return_smiles
        # hardcode exact_break=True; the in-memory API cannot honor the
        # --no_exact_break CLI flag. Verify the launcher never passes it.
        for flag in (True, False):
            src = _build_launcher_source(**self._kwargs(no_exact_break=flag))
            self.assertNotIn("exact_break", src)

    def test_launcher_omits_none_optional_args(self) -> None:
        src = _build_launcher_source(**self._kwargs(add_bb_path=None, mols_to_filter=None))
        self.assertNotIn("add_bb_path", src.split("run_sampling_one")[0].split("run_parallel")[0])

    def test_launcher_includes_optional_when_provided(self) -> None:
        src = _build_launcher_source(
            **self._kwargs(add_bb_path="/abs/bb.pkl", mols_to_filter="/abs/filter.smi")
        )
        self.assertIn("/abs/bb.pkl", src)
        self.assertIn("/abs/filter.smi", src)

    def test_launcher_uses_single_mode_for_one_input(self) -> None:
        # We can't directly observe runtime branching, but we can validate
        # that the source constructs exactly one `run_sampling_one` call when
        # only one molecule is expected. The branching is in the source code.
        src = _build_launcher_source(**self._kwargs())
        self.assertIn("run_sampling_one", src)
        self.assertIn("run_parallel_sampling_return_smiles", src)
        # num_gpus passed to parallel only.
        self.assertIn("num_gpus=2", src)
        # device="cuda:0" used for single mode.
        self.assertIn('device="cuda:0"', src)

    def test_launcher_sets_cuda_visible_devices_first(self) -> None:
        src = _build_launcher_source(**self._kwargs())
        env_line = 'os.environ["CUDA_VISIBLE_DEVICES"]'
        import_line = "from reasyn.chem.mol import Molecule"
        self.assertLess(src.index(env_line), src.index(import_line))


# ---------------------------------------------------------------------------
# Class 7: End-to-end subprocess mode (fake ReaSyn)
# ---------------------------------------------------------------------------


class EndToEndSubprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.venv_python = _reasyn_venv_python()
        if cls.venv_python is None:
            raise unittest.SkipTest(
                "ReaSyn venv python not found; cannot run subprocess tests"
            )

    def setUp(self) -> None:
        self.real_reasyn = _real_reasyn_path()
        self.fake_repo = _build_fake_reasyn_repo(self.real_reasyn)
        self.fake_models = [
            self.fake_repo / "data" / "trained_model" / "fake-ar.ckpt",
            self.fake_repo / "data" / "trained_model" / "fake-eb.ckpt",
        ]
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="reasyn_e2e_"))

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.fake_repo, ignore_errors=True)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _config(self, **overrides) -> ReasynConfig:
        kwargs = dict(
            model_path=[str(self.fake_models[0]), str(self.fake_models[1])],
            reasyn_repo=str(self.fake_repo),
            python_bin=self.venv_python,
            devices=[1, 2],
            temp_dir=str(self.tmp_dir),
            num_cycles=1,
            num_editflow_samples=1,
            num_editflow_steps=1,
            time_limit=30,
        )
        kwargs.update(overrides)
        return ReasynConfig(**kwargs)

    def test_subprocess_returns_dataframe_with_expected_columns(self) -> None:
        from tasks.small_molecule.core.analog import generate_analogs

        df = generate_analogs(["CCO", "CCN"], self._config())
        expected = {"target", "smiles", "score", "synthesis", "num_steps"}
        self.assertTrue(expected.issubset(df.columns))
        self.assertEqual(len(df), 2)

    def test_subprocess_cleans_tempfiles_after_call(self) -> None:
        from tasks.small_molecule.core.analog import generate_analogs

        generate_analogs(["CCO"], self._config())
        leftovers = sorted(p.name for p in self.tmp_dir.iterdir())
        self.assertEqual(leftovers, [], f"tempfiles leaked: {leftovers}")

    def test_subprocess_sets_cuda_visible_devices_env(self) -> None:
        captured: dict = {}

        real_popen = subprocess.Popen

        def fake_popen(cmd, *args, **kwargs):
            captured["cmd"] = cmd
            captured["env"] = kwargs.get("env")
            # Run the real launcher to keep behavior consistent.
            return real_popen(cmd, *args, **kwargs)

        with mock.patch("tasks.small_molecule.core.analog.subprocess.Popen", side_effect=fake_popen):
            from tasks.small_molecule.core.analog import generate_analogs

            generate_analogs(["CCO"], self._config())
        self.assertEqual(captured["env"]["CUDA_VISIBLE_DEVICES"], "1,2")

    def test_subprocess_uses_requested_python_bin(self) -> None:
        captured: dict = {}

        real_popen = subprocess.Popen

        def fake_popen(cmd, *args, **kwargs):
            captured["cmd"] = cmd
            return real_popen(cmd, *args, **kwargs)

        with mock.patch("tasks.small_molecule.core.analog.subprocess.Popen", side_effect=fake_popen):
            from tasks.small_molecule.core.analog import generate_analogs

            generate_analogs(["CCO"], self._config())
        # argv[0] should be the venv python we passed (NOT the test runner's
        # sys.executable).
        self.assertEqual(captured["cmd"][0], self.venv_python)

    def test_subprocess_honors_temp_dir_override(self) -> None:
        # Run with a custom temp_dir; ensure tempfiles land there.
        custom_dir = Path(tempfile.mkdtemp(prefix="custom_tmp_"))
        try:
            from tasks.small_molecule.core.analog import generate_analogs

            generate_analogs(["CCO"], self._config(temp_dir=str(custom_dir)))
            self.assertEqual(list(custom_dir.iterdir()), [])
        finally:
            import shutil

            shutil.rmtree(custom_dir, ignore_errors=True)

    def test_subprocess_string_input_wrapped_to_list(self) -> None:
        from tasks.small_molecule.core.analog import generate_analogs

        df = generate_analogs("CCO", self._config())
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["target"], "CCO")


# ---------------------------------------------------------------------------
# Class 8: End-to-end in-process mode
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    _has_rdkit(),
    "rdkit not installed in calling Python; in-process mode requires it",
)
class EndToEndInProcessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.real_reasyn = _real_reasyn_path()
        self.fake_repo = _build_fake_reasyn_repo(self.real_reasyn)
        self.fake_models = [
            self.fake_repo / "data" / "trained_model" / "fake-ar.ckpt",
            self.fake_repo / "data" / "trained_model" / "fake-eb.ckpt",
        ]

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.fake_repo, ignore_errors=True)

    def _config(self, **overrides) -> ReasynConfig:
        kwargs = dict(
            model_path=[str(self.fake_models[0]), str(self.fake_models[1])],
            reasyn_repo=str(self.fake_repo),
            python_bin=None,    # -> sys.executable -> in-process
            devices=[1, 2],
            num_cycles=1,
            num_editflow_samples=1,
            num_editflow_steps=1,
            time_limit=30,
        )
        kwargs.update(overrides)
        return ReasynConfig(**kwargs)

    def test_inprocess_mode_works_when_python_matches(self) -> None:
        from tasks.small_molecule.core.analog import generate_analogs

        df = generate_analogs(["CCO", "CCN"], self._config())
        self.assertEqual(len(df), 2)
        self.assertEqual(set(df["target"].tolist()), {"CCO", "CCN"})

    def test_inprocess_sets_cuda_visible_devices_before_calling_reasyn(self) -> None:
        from tasks.small_molecule.core.analog import generate_analogs

        with mock.patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=False
        ):
            generate_analogs(["CCO"], self._config(devices=[1]))
            self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "1")
        # Restore for subsequent tests.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def test_canonicalize_applies_when_enabled(self) -> None:
        from tasks.small_molecule.core.analog import generate_analogs

        df = generate_analogs(["CCO", "CCN"], self._config(canonicalize=True))
        # Each target should appear once (fake parallel returns one row per input).
        targets = df["target"].tolist()
        self.assertEqual(sorted(targets), ["CCN", "CCO"])


# ---------------------------------------------------------------------------
# Class 9: error handling
# ---------------------------------------------------------------------------


class ErrorHandlingTests(unittest.TestCase):
    def test_raises_when_model_checkpoint_missing(self) -> None:
        with mock.patch.object(
            analog,
            "_resolve_reasyn_repo",
            return_value=Path(tempfile.mkdtemp(prefix="fake_repo_")),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                ReasynConfig(model_path=["/nope/ar.ckpt", "/nope/eb.ckpt"])
                # Config construction itself does not validate paths; validation
                # happens inside generate_analogs / _parse_model_paths.
                from tasks.small_molecule.core.analog import generate_analogs

                generate_analogs("CCO", ReasynConfig(model_path=["/nope/a.ckpt", "/nope/b.ckpt"]))
        self.assertIn("Missing files", str(ctx.exception))

    def test_raises_when_only_one_checkpoint(self) -> None:
        real_reasyn = _real_reasyn_path()
        fake_repo = _build_fake_reasyn_repo(real_reasyn)
        try:
            from tasks.small_molecule.core.analog import generate_analogs

            with self.assertRaises(RuntimeError) as ctx:
                generate_analogs(
                    "CCO",
                    ReasynConfig(
                        model_path=[str(fake_repo / "data" / "trained_model" / "fake-ar.ckpt")],
                        reasyn_repo=str(fake_repo),
                        python_bin=None,
                        devices=[1, 2],
                    ),
                )
            self.assertIn("exactly 2 checkpoints", str(ctx.exception))
        finally:
            import shutil

            shutil.rmtree(fake_repo, ignore_errors=True)

    def test_subprocess_propagates_stderr_on_failure(self) -> None:
        venv_python = _reasyn_venv_python()
        if venv_python is None:
            self.skipTest("ReaSyn venv python not available")
        # Build a fake launcher that fails fast. Also provide a minimal
        # chem/mol.py so the auto-generated launcher's import of Molecule
        # succeeds before the failure is triggered.
        fake_repo = Path(tempfile.mkdtemp(prefix="fake_repo_fail_"))
        (fake_repo / "reasyn" / "chem").mkdir(parents=True)
        (fake_repo / "reasyn" / "sampler").mkdir(parents=True)
        (fake_repo / "reasyn" / "chem" / "__init__.py").write_text("")
        (fake_repo / "reasyn" / "sampler" / "__init__.py").write_text("")
        (fake_repo / "reasyn" / "__init__.py").write_text("")
        (fake_repo / "reasyn" / "chem" / "mol.py").write_text(
            "class Molecule:\n    def __init__(self, s): self.smiles = s\n",
            encoding="utf-8",
        )
        (fake_repo / "reasyn" / "sampler" / "parallel.py").write_text(
            textwrap.dedent(
                """
                import sys
                def run_sampling_one(input, model_path, device, **kwargs):
                    sys.stderr.write('boom from fake')
                    sys.exit(2)
                def run_parallel_sampling_return_smiles(input, model_path, **kwargs):
                    sys.stderr.write('boom from fake')
                    sys.exit(2)
                """
            ),
            encoding="utf-8",
        )
        (fake_repo / "data" / "trained_model").mkdir(parents=True)
        ar = fake_repo / "data" / "trained_model" / "ar.ckpt"
        eb = fake_repo / "data" / "trained_model" / "eb.ckpt"
        ar.write_text("")
        eb.write_text("")
        try:
            from tasks.small_molecule.core.analog import generate_analogs

            with self.assertRaises(RuntimeError) as ctx:
                generate_analogs(
                    "CCO",
                    ReasynConfig(
                        model_path=[str(ar), str(eb)],
                        reasyn_repo=str(fake_repo),
                        python_bin=venv_python,
                        devices=[1, 2],
                        time_limit=10,
                    ),
                )
            self.assertIn("boom from fake", str(ctx.exception))
        finally:
            import shutil

            shutil.rmtree(fake_repo, ignore_errors=True)

    def test_subprocess_timeout_kills_process_group(self) -> None:
        fake_repo = Path(tempfile.mkdtemp(prefix="fake_repo_timeout_"))
        (fake_repo / "reasyn" / "sampler").mkdir(parents=True)
        (fake_repo / "reasyn" / "sampler" / "parallel.py").write_text("")
        ar = fake_repo / "ar.ckpt"
        eb = fake_repo / "eb.ckpt"
        ar.write_text("")
        eb.write_text("")

        class FakeProcess:
            pid = 12345
            returncode = None

            def communicate(self, timeout=None):
                raise subprocess.TimeoutExpired(["python", "launcher.py"], timeout)

            def kill(self):
                self.returncode = -9

            def wait(self, timeout=None):
                self.returncode = -9
                return self.returncode

        fake_proc = FakeProcess()
        try:
            with mock.patch.object(analog.os, "name", "posix"):
                with mock.patch.object(analog.subprocess, "Popen", return_value=fake_proc):
                    with mock.patch.object(analog.os, "killpg", create=True) as killpg:
                        with self.assertRaises(RuntimeError) as ctx:
                            analog._run_via_subprocess(
                                ["CCO"],
                                [ar, eb],
                                fake_repo,
                                sys.executable,
                                ReasynConfig(
                                    model_path=[str(ar), str(eb)],
                                    reasyn_repo=str(fake_repo),
                                    python_bin=sys.executable,
                                    devices=[0],
                                    time_limit=1,
                                ),
                            )
            killpg.assert_called_once()
            self.assertIn("timed out", str(ctx.exception))
        finally:
            import shutil

            shutil.rmtree(fake_repo, ignore_errors=True)


# ---------------------------------------------------------------------------
# Class 10: real-model smoke test (default skip)
# ---------------------------------------------------------------------------


_REAL_REASYN = _real_reasyn_path()
_REAL_AR = _REAL_REASYN / "data" / "trained_model" / "nv-reasyn-ar-166m-v2.ckpt"
_REAL_EB = _REAL_REASYN / "data" / "trained_model" / "nv-reasyn-eb-174m-v2.ckpt"


def _venv_can_import_reasyn() -> bool:
    """Best-effort check that the ReaSyn venv can import the heavy modules."""
    py = _reasyn_venv_python()
    if not py:
        return False
    try:
        result = subprocess.run(
            [py, "-c", "import rotary_embedding_torch, reasyn.sampler.parallel"],
            capture_output=True, text=True, timeout=60,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _venv_can_load_fpindex() -> bool:
    """Check that fpindex.pkl unpickles cleanly under the venv's sklearn.

    Includes the same ``ManhattanDistance64`` alias the production launcher
    uses, so the probe mirrors actual usage.
    """
    py = _reasyn_venv_python()
    fpindex = _REAL_REASYN / "data" / "processed" / "comp_2048" / "fpindex.pkl"
    if not py or not fpindex.exists():
        return False
    probe = (
        "import pickle, sklearn.metrics._dist_metrics as m\n"
        "if not hasattr(m, 'ManhattanDistance64'):\n"
        "    m.ManhattanDistance64 = m.ManhattanDistance\n"
        f"pickle.load(open({str(fpindex)!r}, 'rb'))\n"
    )
    try:
        result = subprocess.run(
            [py, "-c", probe],
            capture_output=True, text=True, timeout=60,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@unittest.skipUnless(
    _REAL_AR.exists()
    and _REAL_EB.exists()
    and _venv_can_import_reasyn()
    and _venv_can_load_fpindex(),
    "Real ReaSyn checkpoints or venv dependencies not available",
)
class RealReasynIntegrationTests(unittest.TestCase):
    def test_smoke_single_smiles_subprocess(self) -> None:
        # Real ReaSyn needs its own venv python (with rdkit/torch/etc.),
        # so always go via subprocess here.
        venv_python = _reasyn_venv_python()
        if venv_python is None:
            self.skipTest("ReaSyn venv python not available")
        from tasks.small_molecule.core.analog import generate_analogs

        df = generate_analogs(
            "CCO",
            ReasynConfig(
                model_path=[str(_REAL_AR), str(_REAL_EB)],
                reasyn_repo=str(_REAL_REASYN),
                python_bin=venv_python,
                devices=[1, 2],
                num_cycles=1,
                num_editflow_samples=4,
                num_editflow_steps=4,
                time_limit=30,
            ),
        )
        self.assertGreater(len(df), 0)
        self.assertIn("smiles", df.columns)
        self.assertIn("target", df.columns)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
