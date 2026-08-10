"""Tests for :mod:`tasks.small_molecule.core.scorer` (multi-obj extensions + ref registry)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tasks.small_molecule.core.scorer import (  # noqa: E402
    DEFAULT_REF,
    Scorer,
    Scorers,
    as_scorer_tuple,
    register_ref,
    resolve_ref_point,
)


def _scorer_a(smis):
    return [0.0 for _ in smis]


def _scorer_b(smis):
    return [1.0 for _ in smis]


def _scorer_c(smis):
    return [2.0 for _ in smis]


class AsScorerTupleTests(unittest.TestCase):
    def test_single_callable_wrapped(self) -> None:
        out = as_scorer_tuple(_scorer_a)
        self.assertEqual(out, (_scorer_a,))

    def test_tuple_passthrough(self) -> None:
        out = as_scorer_tuple((_scorer_a, _scorer_b))
        self.assertEqual(out, (_scorer_a, _scorer_b))

    def test_three_tuple(self) -> None:
        out = as_scorer_tuple((_scorer_a, _scorer_b, _scorer_c))
        self.assertEqual(len(out), 3)

    def test_non_callable_raises(self) -> None:
        with self.assertRaises(TypeError):
            as_scorer_tuple(42)  # type: ignore[arg-type]

    def test_tuple_with_non_callable_raises(self) -> None:
        with self.assertRaises(TypeError):
            as_scorer_tuple((_scorer_a, "not_callable"))  # type: ignore[arg-type]

    def test_empty_tuple_returns_empty(self) -> None:
        out = as_scorer_tuple(())
        self.assertEqual(out, ())

    def test_lambda_callable_accepted(self) -> None:
        s = lambda smis: [0.0 for _ in smis]
        out = as_scorer_tuple(s)
        self.assertEqual(out, (s,))


class DefaultRefRegistryTests(unittest.TestCase):
    def test_default_keys_present(self) -> None:
        self.assertIn("vina", DEFAULT_REF)
        self.assertIn("nn", DEFAULT_REF)
        self.assertIn("mock", DEFAULT_REF)

    def test_default_values_match_spec(self) -> None:
        self.assertEqual(DEFAULT_REF["vina"], 0.0)
        self.assertEqual(DEFAULT_REF["nn"], 5.0)
        self.assertEqual(DEFAULT_REF["mock"], 0.0)

    def test_register_ref_adds_entry(self) -> None:
        register_ref("my_oracle", 3.14)
        self.assertEqual(DEFAULT_REF.get("my_oracle"), 3.14)
        # Cleanup so other tests are not affected.
        DEFAULT_REF.pop("my_oracle", None)

    def test_register_ref_overwrites_existing(self) -> None:
        register_ref("mock", -1.5)
        self.assertEqual(DEFAULT_REF["mock"], -1.5)
        # Restore.
        DEFAULT_REF["mock"] = 0.0

    def test_register_ref_invalid_name_raises(self) -> None:
        with self.assertRaises(TypeError):
            register_ref("", 0.0)
        with self.assertRaises(TypeError):
            register_ref(42, 0.0)  # type: ignore[arg-type]

    def test_register_ref_invalid_default_raises(self) -> None:
        with self.assertRaises(TypeError):
            register_ref("ok", "zero")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            register_ref("ok", True)  # bool is excluded (rejected)


class ResolveRefPointTests(unittest.TestCase):
    def test_user_ref_overrides_registry(self) -> None:
        out = resolve_ref_point(("vina", "nn"), user_ref=(10.0, 20.0))
        self.assertEqual(out, (10.0, 20.0))

    def test_user_ref_partial_override_via_user(self) -> None:
        """user_ref entirely replaces registry lookups when given."""
        out = resolve_ref_point(("vina", "nn"), user_ref=(7.0, 8.0))
        # Even if registry had nn=5, user_ref=(7, 8) wins.
        self.assertEqual(out, (7.0, 8.0))

    def test_user_ref_none_uses_registry(self) -> None:
        out = resolve_ref_point(("vina", "nn"))
        self.assertEqual(out, (DEFAULT_REF["vina"], DEFAULT_REF["nn"]))

    def test_user_ref_none_unknown_name_falls_back_to_zero(self) -> None:
        out = resolve_ref_point(("unknown_backend", "nn"))
        self.assertEqual(out, (0.0, DEFAULT_REF["nn"]))

    def test_three_objectives_default(self) -> None:
        out = resolve_ref_point(("vina", "nn", "mock"))
        self.assertEqual(out, (0.0, 5.0, 0.0))

    def test_three_objectives_user_ref(self) -> None:
        out = resolve_ref_point(("vina", "nn", "mock"), user_ref=(1.0, 2.0, 3.0))
        self.assertEqual(out, (1.0, 2.0, 3.0))

    def test_user_ref_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_ref_point(("vina", "nn"), user_ref=(1.0,))
        with self.assertRaises(ValueError):
            resolve_ref_point(("vina", "nn"), user_ref=(1.0, 2.0, 3.0))

    def test_empty_objective_parts_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_ref_point(())

    def test_user_ref_string_coerced_to_float(self) -> None:
        out = resolve_ref_point(("vina", "nn"), user_ref=("1.5", "2.5"))
        self.assertEqual(out, (1.5, 2.5))
        self.assertIsInstance(out[0], float)

    def test_user_ref_int_coerced_to_float(self) -> None:
        out = resolve_ref_point(("vina", "nn"), user_ref=(1, 2))
        self.assertEqual(out, (1.0, 2.0))


class ScorerTypeAliasTests(unittest.TestCase):
    def test_scorer_alias_is_callable(self) -> None:
        """Type alias, not runtime-checkable; just ensure importable."""
        self.assertTrue(callable(Scorer))

    def test_scorers_alias_accepts_both(self) -> None:
        """Type alias accepts either form; runtime check via as_scorer_tuple."""
        self.assertEqual(len(as_scorer_tuple(_scorer_a)), 1)
        self.assertEqual(len(as_scorer_tuple((_scorer_a, _scorer_b))), 2)


class PackageSurfaceTests(unittest.TestCase):
    """Public surface invariants (stable, must not break)."""

    def test_default_ref_does_not_mutate_between_calls(self) -> None:
        """Sanity: a no-op call to register_ref for an unused name
        should not appear after a second resolve_ref_point call (the
        cleanup pattern in test_register_ref_adds_entry relies on this)."""
        register_ref("__test_marker__", 99.0)
        self.assertIn("__test_marker__", DEFAULT_REF)
        # Subsequent resolution sees it.
        out = resolve_ref_point(("__test_marker__",))
        self.assertEqual(out, (99.0,))
        DEFAULT_REF.pop("__test_marker__", None)
        # After cleanup, fallback applies.
        out = resolve_ref_point(("__test_marker__",))
        self.assertEqual(out, (0.0,))


if __name__ == "__main__":
    unittest.main()
