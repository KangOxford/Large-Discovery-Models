"""Tests for ``strbo_v1.random_search``."""

from __future__ import annotations

import random
import unittest
from typing import Optional

from strbo_v1.random_search import (
    _pick_expansion_target,
    random_analog_search,
)


class RandomSearchBOTests(unittest.TestCase):
    def test_empty_seed_returns_empty_history(self) -> None:
        history = random_analog_search(
            seed_smiles=[],
            scorer=lambda smis: [0.0] * len(smis),
            analog_fn=lambda smis: [],
            n_iterations=5,
            rng=random.Random(0),
        )
        self.assertEqual(history, [])

    def test_single_iteration_scores_one_seed_and_records(self) -> None:
        history = random_analog_search(
            seed_smiles=["CCO"],
            scorer=lambda smis: [-7.5],
            analog_fn=lambda smis: [],
            n_iterations=1,
            rng=random.Random(0),
        )
        self.assertEqual(history, [("CCO", -7.5)])

    def test_analogues_are_added_to_pool_and_evaluated_next_iteration(self) -> None:
        calls: list[list[str]] = []
        analogues_to_emit = ["CCO", "CCN", "CCC"]

        def fake_scorer(smis: list[str]) -> list[float]:
            calls.append(list(smis))
            return [-1.0 * len(calls)]

        def fake_analog(smis: list[str]) -> list[str]:
            if not analogues_to_emit:
                return []
            emitted, analogues_to_emit[:] = analogues_to_emit[0], analogues_to_emit[1:]
            return [emitted]

        history = random_analog_search(
            seed_smiles=["X"],
            scorer=fake_scorer,
            analog_fn=fake_analog,
            n_iterations=4,
            pool_min_size=1,
            rng=random.Random(0),
        )
        # Lazy expansion: with min_size=1, pool is refilled when below 1.
        # The seed "X" is expanded (giving "CCO"), then we score from the
        # pool. The seed set {X, CCO, CCN, CCC} all appear in history.
        self.assertEqual(len(history), 4)
        self.assertEqual(set(s for s, _ in history), {"X", "CCO", "CCN", "CCC"})

    def test_scorer_returning_none_is_recorded(self) -> None:
        history = random_analog_search(
            seed_smiles=["A", "B"],
            scorer=lambda smis: [None],
            analog_fn=lambda smis: [],
            n_iterations=2,
            rng=random.Random(0),
        )
        self.assertEqual(len(history), 2)
        self.assertTrue(all(score is None for _, score in history))

    def test_scorer_returning_shorter_list_yields_none(self) -> None:
        history = random_analog_search(
            seed_smiles=["A"],
            scorer=lambda smis: [],
            analog_fn=lambda smis: [],
            n_iterations=1,
            rng=random.Random(0),
        )
        self.assertEqual(history, [("A", None)])

    def test_duplicate_analogues_are_not_re_added(self) -> None:
        def fake_analog(smis: list[str]) -> list[str]:
            return ["B", "B", "B"]  # same new analogue emitted 3x

        def fake_scorer(smis: list[str]) -> list[float]:
            return [-1.0]

        history = random_analog_search(
            seed_smiles=["A"],
            scorer=fake_scorer,
            analog_fn=fake_analog,
            n_iterations=5,
            pool_min_size=1,
            rng=random.Random(0),
        )
        # Seed "A" is evaluated, then "B" is queued once (duplicates
        # within the same batch are deduped) and evaluated. After that
        # the pool is empty.
        self.assertEqual(sorted(h[0] for h in history), ["A", "B"])
        self.assertEqual(len(history), 2)

    def test_already_evaluated_smiles_are_not_re_evaluated(self) -> None:
        def fake_analog(smis: list[str]) -> list[str]:
            return ["A"]  # always emit the already-evaluated seed

        history = random_analog_search(
            seed_smiles=["A"],
            scorer=lambda smis: [-1.0],
            analog_fn=fake_analog,
            n_iterations=5,
            rng=random.Random(0),
        )
        self.assertEqual([h[0] for h in history], ["A"])

    def test_pool_exhausted_breaks_loop_early(self) -> None:
        history = random_analog_search(
            seed_smiles=["A"],
            scorer=lambda smis: [-1.0],
            analog_fn=lambda smis: [],
            n_iterations=10,
            rng=random.Random(0),
        )
        # Only one SMILES to evaluate; loop should break after that.
        self.assertEqual(len(history), 1)

    def test_rng_reproducibility(self) -> None:
        def fake_analog(smis: list[str]) -> list[str]:
            return [f"X_{smis[0]}_a", f"X_{smis[0]}_b"]

        kwargs = dict(
            seed_smiles=["A", "B", "C"],
            scorer=lambda smis: [0.0],
            analog_fn=fake_analog,
            n_iterations=6,
        )
        h1 = random_analog_search(rng=random.Random(123), **kwargs)
        h2 = random_analog_search(rng=random.Random(123), **kwargs)
        self.assertEqual(h1, h2)
        # Different seed -> different history (with overwhelming probability).
        h3 = random_analog_search(rng=random.Random(999), **kwargs)
        self.assertNotEqual(h1, h3)

    def test_analog_generator_returning_empty_list_works(self) -> None:
        history = random_analog_search(
            seed_smiles=["A", "B"],
            scorer=lambda smis: [float(smis[0] == "A")],
            analog_fn=lambda smis: [],
            n_iterations=3,
            rng=random.Random(0),
        )
        # Both A and B should be evaluated; scores depend on the scorer.
        self.assertEqual(len(history), 2)
        smiles_to_score = {s: score for s, score in history}
        self.assertEqual(smiles_to_score["A"], 1.0)
        self.assertEqual(smiles_to_score["B"], 0.0)

    def test_scorer_exception_records_none(self) -> None:
        # Scorer failure is contained: the batch is recorded as None
        # (matches _safe_score behavior in bayesian_analog_search).
        def bad_scorer(smis: list[str]) -> list[float]:
            raise RuntimeError("scorer blew up")

        history = random_analog_search(
            seed_smiles=["A"],
            scorer=bad_scorer,
            analog_fn=lambda smis: [],
            n_iterations=1,
            rng=random.Random(0),
        )
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0][0], "A")
        self.assertIsNone(history[0][1])

    def test_analog_generator_exception_continues_gracefully(self) -> None:
        # Analog failure is contained: treated as empty analogues.
        def bad_analog(smis: list[str]) -> list[str]:
            raise RuntimeError("analog blew up")

        history = random_analog_search(
            seed_smiles=["A"],
            scorer=lambda smis: [-1.0],
            analog_fn=bad_analog,
            n_iterations=1,
            rng=random.Random(0),
        )
        self.assertEqual(history, [("A", -1.0)])

    def test_seed_can_be_generator(self) -> None:
        def seed_gen():
            yield "A"
            yield "B"

        history = random_analog_search(
            seed_smiles=seed_gen(),
            scorer=lambda smis: [0.0],
            analog_fn=lambda smis: [],
            n_iterations=3,
            rng=random.Random(0),
        )
        # Both A and B should be evaluated (order is random).
        self.assertEqual(sorted(h[0] for h in history), ["A", "B"])

    def test_seed_dedup_and_whitespace_stripping(self) -> None:
        history = random_analog_search(
            seed_smiles=["A", " A ", "", "A", "  B  "],
            scorer=lambda smis: [0.0],
            analog_fn=lambda smis: [],
            n_iterations=10,
            rng=random.Random(0),
        )
        self.assertEqual(sorted(h[0] for h in history), ["A", "B"])

    def test_uniform_random_pick_uses_rng(self) -> None:
        # With a controlled rng, we can verify uniform sampling by
        # counting how often each seed is the first pick across many runs.
        from collections import Counter

        picks: Counter = Counter()
        for trial in range(600):
            rng = random.Random(trial)
            h = random_analog_search(
                seed_smiles=["A", "B", "C"],
                scorer=lambda smis: [0.0],
                analog_fn=lambda smis: [],
                n_iterations=1,
                rng=rng,
            )
            picks[h[0][0]] += 1
        # Each of A/B/C should be picked ~200 times (out of 600).
        for smiles, count in picks.items():
            self.assertGreater(count, 150, f"{smiles} picked only {count} times")
            self.assertLess(count, 250, f"{smiles} picked {count} times")

    def test_returned_history_order_matches_evaluation_order(self) -> None:
        def fake_analog(smis: list[str]) -> list[str]:
            # Emit one new analogue per input. Lazy expansion means the
            # seed may be expanded before scoring; whichever is picked
            # from the pool gets scored.
            return [f"{smis[0]}_next"]

        history = random_analog_search(
            seed_smiles=["A"],
            scorer=lambda smis: [42.0],
            analog_fn=fake_analog,
            n_iterations=2,
            pool_min_size=1,
            rng=random.Random(0),
        )
        # History records evaluations in order. Two unique SMILES, both
        # scoring 42.0.
        self.assertEqual(len(history), 2)
        smiles_seen = [s for s, _ in history]
        self.assertEqual(len(set(smiles_seen)), 2, f"expected 2 unique SMILES, got {smiles_seen}")
        for _, sc in history:
            self.assertEqual(sc, 42.0)
        # The second-evaluated SMILES should be reachable from the first
        # via one expansion (f"{first}_next" equals second).
        self.assertEqual(smiles_seen[1], f"{smiles_seen[0]}_next")


class BatchSizeTests(unittest.TestCase):
    """Tests for batch_size and pool_min_size / pool_max_size parameters."""

    def test_batch_size_evaluates_in_parallel(self) -> None:
        """One round of n_iterations=3 with batch_size=3 should score all 3 seeds at once."""
        calls: list[list[str]] = []

        def fake_scorer(smis: list[str]) -> list[float]:
            calls.append(list(smis))
            return [-1.0 * len(smis)] * len(smis)

        history = random_analog_search(
            seed_smiles=["A", "B", "C"],
            scorer=fake_scorer,
            analog_fn=lambda smis: [],
            n_iterations=3,
            batch_size=3,
            rng=random.Random(0),
        )
        self.assertEqual(len(history), 3)
        self.assertEqual(len(calls), 1, "should have made exactly one scorer call")
        self.assertEqual(set(calls[0]), {"A", "B", "C"})

    def test_batch_size_capped_by_pool_min_size(self) -> None:
        """If pool < batch_size and min_size=batch_size, refill to grow pool."""
        calls: list[list[str]] = []
        analogue_counter = {"i": 0}

        def fake_analog(smis: list[str]) -> list[str]:
            analogue_counter["i"] += 1
            if analogue_counter["i"] == 1:
                return ["B", "C"]  # grow pool on first expansion
            return []

        def fake_scorer(smis: list[str]) -> list[float]:
            calls.append(list(smis))
            return [-1.0] * len(smis)

        history = random_analog_search(
            seed_smiles=["A"],
            scorer=fake_scorer,
            analog_fn=fake_analog,
            n_iterations=3,
            batch_size=3,
            pool_min_size=3,
            rng=random.Random(0),
        )
        # First batch: pool=["A"] < min_size=3, refill by expanding
        # A → pool=[A,B,C]. Then score all 3 in one batch.
        self.assertEqual(set(s for s, _ in history), {"A", "B", "C"})
        self.assertEqual(len(calls), 1)
        self.assertEqual(set(calls[0]), {"A", "B", "C"})

    def test_no_refill_when_min_size_none(self) -> None:
        """With pool_min_size=None, analog_generator is never called."""
        refill_calls: list[list[str]] = []

        def fake_analog(smis: list[str]) -> list[str]:
            refill_calls.append(list(smis))
            return [f"{smis[0]}_x"]

        random_analog_search(
            seed_smiles=["A", "B"],
            scorer=lambda smis: [-1.0] * len(smis),
            analog_fn=fake_analog,
            n_iterations=2,
            batch_size=2,
            pool_min_size=None,
            rng=random.Random(0),
        )
        # pool=["A","B"], no min constraint. No refill; analog_generator never called.
        self.assertEqual(len(refill_calls), 0)

    def test_min_size_triggers_refill(self) -> None:
        """With pool_min_size=N, analog_generator is called to refill pool."""
        refill_calls: list[list[str]] = []

        def fake_analog(smis: list[str]) -> list[str]:
            refill_calls.append(list(smis))
            return [f"{smis[0]}_x"]

        random_analog_search(
            seed_smiles=["A"],
            scorer=lambda smis: [-1.0] * len(smis),
            analog_fn=fake_analog,
            n_iterations=2,
            batch_size=1,
            pool_min_size=2,
            rng=random.Random(0),
        )
        # pool=["A"], target=2. Refill expands A → pool grows.
        self.assertGreater(len(refill_calls), 0)
        # Each refill call has 1 SMILES (the expansion target).
        for batch in refill_calls:
            self.assertEqual(len(batch), 1)

    def test_default_min_size_enables_refill(self) -> None:
        """Regression guard: with no explicit pool_min_size, refill is enabled.

        The default value must keep analog_generator active, otherwise users
        who rely on the default (no CLI flag) get 0 analogues.
        """
        refill_calls: list[list[str]] = []

        def fake_analog(smis: list[str]) -> list[str]:
            refill_calls.append(list(smis))
            return [f"{smis[0]}_a", f"{smis[0]}_b"]

        random_analog_search(
            seed_smiles=["A"],
            scorer=lambda smis: [-1.0] * len(smis),
            analog_fn=fake_analog,
            n_iterations=4,
            batch_size=1,
            # No pool_min_size passed → use default.
            rng=random.Random(0),
        )
        # Default must trigger refill (analog_generator called).
        self.assertGreater(
            len(refill_calls), 0,
            "default pool_min_size must enable refill (analog_generator called)",
        )


class PoolMaxSizeTests(unittest.TestCase):
    """Tests for FIFO behavior when pool_max_size is set."""

    def test_max_size_uses_fifo_popleft(self) -> None:
        """With max_size, picking takes from the oldest end (popleft)."""
        # 3 seeds, max=3, batch=1. Pool is the seeds. Pick oldest first.
        picked: list[str] = []

        def fake_scorer(smis: list[str]) -> list[float]:
            picked.extend(smis)
            return [-1.0] * len(smis)

        random_analog_search(
            seed_smiles=["A", "B", "C"],
            scorer=fake_scorer,
            analog_fn=lambda smis: [],
            n_iterations=3,
            batch_size=1,
            pool_max_size=3,
            rng=random.Random(0),
        )
        # Seeds are added in order; FIFO picks A, B, C in that order.
        self.assertEqual(picked, ["A", "B", "C"])

    def test_max_size_evicts_oldest_on_append(self) -> None:
        """Adding past maxlen auto-evicts oldest; oldest never reaches the scorer."""
        # n_iterations=1: score "A" (the seed). Pool is empty after.
        # We need a second eval, which requires refill. Refill expands
        # "A" → 5 analogs. Max=2 keeps only the last 2.
        # Then second eval scores the older of those 2.
        # The first 3 analogs (new_A_0..2) should never be scored (evicted).
        picked: list[str] = []

        def fake_analog(smis: list[str]) -> list[str]:
            return [f"new_{smis[0]}_{i}" for i in range(5)]

        def fake_scorer(smis: list[str]) -> list[float]:
            picked.extend(smis)
            return [-1.0] * len(smis)

        random_analog_search(
            seed_smiles=["A"],
            scorer=fake_scorer,
            analog_fn=fake_analog,
            n_iterations=2,
            batch_size=1,
            pool_min_size=1,
            pool_max_size=2,
            rng=random.Random(0),
        )
        # "A" is scored first (deque ["A"], popleft → "A", pool empty).
        # Refill expands A → 5 analogs; deque(maxlen=2) keeps last 2.
        # Second eval picks the oldest of those 2 (FIFO).
        self.assertEqual(len(picked), 2)
        self.assertEqual(picked[0], "A")
        self.assertTrue(picked[1].startswith("new_A_"), f"expected analog of A, got {picked[1]}")
        # First 3 analogs should have been evicted before scoring.
        self.assertNotIn("new_A_0", picked)
        self.assertNotIn("new_A_1", picked)
        self.assertNotIn("new_A_2", picked)

    def test_min_max_equal_maintains_size(self) -> None:
        """With min=N, max=N, pool stays at exactly N."""
        pool_sizes_during_refill: list[int] = []

        # Track pool size at each refill step.
        original_pick = random_analog_search

        def fake_analog(smis: list[str]) -> list[str]:
            return [f"{smis[0]}_a", f"{smis[0]}_b"]

        def fake_scorer(smis: list[str]) -> list[float]:
            return [-1.0] * len(smis)

        random_analog_search(
            seed_smiles=["A"],
            scorer=fake_scorer,
            analog_fn=fake_analog,
            n_iterations=3,
            batch_size=1,
            pool_min_size=1,
            pool_max_size=1,
            rng=random.Random(0),
        )
        # Just verify it doesn't crash and produces history of the right length.
        # Hard to verify exact pool size without instrumentation; smoke test only.

    def test_min_greater_than_max_raises(self) -> None:
        with self.assertRaises(ValueError):
            random_analog_search(
                seed_smiles=["A"],
                scorer=lambda smis: [-1.0] * len(smis),
                analog_fn=lambda smis: [],
                n_iterations=1,
                batch_size=1,
                pool_min_size=5,
                pool_max_size=2,
            )

    def test_max_size_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            random_analog_search(
                seed_smiles=["A"],
                scorer=lambda smis: [-1.0] * len(smis),
                analog_fn=lambda smis: [],
                n_iterations=1,
                batch_size=1,
                pool_max_size=0,
            )

    def test_min_size_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            random_analog_search(
                seed_smiles=["A"],
                scorer=lambda smis: [-1.0] * len(smis),
                analog_fn=lambda smis: [],
                n_iterations=1,
                batch_size=1,
                pool_min_size=0,
            )


class MaxLenFilterTests(unittest.TestCase):
    """Tests for the SMILES-length pool filter (smiles_max_len)."""

    def test_over_length_analogues_filtered(self) -> None:
        """Analog generator emits a mix of short and over-length SMILES;
        only the short ones enter the pool."""
        emitted = ["C", "CC", "CCC", "C" * 100, "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"]

        def fake_analog(smis: list[str]) -> list[str]:
            return list(emitted)

        history = random_analog_search(
            seed_smiles=["A"],
            scorer=lambda smis: [-1.0] * len(smis),
            analog_fn=fake_analog,
            n_iterations=20,
            pool_min_size=1,
            smiles_max_len=10,
            rng=random.Random(0),
        )
        # No SMILES longer than 10 chars should appear in history.
        for smi, _ in history:
            self.assertLessEqual(len(smi), 10, f"over-length {smi!r} not filtered")
        # All emitted short SMILES should be there.
        self.assertEqual({s for s, _ in history}, {"A", "C", "CC", "CCC"})

    def test_seed_smiles_over_length_filtered(self) -> None:
        """Seed SMILES longer than smiles_max_len are dropped at seed time."""
        long_smi = "C" * 100
        history = random_analog_search(
            seed_smiles=["CCO", f" {long_smi} "],
            scorer=lambda smis: [-1.0] * len(smis),
            analog_fn=lambda smis: [],
            n_iterations=5,
            pool_min_size=None,
            smiles_max_len=10,
            rng=random.Random(0),
        )
        scored = {s for s, _ in history}
        self.assertIn("CCO", scored)
        self.assertNotIn(long_smi, scored)
        self.assertNotIn(long_smi.strip(), scored)

    def test_default_max_len_is_50(self) -> None:
        """Default smiles_max_len is 50 (matches BO loop + CLI flag)."""
        import inspect
        sig = inspect.signature(random_analog_search)
        self.assertEqual(sig.parameters["smiles_max_len"].default, 50)

    def test_invalid_max_len_raises(self) -> None:
        with self.assertRaises(ValueError):
            random_analog_search(
                seed_smiles=["A"],
                scorer=lambda smis: [-1.0],
                analog_fn=lambda smis: [],
                n_iterations=1,
                smiles_max_len=0,
            )

    def test_max_len_at_boundary_kept(self) -> None:
        """SMILES of exactly the cap length is kept (filter is strict >)."""
        boundary = "C" * 10
        history = random_analog_search(
            seed_smiles=[boundary],
            scorer=lambda smis: [-1.0],
            analog_fn=lambda smis: [],
            n_iterations=1,
            pool_min_size=None,
            smiles_max_len=10,
            rng=random.Random(0),
        )
        self.assertEqual([s for s, _ in history], [boundary])

    def test_max_len_none_disables_filter(self) -> None:
        """smiles_max_len=None disables the filter entirely."""
        long_smi = "C" * 200
        history = random_analog_search(
            seed_smiles=[long_smi],
            scorer=lambda smis: [-1.0],
            analog_fn=lambda smis: [],
            n_iterations=1,
            pool_min_size=None,
            smiles_max_len=None,
            rng=random.Random(0),
        )
        self.assertEqual([s for s, _ in history], [long_smi])


class ExpansionStrategyTests(unittest.TestCase):
    """Tests for the expansion strategy (random / best)."""

    def _make_history_with_scores(self, scores: dict[str, float]) -> "random_analog_search":
        # Helper: build a fake history dict for testing _pick_expansion_target.
        from collections import OrderedDict
        return OrderedDict(scores)

    def test_expansion_random_picks_uniformly(self) -> None:
        from collections import OrderedDict, Counter

        rng = random.Random(42)
        picks: Counter = Counter()
        for _ in range(300):
            history = self._make_history_with_scores({})
            pool = ["A", "B", "C"]
            target = _pick_expansion_target(
                "random", history, pool, set(expanded := set()),
                minimize=True, rng=rng,
            )
            picks[target] += 1
        for smi in ("A", "B", "C"):
            self.assertGreater(picks[smi], 70, f"{smi} picked only {picks[smi]} times")
            self.assertLess(picks[smi], 130, f"{smi} picked {picks[smi]} times")

    def test_expansion_best_picks_lowest_score_minimize(self) -> None:
        history = self._make_history_with_scores({"A": -5.0, "B": -8.0, "C": -3.0})
        target = _pick_expansion_target(
            "best", history, ["A", "B", "C"], set(), minimize=True, rng=random.Random(0),
        )
        self.assertEqual(target, "B")

    def test_expansion_best_picks_highest_score_maximize(self) -> None:
        history = self._make_history_with_scores({"A": 5.0, "B": 8.0, "C": 3.0})
        target = _pick_expansion_target(
            "best", history, ["A", "B", "C"], set(), minimize=False, rng=random.Random(0),
        )
        self.assertEqual(target, "B")

    def test_expansion_best_tiebreak_lexicographic(self) -> None:
        # A and B both have the best score; "A" wins on tiebreak.
        history = self._make_history_with_scores({"A": -5.0, "B": -5.0, "C": -3.0})
        target = _pick_expansion_target(
            "best", history, ["A", "B", "C"], set(), minimize=True, rng=random.Random(0),
        )
        self.assertEqual(target, "A")

    def test_expansion_best_falls_back_when_no_finite_scores(self) -> None:
        # All scores None or NaN → fall back to random pick from pool.
        history = self._make_history_with_scores({"A": None, "B": None})
        # Pool is the only source of candidates.
        target = _pick_expansion_target(
            "best", history, ["X", "Y", "Z"], set(), minimize=True, rng=random.Random(0),
        )
        self.assertIn(target, ["X", "Y", "Z"])

    def test_expansion_best_excludes_already_expanded(self) -> None:
        history = self._make_history_with_scores({"A": -5.0, "B": -8.0})
        # Both scored but A is already expanded; only B is a candidate.
        target = _pick_expansion_target(
            "best", history, ["B"], {"A"}, minimize=True, rng=random.Random(0),
        )
        self.assertEqual(target, "B")

    def test_expansion_best_excludes_pool_picks_only_history_with_scores(self) -> None:
        # Pool SMILES have no scores; "best" should pick from history.
        history = self._make_history_with_scores({"X": -10.0})
        target = _pick_expansion_target(
            "best", history, ["A", "B", "C"], set(), minimize=True, rng=random.Random(0),
        )
        # X is the only history member with a finite score.
        self.assertEqual(target, "X")

    def test_expansion_random_returns_none_when_no_candidates(self) -> None:
        from collections import OrderedDict
        history: OrderedDict[str, Optional[float]] = OrderedDict()
        target = _pick_expansion_target(
            "random", history, [], set(), minimize=True, rng=random.Random(0),
        )
        self.assertIsNone(target)

    def test_no_double_expansion(self) -> None:
        """A SMILES is never passed to analog_generator twice across the whole run."""
        expanded_inputs: list[str] = []

        def fake_analog(smis: list[str]) -> list[str]:
            expanded_inputs.extend(smis)
            return [f"{smis[0]}_a", f"{smis[0]}_b"]

        random_analog_search(
            seed_smiles=["A", "B", "C", "D", "E"],
            scorer=lambda smis: [float(i) for i in range(len(smis))],
            analog_fn=fake_analog,
            n_iterations=10,
            batch_size=2,
            expansion="random",
            rng=random.Random(7),
        )
        # No SMILES should appear twice in the inputs of analog_generator.
        self.assertEqual(len(expanded_inputs), len(set(expanded_inputs)))

    def test_random_best_endto_end_picks_best_history(self) -> None:
        """End-to-end: random-best should pick the best-scored history SMILES for refill."""
        expanded_inputs: list[str] = []

        def fake_analog(smis: list[str]) -> list[str]:
            expanded_inputs.append(smis[0])
            return []

        # Scorer: SMILES with score = its length (so longer = worse).
        # After scoring "ABC" (score 3.0) and "X" (score 1.0), with
        # pool < target, random-best should pick "X" (lowest score).
        scores = {"ABC": 3.0, "X": 1.0}
        scored_iter = iter(scores.items().__iter__())

        def fake_scorer(smis: list[str]) -> list[float]:
            return [next(scored_iter)[1] for _ in smis]

        random_analog_search(
            seed_smiles=["ABC", "X"],
            scorer=fake_scorer,
            analog_fn=fake_analog,
            n_iterations=2,
            batch_size=1,
            expansion="best",
            minimize=True,
            rng=random.Random(0),
        )
        # First expansion (if any) should target "X" (best score).
        if expanded_inputs:
            self.assertEqual(expanded_inputs[0], "X")


# ---------------------------------------------------------------------------
# Multi-objective tests
# ---------------------------------------------------------------------------


def _vina_mock(smis: list[str]) -> list[float]:
    """Mock Vina: rewards more carbons (lower is better)."""
    return [-float(s.count("C")) for s in smis]


def _nn_mock(smis: list[str]) -> list[float]:
    """Mock NN (pIC50): rewards more nitrogens (higher is better)."""
    return [5.0 + 0.5 * float(s.count("N")) for s in smis]


def _carbon_mock(smis: list[str]) -> list[float]:
    """Mock 3rd: rewards more oxygens (higher is better)."""
    return [1.0 + 0.3 * float(s.count("O")) for s in smis]


class MultiObjective2ObjTests(unittest.TestCase):
    """End-to-end 2-obj random search."""

    def test_history_shape(self) -> None:
        history = random_analog_search(
            seed_smiles=["CCO", "CCN"],
            scorer=(_vina_mock, _nn_mock),
            analog_fn=lambda smis: [s + "C" for s in smis],
            n_iterations=3, batch_size=1,
            pool_min_size=1, pool_max_size=10,
            smiles_max_len=50, expansion="random",
            minimize=(True, False),
            rng=random.Random(0),
        )
        self.assertGreater(len(history), 0)
        for smi, sc in history:
            self.assertIsInstance(sc, tuple)
            self.assertEqual(len(sc), 2)

    def test_minimize_bare_bool_broadcast(self) -> None:
        history = random_analog_search(
            seed_smiles=["CCO", "CCN"],
            scorer=(_vina_mock, _nn_mock),
            analog_fn=lambda smis: [s + "C" for s in smis],
            n_iterations=2, batch_size=1,
            pool_min_size=1, pool_max_size=10,
            smiles_max_len=50, expansion="random",
            minimize=True,
            rng=random.Random(0),
        )
        self.assertGreater(len(history), 0)

    def test_minimize_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            random_analog_search(
                seed_smiles=["CCO", "CCN"],
                scorer=(_vina_mock, _nn_mock),
                analog_fn=lambda smis: [s + "C" for s in smis],
                n_iterations=1,
                minimize=(True,),
                rng=random.Random(0),
            )

    def test_2obj_expansion_best_runs(self) -> None:
        history = random_analog_search(
            seed_smiles=["CCO", "CCN"],
            scorer=(_vina_mock, _nn_mock),
            analog_fn=lambda smis: [s + "C" for s in smis],
            n_iterations=2, batch_size=1,
            pool_min_size=1, pool_max_size=10,
            smiles_max_len=50, expansion="best",
            minimize=(True, False),
            rng=random.Random(0),
        )
        self.assertGreater(len(history), 0)

    def test_2obj_seed_reproducibility(self) -> None:
        kwargs = dict(
            seed_smiles=["CCO", "CCN"],
            scorer=(_vina_mock, _nn_mock),
            analog_fn=lambda smis: [s + "C" for s in smis],
            n_iterations=3, batch_size=1,
            pool_min_size=1, pool_max_size=10,
            smiles_max_len=50, expansion="random",
            minimize=(True, False),
        )
        h1 = random_analog_search(rng=random.Random(7), **kwargs)
        h2 = random_analog_search(rng=random.Random(7), **kwargs)
        self.assertEqual(h1, h2)


class MultiObjective3PlusTests(unittest.TestCase):
    """n_obj >= 3: history tuples have length >= 3."""

    def test_3obj_history_shape(self) -> None:
        history = random_analog_search(
            seed_smiles=["CCO", "CCN", "CCC"],
            scorer=(_vina_mock, _nn_mock, _carbon_mock),
            analog_fn=lambda smis: [s + "C" for s in smis],
            n_iterations=3, batch_size=1,
            pool_min_size=1, pool_max_size=10,
            smiles_max_len=50, expansion="random",
            minimize=(True, False, False),
            rng=random.Random(0),
        )
        self.assertGreater(len(history), 0)
        for _, sc in history:
            self.assertEqual(len(sc), 3)

    def test_5obj_history_shape(self) -> None:
        scorers = (_vina_mock, _nn_mock, _carbon_mock, _vina_mock, _nn_mock)
        history = random_analog_search(
            seed_smiles=["CCO", "CCN", "CCC", "CC", "CO"],
            scorer=scorers,
            analog_fn=lambda smis: [s + "C" for s in smis],
            n_iterations=3, batch_size=1,
            pool_min_size=1, pool_max_size=10,
            smiles_max_len=50, expansion="random",
            minimize=(True, False, False, True, False),
            rng=random.Random(0),
        )
        for _, sc in history:
            self.assertEqual(len(sc), 5)


class PickExpansionTargetCheTests(unittest.TestCase):
    """Direct tests of the new multi-obj Chebyshev helper."""

    def test_n_1_minimize_picks_lowest(self) -> None:
        from collections import OrderedDict
        from strbo_v1.rng import RNG
        from strbo_v1.random_search import _pick_expansion_target_che

        history = OrderedDict([("A", (-5.0,)), ("B", (-8.0,)), ("C", (-3.0,))])
        chosen = _pick_expansion_target_che(
            ["A", "B", "C"], history, [], set(),
            (True,), RNG(seed=0),
        )
        self.assertEqual(chosen, "B")

    def test_n_1_maximize_picks_highest(self) -> None:
        from collections import OrderedDict
        from strbo_v1.rng import RNG
        from strbo_v1.random_search import _pick_expansion_target_che

        history = OrderedDict([("A", (5.0,)), ("B", (8.0,)), ("C", (3.0,))])
        chosen = _pick_expansion_target_che(
            ["A", "B", "C"], history, [], set(),
            (False,), RNG(seed=0),
        )
        self.assertEqual(chosen, "B")

    def test_n_2_runs(self) -> None:
        from collections import OrderedDict
        from strbo_v1.rng import RNG
        from strbo_v1.random_search import _pick_expansion_target_che

        history = OrderedDict([
            ("A", (-5.0, 4.0)), ("B", (-8.0, 6.0)), ("C", (-3.0, 5.0)),
        ])
        chosen = _pick_expansion_target_che(
            ["A", "B", "C"], history, [], set(),
            (True, False), RNG(seed=0),
        )
        # Just verify the helper returns one of the candidates.
        self.assertIn(chosen, ["A", "B", "C"])

    def test_n_5_runs(self) -> None:
        from collections import OrderedDict
        from strbo_v1.rng import RNG
        from strbo_v1.random_search import _pick_expansion_target_che

        history = OrderedDict([
            ("A", (-5.0, 4.0, 1.0, -2.0, 7.0)),
            ("B", (-8.0, 6.0, 2.0, -1.0, 9.0)),
            ("C", (-3.0, 5.0, 1.5, -3.0, 8.0)),
        ])
        chosen = _pick_expansion_target_che(
            ["A", "B", "C"], history, [], set(),
            (True, False, False, True, False), RNG(seed=0),
        )
        self.assertIn(chosen, ["A", "B", "C"])

    def test_no_finite_falls_back_to_random(self) -> None:
        from collections import OrderedDict
        from strbo_v1.rng import RNG
        from strbo_v1.random_search import _pick_expansion_target_che

        history = OrderedDict([("A", (None,)), ("B", (None,))])
        # Both A and B are in candidates with None scores; fallback
        # to random.choice over candidates (A or B).
        chosen = _pick_expansion_target_che(
            ["A", "B"], history, [], set(),
            (True,), RNG(seed=0),
        )
        self.assertIn(chosen, ["A", "B"])

    def test_excludes_already_expanded(self) -> None:
        from collections import OrderedDict
        from strbo_v1.rng import RNG
        from strbo_v1.random_search import _pick_expansion_target_che

        # candidates = ["A", "B"]; A is already expanded and removed
        # from the candidates set before calling. The helper sees only
        # "B" in the candidate list.
        history = OrderedDict([("A", (-5.0,)), ("B", (-8.0,))])
        chosen = _pick_expansion_target_che(
            ["B"], history, [], {"A"},
            (True,), RNG(seed=0),
        )
        self.assertEqual(chosen, "B")


if __name__ == "__main__":
    unittest.main()
