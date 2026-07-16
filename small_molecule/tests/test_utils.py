"""Tests for ``strbo_v1.utils`` (``FIFOSet``)."""

from __future__ import annotations

import unittest

from strbo_v1.utils import FIFOSet


class FIFOSetTests(unittest.TestCase):
    """``FIFOSet`` is a FIFO-ordered collection with O(1) membership.

    Combines a ``collections.deque`` (insertion-ordered, optionally
    bounded with auto-eviction of the oldest entry on append) with a
    ``set`` (O(1) membership and dedup). The two structures must
    stay in sync — these tests cover both the FIFO order and the
    set/queue invariant under all operations.
    """

    def test_unbounded_basic(self) -> None:
        f: FIFOSet = FIFOSet()
        f.add("a")
        f.add("b")
        f.add("c")
        self.assertEqual(len(f), 3)
        self.assertEqual(list(f), ["a", "b", "c"])  # FIFO
        self.assertIn("a", f)
        self.assertIn("b", f)
        self.assertIn("c", f)
        self.assertNotIn("d", f)

    def test_dedup_returns_false(self) -> None:
        f: FIFOSet = FIFOSet()
        self.assertTrue(f.add("a"))
        self.assertFalse(f.add("a"))  # duplicate
        self.assertFalse(f.add("a"))
        self.assertEqual(len(f), 1)
        self.assertEqual(list(f), ["a"])

    def test_iteration_preserves_insertion_order(self) -> None:
        f: FIFOSet = FIFOSet()
        for s in ["x", "y", "z", "w"]:
            f.add(s)
        self.assertEqual(list(f), ["x", "y", "z", "w"])
        self.assertEqual(list(iter(f)), ["x", "y", "z", "w"])

    def test_popleft_returns_oldest(self) -> None:
        f: FIFOSet = FIFOSet()
        f.add("first")
        f.add("second")
        f.add("third")
        self.assertEqual(f.popleft(), "first")
        self.assertEqual(f.popleft(), "second")
        self.assertEqual(f.popleft(), "third")
        # After all popped: set is empty.
        self.assertEqual(len(f), 0)
        self.assertFalse(f)

    def test_popleft_on_empty_raises(self) -> None:
        f: FIFOSet = FIFOSet()
        with self.assertRaises(IndexError):
            f.popleft()

    def test_discard_present(self) -> None:
        f: FIFOSet = FIFOSet()
        f.add("a")
        f.add("b")
        f.discard("a")
        self.assertNotIn("a", f)
        self.assertIn("b", f)
        self.assertEqual(len(f), 1)
        self.assertEqual(list(f), ["b"])

    def test_discard_absent_no_error(self) -> None:
        f: FIFOSet = FIFOSet()
        f.discard("nope")  # no-op
        self.assertEqual(len(f), 0)

    def test_max_size_auto_evicts_oldest(self) -> None:
        f: FIFOSet = FIFOSet(max_size=3)
        f.add("a")
        f.add("b")
        f.add("c")
        f.add("d")  # evicts "a"
        f.add("e")  # evicts "b"
        self.assertEqual(len(f), 3)
        self.assertEqual(list(f), ["c", "d", "e"])
        self.assertNotIn("a", f)
        self.assertNotIn("b", f)

    def test_max_size_one_only_most_recent_survives(self) -> None:
        f: FIFOSet = FIFOSet(max_size=1)
        for s in ["a", "b", "c", "d", "e"]:
            f.add(s)
        self.assertEqual(len(f), 1)
        self.assertEqual(list(f), ["e"])
        self.assertEqual(f.popleft(), "e")
        self.assertEqual(len(f), 0)

    def test_max_size_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            FIFOSet(max_size=0)

    def test_max_size_negative_raises(self) -> None:
        with self.assertRaises(ValueError):
            FIFOSet(max_size=-1)

    def test_max_size_property(self) -> None:
        self.assertIsNone(FIFOSet().max_size)
        self.assertEqual(FIFOSet(max_size=5).max_size, 5)
        self.assertEqual(FIFOSet(max_size=1).max_size, 1)

    def test_contains_object_lookup(self) -> None:
        f: FIFOSet = FIFOSet()
        f.add("a")
        # `42 in f` should not raise.
        self.assertNotIn(42, f)
        self.assertIn("a", f)

    def test_bool(self) -> None:
        self.assertFalse(bool(FIFOSet()))
        f: FIFOSet = FIFOSet()
        f.add("a")
        self.assertTrue(bool(f))

    def test_repr_includes_max_size(self) -> None:
        self.assertIn("None", repr(FIFOSet()))
        self.assertIn("3", repr(FIFOSet(max_size=3)))

    def test_popleft_keeps_set_in_sync(self) -> None:
        """Auto-eviction on add + manual popleft must keep set/queue aligned."""
        f: FIFOSet = FIFOSet(max_size=2)
        f.add("a")
        f.add("b")
        f.add("c")  # evicts "a"
        # Queue = ["b", "c"], set = {"b", "c"}.
        self.assertEqual(f.popleft(), "b")
        # Queue = ["c"], set = {"c"}.
        self.assertEqual(len(f), 1)
        self.assertEqual(list(f), ["c"])
        # `c in f` should be True; `b in f` should be False.
        self.assertIn("c", f)
        self.assertNotIn("b", f)

    def test_repeated_popleft_then_add_stays_in_sync(self) -> None:
        """Stress: evict via popleft, add more, repeat; invariant holds."""
        f: FIFOSet = FIFOSet(max_size=2)
        for s in ["a", "b", "c", "d", "e", "f"]:
            f.add(s)
        # Queue should be ["e", "f"], set = {"e", "f"}.
        self.assertEqual(list(f), ["e", "f"])
        self.assertEqual(len(f), 2)
        f.popleft()  # removes "e"
        f.add("g")
        # Queue = ["f", "g"], set = {"f", "g"}.
        self.assertEqual(list(f), ["f", "g"])
        self.assertEqual(len(f), 2)
        self.assertNotIn("e", f)
        self.assertNotIn("d", f)

    def test_iter_empty(self) -> None:
        self.assertEqual(list(FIFOSet()), [])


if __name__ == "__main__":
    unittest.main()
