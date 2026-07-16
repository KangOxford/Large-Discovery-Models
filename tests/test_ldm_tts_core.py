from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ldm_tts import (
    JsonlTrajectoryRecorder,
    LDMSearchRoundResult,
    best_item,
    finite_or_none,
    is_finite_number,
    load_jsonl,
    ranked_items,
    run_budgeted_search,
)


class LDMScoringTests(unittest.TestCase):
    def test_finite_helpers_reject_nan_and_missing_values(self) -> None:
        self.assertTrue(is_finite_number("1.25"))
        self.assertEqual(finite_or_none(3), 3.0)
        self.assertIsNone(finite_or_none(float("nan")))
        self.assertIsNone(finite_or_none(None))

    def test_ranking_ignores_nonfinite_scores(self) -> None:
        rows = [
            {"name": "bad", "score": None},
            {"name": "middle", "score": 2.0},
            {"name": "best", "score": 1.0},
        ]
        ranked = ranked_items(rows, lambda row: row["score"], minimize=True)
        self.assertEqual([row["name"] for row in ranked], ["best", "middle"])
        self.assertEqual(best_item(rows, lambda row: row["score"])["name"], "best")


class LDMSearchLoopTests(unittest.TestCase):
    def test_budgeted_loop_appends_history_and_records_rounds(self) -> None:
        history: list[int] = [0]
        records: list[dict] = []

        def build_round(round_idx: int, round_history: list[int]) -> LDMSearchRoundResult[int]:
            return LDMSearchRoundResult(
                history_delta=[round_history[-1] + 1],
                record={"round_idx": round_idx, "history_size": len(round_history)},
            )

        result = run_budgeted_search(
            history,
            budget=4,
            build_round=build_round,
            record_round=records.append,
            start_round=2,
        )

        self.assertEqual(history, [0, 1, 2, 3])
        self.assertEqual(result.rounds_run, 3)
        self.assertIsNone(result.early_stop_reason)
        self.assertEqual([record["round_idx"] for record in records], [2, 3, 4])

    def test_budgeted_loop_stops_after_empty_reservoir_limit(self) -> None:
        history: list[int] = []
        empty_counts: list[tuple[int, int]] = []

        result = run_budgeted_search(
            history,
            budget=1,
            build_round=lambda round_idx, _history: LDMSearchRoundResult(
                record={"round_idx": round_idx},
                empty_reservoir=True,
            ),
            on_empty_reservoir=lambda round_idx, count: empty_counts.append((round_idx, count)),
            max_empty_reservoir_rounds=2,
        )

        self.assertEqual(result.early_stop_reason, "empty_reservoir_limit")
        self.assertEqual(result.rounds_run, 2)
        self.assertEqual(empty_counts, [(0, 1), (1, 2)])


class LDMTrajectoryTests(unittest.TestCase):
    def test_jsonl_recorder_writes_config_and_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            recorder = JsonlTrajectoryRecorder(
                run_dir,
                config_snapshot={"task": "smoke"},
                rounds_filename="events.jsonl",
                sort_keys=True,
            )
            recorder.append_round({"b": 2, "a": 1})
            recorder.write_json("summary.json", {"ok": True})

            self.assertEqual(json.loads((run_dir / "config.json").read_text()), {"task": "smoke"})
            self.assertEqual(load_jsonl(run_dir / "events.jsonl"), [{"a": 1, "b": 2}])
            self.assertEqual(json.loads((run_dir / "summary.json").read_text()), {"ok": True})


if __name__ == "__main__":
    unittest.main()
