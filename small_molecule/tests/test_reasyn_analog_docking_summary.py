import csv
import tempfile
import unittest
from pathlib import Path

try:
    import extract_and_dock as workflow
except ModuleNotFoundError as exc:
    if exc.name != "pydantic":
        raise
    workflow = None


@unittest.skipIf(workflow is None, "extract_and_dock tests require project dependencies such as pydantic")
class ReaSynAnalogDockingSummaryTests(unittest.TestCase):
    def test_reasyn_output_is_grouped_and_ranked_by_vina_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            analog_csv = tmp_path / "reasyn_analogs.csv"
            analog_csv.write_text(
                "target,smiles,score,synthesis,num_steps\n"
                "CCO,CCN,0.6,route_a,2\n"
                "CCO,CCC,0.7,route_b,1\n"
                "CCCl,CCBr,0.8,route_c,1\n",
                encoding="utf-8",
            )
            compounds = workflow.compounds_from_csv(analog_csv, approved_ids=[], allow_unreviewed=True)
            results = [
                workflow.DockingResult(
                    compound_id=compounds[0].compound_id,
                    canonical_smiles="CCN",
                    score=-7.0,
                    pose_ref="pose_a.sdf",
                    status="ok",
                ),
                workflow.DockingResult(
                    compound_id=compounds[1].compound_id,
                    canonical_smiles="CCC",
                    score=-8.5,
                    pose_ref="pose_b.sdf",
                    status="ok",
                ),
                workflow.DockingResult(
                    compound_id=compounds[2].compound_id,
                    canonical_smiles="CCBr",
                    score=-9.1,
                    pose_ref="pose_c.sdf",
                    status="ok",
                ),
            ]

            summary = workflow.write_analog_group_score_csvs(
                tmp_path / "analog_group_topk.csv",
                tmp_path / "analog_overall_best.csv",
                compounds,
                results,
                top_k=1,
            )
            with (tmp_path / "analog_group_topk.csv").open(encoding="utf-8") as handle:
                topk_rows = list(csv.DictReader(handle))
            with (tmp_path / "analog_overall_best.csv").open(encoding="utf-8") as handle:
                overall_rows = list(csv.DictReader(handle))

        self.assertTrue(summary["written"])
        self.assertEqual([compound.analog_group_id for compound in compounds], ["seed_1", "seed_1", "seed_2"])
        self.assertEqual(
            [compound.compound_id for compound in compounds],
            ["seed_1_analog_1", "seed_1_analog_2", "seed_2_analog_1"],
        )
        self.assertEqual(len(topk_rows), 2)
        self.assertEqual(len(summary["group_best"]), 2)
        self.assertEqual(topk_rows[0]["compound_id"], "seed_1_analog_2")
        self.assertEqual(summary["group_best"][0]["compound_id"], "seed_1_analog_2")
        self.assertEqual(topk_rows[0]["analog_rank"], "1")
        self.assertEqual(overall_rows[0]["compound_id"], "seed_2_analog_1")


if __name__ == "__main__":
    unittest.main()
