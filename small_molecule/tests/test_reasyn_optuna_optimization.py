import csv
import tempfile
import unittest
from pathlib import Path

import reasyn_optuna_optimization as optimizer
from strbo import Trial


class ReaSynStrBOOptimizationTests(unittest.TestCase):
    def test_archive_truncates_to_best_unique_candidates(self) -> None:
        candidates = [
            optimizer.Candidate(root_id="A", root_smiles="CCO", compound_id="a1", smiles="CCN", score=-7.0),
            optimizer.Candidate(root_id="A", root_smiles="CCO", compound_id="a2", smiles="CCC", score=-9.0),
            optimizer.Candidate(root_id="A", root_smiles="CCO", compound_id="a3", smiles="CCCl", score=-8.0),
            optimizer.Candidate(root_id="A", root_smiles="CCO", compound_id="a4", smiles="CCC", score=-6.0),
        ]
        kept = optimizer.truncate_candidate_archive(candidates, max_candidates=2)
        self.assertEqual([candidate.compound_id for candidate in kept], ["a2", "a3"])

    def test_update_candidate_state_uses_previous_trial_topk_as_next_active(self) -> None:
        initial = [
            optimizer.Candidate(root_id="A", root_smiles="CCO", compound_id="A", smiles="CCO"),
            optimizer.Candidate(root_id="B", root_smiles="CCCl", compound_id="B", smiles="CCCl"),
        ]
        new_topk = [
            optimizer.Candidate(root_id="A", root_smiles="CCO", compound_id="a1", smiles="CCN", score=-8.0, source_trial=0),
            optimizer.Candidate(root_id="A", root_smiles="CCO", compound_id="a2", smiles="CCC", score=-7.0, source_trial=0),
            optimizer.Candidate(root_id="B", root_smiles="CCCl", compound_id="b1", smiles="CCBr", score=-9.0, source_trial=0),
        ]
        archive, active = optimizer.update_candidate_state(
            {"A": [initial[0]], "B": [initial[1]]},
            initial,
            new_topk,
            top_k=1,
            max_candidates_per_seed=2,
        )
        self.assertEqual([candidate.compound_id for candidate in active], ["a1", "b1"])
        self.assertLessEqual(len(archive["A"]), 2)
        self.assertLessEqual(len(archive["B"]), 2)

    def test_annotate_reasyn_output_keeps_initial_seed_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw = tmp_path / "reasyn_analogs.csv"
            raw.write_text(
                "target,smiles,score,synthesis,num_steps\n"
                "CCN,CCC,0.7,route,1\n",
                encoding="utf-8",
            )
            annotated = tmp_path / "annotated.csv"
            active = [
                optimizer.Candidate(
                    root_id="seed_1",
                    root_smiles="CCO",
                    compound_id="seed_1_trial0_analog_1",
                    smiles="CCN",
                    score=-8.0,
                    activity_nM="0.34",
                )
            ]
            rows = optimizer.annotate_reasyn_analog_csv(raw, annotated, active, trial_number=1)
            with annotated.open(newline="", encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["analog_group_id"], "seed_1")
        self.assertEqual(rows[0]["parent_smiles"], "CCN")
        self.assertEqual(rows[0]["root_seed_smiles"], "CCO")
        self.assertEqual(csv_rows[0]["Activity_nM"], "0.34")

    def test_strbo_trial_samples_reasyn_params(self) -> None:
        args = optimizer.build_arg_parser().parse_args([])
        trial = Trial(
            0,
            {
                "search_width": 8,
                "exhaustiveness": 64,
                "num_cycles": 6,
                "num_editflow_samples": 30,
                "num_editflow_steps": 40,
                "filter_sim": 0.75,
                "no_exact_break": True,
            },
        )

        params = optimizer.sample_reasyn_params(trial, args)

        self.assertEqual(params["search_width"], 8)
        self.assertEqual(params["exhaustiveness"], 64)
        self.assertEqual(params["filter_sim"], 0.75)
        self.assertTrue(params["no_exact_break"])


if __name__ == "__main__":
    unittest.main()
