import csv
import json
import tempfile
import textwrap
import unittest
from pathlib import Path

import docking_to_analog_search_example as analog_search


class DockingToAnalogSearchExampleTests(unittest.TestCase):
    def test_top_hits_accept_joint_score_csv_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docking_csv = Path(tmp) / "docking_activity_joint_score.csv"
            docking_csv.write_text(
                "compound_id,canonical_smiles,vina_score_kcal_mol,docking_status\n"
                "A,CC(=O)NC,-6.1,ok\n"
                "B,CC(=O)NCC,-8.2,ok\n",
                encoding="utf-8",
            )
            hits = analog_search.read_top_docking_hits(docking_csv, 1)
        self.assertEqual(hits[0]["compound_id"], "B")
        self.assertEqual(hits[0]["score"], "-8.2")

    def test_top_hits_accept_smiles_csv_without_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            smiles_csv = Path(tmp) / "smiles.csv"
            smiles_csv.write_text(
                "compound_id,SMILES\n"
                "A,CC(=O)NC\n"
                "B,CC(=O)NCC\n",
                encoding="utf-8",
            )
            hits = analog_search.read_top_docking_hits(smiles_csv, 2)
        self.assertEqual([hit["compound_id"] for hit in hits], ["A", "B"])
        self.assertEqual(hits[0]["score"], "")

    def test_seed_selection_prefers_literature_activity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            smiles_csv = Path(tmp) / "seed_smiles_activity.csv"
            smiles_csv.write_text(
                "Compound,SMILES,Activity_nM\n"
                "A,CC(=O)NC,5.0\n"
                "B,CC(=O)NCC,0.5\n",
                encoding="utf-8",
            )
            seeds = analog_search.read_seed_smiles(smiles_csv, 1)
        self.assertEqual(seeds[0]["compound_id"], "B")
        self.assertEqual(seeds[0]["activity_nM"], "0.5")

    def test_reasyn_input_txt_uses_smiles_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_txt = Path(tmp) / "reasyn_input.txt"
            rows = analog_search.write_reasyn_input_txt(
                input_txt,
                [{"compound_id": "A", "canonical_smiles": "CCO", "score": "-7.1"}],
            )
            lines = input_txt.read_text(encoding="utf-8").splitlines()
        self.assertEqual(rows[0]["SMILES"], "CCO")
        self.assertEqual(lines, ["SMILES", "CCO"])

    def test_reasyn_command_uses_hit_expansion_flags(self) -> None:
        cmd = analog_search.reasyn_command(
            python_bin="python",
            entrypoint=Path("/tmp/ReaSyn/scripts/sample.py"),
            model_paths_csv="/tmp/ar.ckpt,/tmp/eb.ckpt",
            input_txt=Path("input.txt"),
            output_csv=Path("output.csv"),
            search_width=12,
            exhaustiveness=128,
            num_gpus=-1,
            num_workers_per_gpu=8,
            task_qsize=0,
            result_qsize=0,
            time_limit=10000,
            add_bb_path=None,
            no_exact_break=True,
            num_cycles=12,
            num_editflow_samples=100,
            num_editflow_steps=100,
            mols_to_filter=None,
            filter_sim=0.8,
        )
        self.assertIn("-m", cmd)
        self.assertIn("--search_width", cmd)
        self.assertIn("12", cmd)
        self.assertIn("--exhaustiveness", cmd)
        self.assertIn("128", cmd)
        self.assertIn("--no_exact_break", cmd)
        self.assertNotIn("--add_bb_path", cmd)

    def test_model_paths_resolve_relative_to_reasyn_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "ReaSyn"
            repo.mkdir()
            paths, csv_value = analog_search.resolve_reasyn_model_paths(
                "data/trained_model/ar.ckpt,data/trained_model/eb.ckpt",
                repo,
            )
        self.assertEqual(len(paths), 2)
        self.assertIn(str(repo), csv_value)

    def test_cli_prepares_manifest_input_and_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            seed_csv = tmp_path / "seed_smiles.csv"
            with seed_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["compound_id", "canonical_smiles", "Activity_nM"])
                writer.writeheader()
                writer.writerow({"compound_id": "seed1", "canonical_smiles": "CCO", "Activity_nM": "1.0"})
            reasyn_repo = tmp_path / "ReaSyn"
            (reasyn_repo / "scripts").mkdir(parents=True)
            (reasyn_repo / "scripts" / "sample.py").write_text("", encoding="utf-8")
            output_dir = tmp_path / "reasyn_out"

            return_code = analog_search.main(
                [
                    "--smiles-path",
                    str(seed_csv),
                    "--top-n",
                    "1",
                    "--output-dir",
                    str(output_dir),
                    "--reasyn-repo",
                    str(reasyn_repo),
                ]
            )
            manifest = json.loads((output_dir / "reasyn_manifest.json").read_text(encoding="utf-8"))
            input_lines = (output_dir / "reasyn_input.txt").read_text(encoding="utf-8").splitlines()
            command_script = (output_dir / "run_reasyn_commands.sh").read_text(encoding="utf-8")

        self.assertEqual(return_code, 0)
        self.assertEqual(input_lines, ["SMILES", "CCO"])
        self.assertEqual(manifest["engine"], "reasyn")
        self.assertEqual(manifest["records"][0]["seed_smiles"], "CCO")
        self.assertNotIn("route_file", manifest["records"][0])
        self.assertIn("scripts/sample.py", command_script)
        self.assertIn("--no_exact_break", command_script)
        self.assertIn("(cd ", command_script)

    def test_cli_can_run_fake_reasyn_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            seed_csv = tmp_path / "seed_smiles.csv"
            seed_csv.write_text(
                "compound_id,canonical_smiles,Activity_nM\n"
                "seed1,CCO,1.0\n",
                encoding="utf-8",
            )
            reasyn_repo = tmp_path / "ReaSyn"
            (reasyn_repo / "scripts").mkdir(parents=True)
            (reasyn_repo / "data" / "trained_model").mkdir(parents=True)
            (reasyn_repo / "data" / "trained_model" / "nv-reasyn-ar-166m-v2.ckpt").write_text("fake", encoding="utf-8")
            (reasyn_repo / "data" / "trained_model" / "nv-reasyn-eb-174m-v2.ckpt").write_text("fake", encoding="utf-8")
            (reasyn_repo / "scripts" / "sample.py").write_text(
                textwrap.dedent(
                    """
                    import argparse
                    import csv

                    parser = argparse.ArgumentParser()
                    parser.add_argument("-m", "--model_path")
                    parser.add_argument("-i", "--input")
                    parser.add_argument("-o", "--output")
                    parser.add_argument("--search_width")
                    parser.add_argument("--exhaustiveness")
                    parser.add_argument("--num_gpus")
                    parser.add_argument("--num_workers_per_gpu")
                    parser.add_argument("--task_qsize")
                    parser.add_argument("--result_qsize")
                    parser.add_argument("--time_limit")
                    parser.add_argument("--add_bb_path")
                    parser.add_argument("--no_exact_break", action="store_true")
                    parser.add_argument("--num_cycles")
                    parser.add_argument("--num_editflow_samples")
                    parser.add_argument("--num_editflow_steps")
                    parser.add_argument("--mols_to_filter")
                    parser.add_argument("--filter_sim")
                    args = parser.parse_args()

                    with open(args.output, "w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=["target", "smiles", "score", "synthesis", "num_steps"])
                        writer.writeheader()
                        writer.writerow({"target": "CCO", "smiles": "CCN", "score": "0.5", "synthesis": "demo", "num_steps": "1"})
                    """
                ).lstrip(),
                encoding="utf-8",
            )
            output_dir = tmp_path / "reasyn_out"

            return_code = analog_search.main(
                [
                    "--smiles-path",
                    str(seed_csv),
                    "--output-dir",
                    str(output_dir),
                    "--reasyn-repo",
                    str(reasyn_repo),
                    "--run-reasyn",
                ]
            )
            manifest = json.loads((output_dir / "reasyn_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(return_code, 0)
        self.assertEqual(manifest["run_status"], "ok")
        self.assertEqual(manifest["reasyn_outputs"]["output_summary"]["analog_count"], 1)
        self.assertEqual(manifest["reasyn_outputs"]["output_summary"]["max_similarity"], 0.5)

    def test_docking_score_aggregation_prefers_more_negative_scores(self) -> None:
        scores = [-7.0, -9.0, -8.0]
        self.assertEqual(
            analog_search.aggregate_docking_scores(scores, objective="dock_best_score", top_k=2),
            -9.0,
        )
        self.assertEqual(
            analog_search.aggregate_docking_scores(scores, objective="dock_topk_mean_score", top_k=2),
            -8.5,
        )

    def test_unique_candidate_smiles_and_sampling_are_bounded(self) -> None:
        smiles = analog_search.unique_candidate_smiles([{"smiles": "CCO"}, {"SMILES": "CCO"}, "CCN"])
        self.assertIn("CCO", smiles)
        self.assertIn("CCN", smiles)
        sampled = analog_search.sample_smiles_for_docking(smiles, sample_size=1, seed=1, strategy="first")
        self.assertEqual(len(sampled), 1)

    def test_cli_uses_packaged_demo_csv_when_no_path_is_given(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "reasyn"
            return_code = analog_search.main(
                [
                    "--top-n",
                    "1",
                    "--output-dir",
                    str(output_dir),
                ]
            )
            manifest = json.loads((output_dir / "reasyn_manifest.json").read_text(encoding="utf-8"))
            input_lines = (output_dir / "reasyn_input.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(return_code, 0)
        self.assertEqual(len(manifest["records"]), 1)
        self.assertIn("reasyn_demo", manifest["source_seed_csv"])
        self.assertEqual(manifest["records"][0]["compound_id"], "37")
        self.assertEqual(input_lines[0], "SMILES")


if __name__ == "__main__":
    unittest.main()
