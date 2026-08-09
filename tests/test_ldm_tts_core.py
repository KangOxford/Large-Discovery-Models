from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from ldm_tts import (
    BOObservation,
    BOPrediction,
    BOSelectionResult,
    CandidateTraceRecord,
    FeatureVector,
    JsonlTrajectoryRecorder,
    LDMRoundTrace,
    LDMSearchRoundResult,
    best_item,
    finite_or_none,
    initial_active_operation_schema,
    is_finite_number,
    load_json_object,
    load_jsonl,
    load_operation_schema,
    operation_feature_dim,
    ranked_items,
    reject_keys,
    run_budgeted_search,
    validate_operation_payload,
)
from ldm_tts.dependency_checks import (
    check_plan,
    check_reasyn,
    check_vina,
    cli_args_to_map,
    has_failures,
)
from ldm_tts.runner import build_plan


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

    def test_budgeted_loop_stops_on_empty_selection_by_default(self) -> None:
        history: list[int] = []

        result = run_budgeted_search(
            history,
            budget=1,
            build_round=lambda round_idx, _history: LDMSearchRoundResult(
                record={"round_idx": round_idx},
            ),
        )

        self.assertEqual(history, [])
        self.assertEqual(result.early_stop_reason, "empty_selection")
        self.assertEqual(result.rounds_run, 1)

    def test_budgeted_loop_retries_empty_selection_when_early_stop_disabled(self) -> None:
        history: list[int] = []

        def build_round(round_idx: int, _history: list[int]) -> LDMSearchRoundResult[int]:
            if round_idx < 2:
                return LDMSearchRoundResult(record={"round_idx": round_idx})
            return LDMSearchRoundResult(
                history_delta=[round_idx],
                record={"round_idx": round_idx},
            )

        result = run_budgeted_search(
            history,
            budget=1,
            build_round=build_round,
            allow_early_stop=False,
        )

        self.assertEqual(history, [2])
        self.assertIsNone(result.early_stop_reason)
        self.assertEqual(result.rounds_run, 3)


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


class LDMRunnerConfigTests(unittest.TestCase):
    def test_registered_tasks_resolve_under_unified_tasks_directory(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        expected_modules = {
            "nanogpt": "tasks.nanogpt.ldm_task.procedure",
            "small_molecule": "tasks.small_molecule.ldm_task.procedure",
            "antibody": "tasks.antibody.ldm_task.procedure",
        }

        for task, module in expected_modules.items():
            plan = build_plan(
                {"name": f"{task}_layout", "task": task, "args": {}},
                Path(f"config/{task}/layout.yaml"),
            )

            self.assertEqual(plan["module"], module)
            self.assertEqual(Path(plan["cwd"]), repo_root / "tasks" / task)

    def test_args_can_reference_config_env_values(self) -> None:
        config = {
            "name": "small_molecule_env_refs",
            "task": "small_molecule",
            "env": {
                "G12D": "tasks/small_molecule/resources/models/best_g12d_model.joblib",
                "VINA_BIN": "/opt/vina/bin/vina",
            },
            "args": {
                "nn-model-path": "${G12D}",
                "vina-bin": "${VINA_BIN}",
            },
        }

        plan = build_plan(config, Path("config/small_molecule/test.yaml"))

        self.assertIn("--nn-model-path", plan["argv"])
        nn_path_index = plan["argv"].index("--nn-model-path") + 1
        self.assertTrue(
            plan["argv"][nn_path_index].endswith(
                "tasks/small_molecule/resources/models/best_g12d_model.joblib"
            )
        )
        self.assertIn("--vina-bin", plan["argv"])
        self.assertEqual(plan["argv"][plan["argv"].index("--vina-bin") + 1], "/opt/vina/bin/vina")

    def test_negatable_boolean_false_emits_no_flag(self) -> None:
        config = {
            "name": "small_molecule_boolean_flags",
            "task": "small_molecule",
            "args": {
                "allow-early-stop": False,
                "mock": False,
            },
        }

        plan = build_plan(config, Path("config/small_molecule/test.yaml"))

        self.assertIn("--no-allow-early-stop", plan["argv"])
        self.assertNotIn("--mock", plan["argv"])
        self.assertNotIn("--no-mock", plan["argv"])


class LDMDependencyCheckTests(unittest.TestCase):
    def test_cli_args_to_map_handles_flags_values_and_negation(self) -> None:
        parsed = cli_args_to_map([
            "--mock",
            "--budget",
            "8",
            "--no-allow-early-stop",
            "--seed",
            "-1",
        ])

        self.assertTrue(parsed["mock"])
        self.assertEqual(parsed["budget"], "8")
        self.assertFalse(parsed["allow-early-stop"])
        self.assertEqual(parsed["seed"], "-1")

    def test_mock_small_molecule_dependency_check_skips_external_tools(self) -> None:
        config = {
            "name": "small_molecule_mock_deps",
            "task": "small_molecule",
            "mode": "mock",
            "args": {
                "mock": True,
                "gp-device": "cpu",
                "llm-model-name": "mock-model",
            },
        }
        checks = check_plan(build_plan(config, Path("config/small_molecule/test.yaml")))
        by_name = {check.name: check for check in checks}

        self.assertFalse(has_failures(checks))
        self.assertEqual(by_name["Vina"].status, "skip")
        self.assertEqual(by_name["G12D activity model"].status, "skip")
        self.assertEqual(by_name["ReaSyn"].status, "skip")

    def test_real_small_molecule_dependency_check_reports_missing_external_tools(self) -> None:
        config = {
            "name": "small_molecule_real_deps",
            "task": "small_molecule",
            "mode": "real",
            "args": {
                "method": "m1_stratified_direct_llm_oversample_sir",
                "gp-device": "cpu",
                "llm-url": "http://127.0.0.1:52308/v1",
                "llm-model-name": "mock-model",
                "api-key": "EMPTY",
                "vina-bin": "/definitely/missing/vina",
                "nn-model-path": "missing_activity_model.joblib",
            },
        }
        checks = check_plan(
            build_plan(config, Path("config/small_molecule/test.yaml")),
            include_optional=False,
        )
        failed = {check.name for check in checks if check.status == "fail"}

        self.assertIn("Vina", failed)
        self.assertIn("G12D activity model", failed)

    def test_skip_eval_nanogpt_plan_skips_optional_data_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_file = root / "train.py"
            operation_schema = root / "operation-schema.json"
            train_file.write_text("", encoding="utf-8")
            operation_schema.write_text("{}", encoding="utf-8")
            config = {
                "name": "nanogpt_real_light_check",
                "task": "nanogpt",
                "mode": "real",
                "args": {
                    "train-file": str(train_file),
                    "operation-schema": str(operation_schema),
                    "generator": "operation_tool",
                    "llm-url": "https://llm.example.test/v1",
                    "llm-model-name": "served-model",
                    "api-key": "test-key",
                    "iterations": 0,
                    "skip-eval": True,
                },
            }
            plan = build_plan(config, Path("config/nanogpt/test.yaml"))

            with (
                patch("ldm_tts.dependency_checks.NANOGPT_DATA_DIR", root / "missing-data"),
                patch("ldm_tts.dependency_checks.NANOGPT_TOKENIZER_DIR", root / "missing-tokenizer"),
            ):
                checks = check_plan(plan, include_optional=False)

        by_name = {check.name: check for check in checks}
        self.assertEqual(by_name["prepare.py data"].status, "skip")
        self.assertEqual(by_name["prepare.py tokenizer"].status, "skip")
        self.assertEqual(by_name["CUDA"].status, "skip")

    def test_skip_eval_nanogpt_plan_checks_data_without_no_optional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_file = root / "train.py"
            operation_schema = root / "operation-schema.json"
            train_file.write_text("", encoding="utf-8")
            operation_schema.write_text("{}", encoding="utf-8")
            config = {
                "name": "nanogpt_real_strict_check",
                "task": "nanogpt",
                "mode": "real",
                "args": {
                    "train-file": str(train_file),
                    "operation-schema": str(operation_schema),
                    "generator": "operation_tool",
                    "llm-url": "https://llm.example.test/v1",
                    "llm-model-name": "served-model",
                    "api-key": "test-key",
                    "skip-eval": True,
                },
            }
            plan = build_plan(config, Path("config/nanogpt/test.yaml"))

            with (
                patch("ldm_tts.dependency_checks.NANOGPT_DATA_DIR", root / "missing-data"),
                patch("ldm_tts.dependency_checks.NANOGPT_TOKENIZER_DIR", root / "missing-tokenizer"),
            ):
                checks = check_plan(plan)

        by_name = {check.name: check for check in checks}
        self.assertEqual(by_name["prepare.py data"].status, "fail")
        self.assertEqual(by_name["prepare.py tokenizer"].status, "fail")

    def test_vina_check_rejects_executable_with_failing_help(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vina = Path(tmp) / "vina"
            vina.write_text("#!/bin/sh\necho broken >&2\nexit 2\n", encoding="utf-8")
            vina.chmod(0o755)

            check = check_vina("small_molecule", str(vina), {})

        self.assertEqual(check.status, "fail")
        self.assertIn("status 2", check.message)
        self.assertIn("broken", check.detail)

    def test_reasyn_check_probes_configured_interpreter_imports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "ReaSyn"
            (repo / "reasyn" / "chem").mkdir(parents=True)
            (repo / "reasyn" / "sampler").mkdir(parents=True)
            (repo / "data" / "trained_model").mkdir(parents=True)
            for package in (
                repo / "reasyn",
                repo / "reasyn" / "chem",
                repo / "reasyn" / "sampler",
            ):
                (package / "__init__.py").write_text("", encoding="utf-8")
            (repo / "reasyn" / "chem" / "mol.py").write_text(
                "class Molecule:\n    def __init__(self, smiles):\n        self.smiles = smiles\n",
                encoding="utf-8",
            )
            (repo / "reasyn" / "sampler" / "parallel.py").write_text(
                "READY = True\n", encoding="utf-8"
            )
            ar = repo / "data" / "trained_model" / "ar.ckpt"
            eb = repo / "data" / "trained_model" / "eb.ckpt"
            ar.write_text("ar", encoding="utf-8")
            eb.write_text("eb", encoding="utf-8")

            checks = check_reasyn(
                {
                    "reasyn-repo": str(repo),
                    "reasyn-python": sys.executable,
                    "reasyn-model-path": f"{ar},{eb}",
                    "reasyn-devices": "",
                },
                dict(os.environ),
                Path(tmp),
            )

        by_name = {check.name: check for check in checks}
        self.assertEqual(by_name["ReaSyn repo"].status, "ok")
        self.assertEqual(by_name["ReaSyn Python"].status, "ok")
        self.assertEqual(by_name["ReaSyn import"].status, "ok")
        self.assertEqual(by_name["ReaSyn checkpoints"].status, "ok")

    def test_reasyn_check_reports_broken_runtime_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "ReaSyn"
            (repo / "reasyn" / "chem").mkdir(parents=True)
            (repo / "reasyn" / "sampler").mkdir(parents=True)
            (repo / "reasyn" / "chem" / "mol.py").write_text(
                "class Molecule:\n    pass\n", encoding="utf-8"
            )
            (repo / "reasyn" / "sampler" / "parallel.py").write_text(
                "import dependency_that_is_not_installed\n", encoding="utf-8"
            )

            checks = check_reasyn(
                {
                    "reasyn-repo": str(repo),
                    "reasyn-python": sys.executable,
                    "reasyn-devices": "",
                },
                dict(os.environ),
                Path(tmp),
            )

        import_check = next(
            check for check in checks if check.name == "ReaSyn import"
        )
        self.assertEqual(import_check.status, "fail")
        self.assertIn("dependency_that_is_not_installed", import_check.detail)


class LDMResponseContractTests(unittest.TestCase):
    def test_json_object_loader_accepts_markdown_fenced_objects(self) -> None:
        self.assertEqual(load_json_object('```json\n{"a": 1}\n```'), {"a": 1})

    def test_nested_banned_keys_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "score"):
            reject_keys({"items": [{"score": 1.0}]}, {"score"})


class LDMOperationSpaceTests(unittest.TestCase):
    def test_shared_operation_schema_reports_full_and_active_dimensions(self) -> None:
        project_root = Path(__file__).resolve().parents[1] / "tasks" / "nanogpt"
        schema = load_operation_schema(
            Path("resources/schemas/mock_operations.json"),
            project_root,
        )
        active_schema = initial_active_operation_schema(
            schema,
            Namespace(initial_operation_features="5"),
        )

        self.assertEqual(operation_feature_dim(schema), 27)
        self.assertEqual(operation_feature_dim(active_schema), 16)
        self.assertEqual(list(active_schema.parameters), [
            "DEPTH",
            "WIDTH",
            "MATRIX_LR",
            "EMBEDDING_LR",
            "WEIGHT_DECAY",
        ])

    def test_shared_operation_payload_validation_canonicalizes_values(self) -> None:
        project_root = Path(__file__).resolve().parents[1] / "tasks" / "nanogpt"
        schema = load_operation_schema(
            Path("resources/schemas/mock_operations.json"),
            project_root,
        )

        operations = validate_operation_payload(
            {
                "operations": [
                    {"name": "depth", "op": "set_numeric", "value": 4.0},
                    {"name": "width", "op": "set_choice", "value": "512"},
                ],
            },
            schema,
            max_operations=2,
        )

        self.assertEqual([(op.name, op.op, op.value) for op in operations], [
            ("DEPTH", "set_numeric", 4),
            ("WIDTH", "set_choice", 512),
        ])


class LDMBOTraceContractTests(unittest.TestCase):
    def test_bo_records_serialize_nested_feature_vectors(self) -> None:
        feature = FeatureVector(
            values=(0.1, 0.9),
            version="mock:v1",
            source_id="candidate-1",
            metadata={"space": "toy"},
        )
        observation = BOObservation(
            candidate_id="candidate-1",
            objectives=(-1.0,),
            feature=feature,
        )
        prediction = BOPrediction(
            candidate_id="candidate-1",
            mean=(-1.2,),
            std=(0.3,),
            acquisition_score=0.42,
        )
        selection = BOSelectionResult(
            selected_candidate_ids=("candidate-1",),
            predictions=(prediction,),
        )

        self.assertEqual(feature.to_dict()["values"], (0.1, 0.9))
        self.assertEqual(observation.to_dict()["feature"]["source_id"], "candidate-1")
        self.assertEqual(selection.to_dict()["predictions"][0]["candidate_id"], "candidate-1")
        json.dumps(selection.to_dict())

    def test_round_trace_serializes_candidate_rows(self) -> None:
        trace = LDMRoundTrace(
            round_idx=2,
            task="mock_task",
            history_size_before=3,
            history_size_after=4,
            response_space="direct_candidates",
            acquisition="gp_ucb",
            candidates=(
                CandidateTraceRecord(
                    candidate_id="candidate-1",
                    payload={"x": 1},
                    prediction={"acquisition_score": 0.5},
                    true_scores=(-1.0,),
                    selected=True,
                ),
            ),
            selected_candidate_ids=("candidate-1",),
        )

        payload = trace.to_dict()

        self.assertEqual(payload["candidates"][0]["payload"], {"x": 1})
        self.assertTrue(payload["candidates"][0]["selected"])
        json.dumps(payload)


class LDMTaskSpecTests(unittest.TestCase):
    def test_nanogpt_operation_spec_reports_active_and_full_dimensions(self) -> None:
        from tasks.nanogpt.ldm_task import procedure as nanogpt_procedure

        project_root = Path(__file__).resolve().parents[1] / "tasks" / "nanogpt"
        args = nanogpt_procedure.parse_args([
            "--generator",
            "operation_mock",
            "--operation-schema",
            "resources/schemas/mock_operations.json",
            "--train-file",
            "resources/train/mock_train.py",
            "--method",
            "best_of_n",
        ])
        args.project_root = project_root
        schema = nanogpt_procedure.load_operation_schema(
            Path("resources/schemas/mock_operations.json"),
            project_root,
        )
        active_schema = nanogpt_procedure.initial_active_operation_schema(schema, args)

        spec = nanogpt_procedure.describe_ldm_task(
            args,
            schema,
            active_schema,
            effective_method="best_of_n",
        )

        self.assertEqual(spec.task, "nanogpt")
        self.assertEqual(spec.candidate_space.dimension, 16)
        self.assertEqual(spec.candidate_space.metadata["full_feature_dimension"], 27)
        self.assertIn(
            "train_operations",
            {response_space.name for response_space in spec.response_spaces},
        )

    def test_small_molecule_spec_reports_two_objective_space(self) -> None:
        from tasks.small_molecule.ldm_task import procedure as molecule_procedure

        args = molecule_procedure.parse_args(["--mock"])

        spec = molecule_procedure.describe_ldm_task(args)

        self.assertEqual(spec.task, "small_molecule")
        self.assertEqual(
            [(objective.name, objective.direction) for objective in spec.objectives],
            [("vina", "minimize"), ("activity", "maximize")],
        )
        self.assertEqual(spec.candidate_space.constraints["max_smiles_len"], 80)

    def test_antibody_spec_reports_categorical_sequence_space(self) -> None:
        from tasks.antibody.ldm_task import procedure as antibody_procedure

        args = antibody_procedure.parse_args(
            ["--mock", "--antigen", "SMOKE_ANTIGEN", "--device", "cpu"]
        )

        spec = antibody_procedure.describe_ldm_task(args, {"seq_len": 11}, ["SMOKE_ANTIGEN"])

        self.assertEqual(args.device, "cpu")
        self.assertEqual(spec.task, "antibody")
        self.assertEqual(spec.candidate_space.dimension, 11)
        self.assertEqual(spec.candidate_space.constraints["alphabet_size"], 20)
        self.assertEqual(spec.objectives[0].direction, "minimize")


if __name__ == "__main__":
    unittest.main()
