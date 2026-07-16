# Standard G12C Screening Experiment Protocol

This protocol is the frozen workflow for turning the G12C QSAR/docking pipeline
from a pilot into a defensible retrospective benchmark.

## 1. Freeze the Benchmark

Use the ChEMBL-derived binary benchmark:

- File: `activity_modeling/runs/g12c_expanded_20260612_122903/g12c_docking_benchmark_binary.csv`
- Active: `pIC50 >= 7`
- Inactive: `pIC50 <= 6`
- Intermediate compounds are excluded from the primary benchmark.

Current benchmark size:

- 1,166 molecules
- 423 active
- 743 inactive

Regenerate only if the source dataset or label thresholds intentionally change:

```bash
conda activate markush-dock
python activity_modeling/make_g12c_docking_benchmark.py \
  --exclude-intermediate \
  --output-csv activity_modeling/runs/g12c_expanded_20260612_122903/g12c_docking_benchmark_binary.csv
```

## 2. Freeze the Splits

Prepare standard split manifests and docking inputs:

```bash
conda activate markush-dock
python activity_modeling/prepare_g12c_standard_experiment.py
```

Generated directory:

`activity_modeling/runs/g12c_expanded_20260612_122903/standard_experiment`

Key files:

- `frozen_split_manifest.csv`
- `split_summary.csv`
- `best_models_by_split.csv`
- `docking_inputs/random_test.csv`
- `docking_inputs/scaffold_test.csv`
- `docking_inputs/document_test.csv`
- `docking_inputs/assay_test.csv`
- `docking_inputs/all_unique_test.csv`
- `run_docking_commands.sh`

Current frozen split sizes:

| Split | Rows | Active | Inactive |
|---|---:|---:|---:|
| random | 224 | 76 | 148 |
| scaffold | 241 | 110 | 131 |
| document | 318 | 33 | 285 |
| assay | 262 | 165 | 97 |

## 3. Fixed Scoring Strategies

Final claims must use fixed strategies:

- `random_expected`
- `qsar_primary_fixed`: fixed model `ensemble_nn_ridge_lgbm`
- `nn_train_only`: nearest neighbor computed from train split only
- `qsar_plus_nn_fixed`
- `docking_only`
- `qsar_plus_docking_fixed`
- `qsar_nn_docking_fixed`
- `balanced_qsar_nn_docking_property`

Do not use `oracle_best_rmse_diagnostic` for final claims. It is a diagnostic
upper-bound because it selects the best model per split after evaluation.

## 4. Run Full Docking

Recommended command: dock each unique frozen test molecule once, then evaluate
by split.

```bash
conda activate markush-dock
python3 extract_and_dock.py dock \
  --csv activity_modeling/runs/g12c_expanded_20260612_122903/standard_experiment/docking_inputs/all_unique_test.csv \
  --allow-unreviewed \
  --pdb-id 8UN5 \
  --chain-id A \
  --work-dir output/docking_work \
  --output-dir output/docking_work/g12c_standard/all_unique_test \
  --exhaustiveness 4 \
  --num-modes 1 \
  --seed 42
```

Local single-process Vina is slow for large, flexible G12C ligands. For the full
standard benchmark, use parallel workers or a remote/GPU-capable environment.
Keep the receptor, box, seed, exhaustiveness, and mode count fixed.

## 5. Evaluate Without Docking

This establishes the QSAR/nearest-neighbor baseline:

```bash
conda activate markush-dock
python activity_modeling/evaluate_g12c_standard_experiment.py \
  --bootstrap-iters 500
```

Outputs:

- `standard_experiment/evaluation/standard_scores.csv`
- `standard_experiment/evaluation/standard_point_metrics.csv`
- `standard_experiment/evaluation/standard_bootstrap_ci.csv`
- `standard_experiment/evaluation/STANDARD_EXPERIMENT_REPORT.md`

## 6. Evaluate With Docking

After docking `all_unique_test.csv`:

```bash
python activity_modeling/evaluate_g12c_standard_experiment.py \
  --docking-csv output/docking_work/g12c_standard/all_unique_test/docking_results.csv \
  --bootstrap-iters 1000 \
  --output-dir activity_modeling/runs/g12c_expanded_20260612_122903/standard_experiment/evaluation_with_docking
```

Final docking conclusions require high docking coverage on every split. A small
pilot run is a plumbing/sanity check only.

## 7. Required Metrics

Primary threshold:

- active if `pIC50 >= 7`

Primary ranking metrics:

- `precision@top5%`
- `enrichment@top5%`
- `ROC-AUC`
- `average_precision`

Secondary metrics:

- `precision@10`
- `precision@20`
- `precision@top10%`
- docking failure rate
- active/inactive docking coverage

Every primary metric should include bootstrap 95% confidence intervals.

## 8. Pose and Structure Checks

Before using docking as scientific evidence, confirm the receptor structure and
residue numbering. The current cleaned `8UN5` receptor does not expose a simple
`CYS A 12` record, so the G12C reactive residue mapping must be verified before
hard-coding covalent geometry checks.

Minimum pose review:

- inspect top active hits
- inspect top false positives
- confirm pocket occupancy
- confirm plausible orientation toward the G12C reactive site after residue
  numbering is mapped
- record whether docking failures are biased toward a class or scaffold family

## 9. Scientific Claim Criteria

A final claim should look like:

> On the frozen G12C scaffold/document/assay holdouts, fixed
> `qsar_nn_docking_fixed` improves top-5% enrichment over fixed
> `qsar_plus_nn_fixed` by X, with bootstrap 95% CI [L, U], while maintaining
> docking coverage above Y%.

Do not claim docking improves the workflow unless:

- docking coverage is high enough for every split
- `qsar_nn_docking_fixed` improves over `qsar_plus_nn_fixed`
- confidence intervals are reported
- pose case studies support the score-based result
- assay/document split performance is explicitly included
