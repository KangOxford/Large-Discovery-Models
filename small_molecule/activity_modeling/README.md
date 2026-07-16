# KRAS G12C Activity Modeling

Independent QSAR workspace for KRAS G12C activity scoring.

The first model target is a small, target-specific regression model trained from
public ChEMBL KRAS G12C IC50 records. It is designed to produce an activity
score that can later be merged with docking, pose-interaction, ADMET, and
synthesis scores.

## Quick Start

```bash
python3 activity_modeling/train_g12c_qsar.py
```

For the RDKit/Morgan fingerprint and scaffold-split comparison, use an
environment with both RDKit and scikit-learn, for example:

```bash
conda activate markush-dock
python -m pip install scikit-learn joblib lightgbm optuna
python activity_modeling/train_g12c_qsar.py
```

The committed run directory contains metrics, benchmark CSVs, and ranked demo
outputs, but not `best_model.joblib` or the full cleaned training dataset. To
score new molecules, regenerate the model with `train_g12c_qsar.py` or pass an
explicit `--model-path` to the prediction/ranking scripts.

For local Optuna debugging without modifying your main environment:

```bash
python3 -m venv --system-site-packages .venv-optuna
.venv-optuna/bin/python -m pip install --upgrade pip optuna
```

Default outputs are written to `activity_modeling/runs/g12c_<timestamp>/`:

- `raw_g12c_chembl_activities.csv`: raw ChEMBL activity rows with SMILES.
- `g12c_ic50_dataset.csv`: cleaned, deduplicated modeling table.
- `metrics.csv`: model comparison table.
- `best_model.joblib`: best fitted sklearn pipeline.
- `best_model_metadata.json`: dataset, split, and metric metadata.
- `predictions_random_split.csv`: holdout predictions for inspection.

## Optuna Tuning

Run a small tuning pass and include the tuned pipeline in the normal model
comparison:

```bash
.venv-optuna/bin/python activity_modeling/train_g12c_qsar.py \
  --output-dir activity_modeling/runs/g12c_optuna_smoke \
  --model-filter char_tfidf_ridge,char_tfidf_linear_svr \
  --optuna-trials 20 \
  --optuna-models char_tfidf_ridge,char_tfidf_linear_svr \
  --optuna-split scaffold
```

Optuna writes `optuna_<model>_trials.csv` and `optuna_summary.json` under the
run directory. The final `best_model.joblib` is still selected by the existing
split-aware comparison across baseline and tuned models.

## Current Model Families

The script always supports SMILES character n-gram models:

- `char_tfidf_ridge`
- `char_tfidf_linear_svr`
- `char_tfidf_svd_random_forest`
- `char_tfidf_svd_extra_trees`
- `char_tfidf_svd_hist_gradient_boosting`

If LightGBM is installed, it also enables:

- `char_tfidf_svd_lightgbm`
- `morgan_lightgbm`
- `rdkit_desc_lightgbm`
- `morgan_plus_desc_lightgbm`

If RDKit is installed, it also enables Morgan fingerprint models and Murcko
scaffold splitting:

- `morgan_ridge`
- `morgan_elastic_net`
- `morgan_linear_svr`
- `morgan_random_forest`
- `morgan_extra_trees`
- `morgan_hist_gradient_boosting`
- `morgan_tanimoto_knn_k3/k5/k10`
- `rdkit_desc_*`
- `morgan_plus_desc_*`
- `ensemble_nn_ridge_rf`
- `ensemble_nn_ridge_lgbm`

Validation splits currently include:

- `random`
- `scaffold` when RDKit is installed
- `document` holdout by primary ChEMBL document
- `assay` holdout by primary ChEMBL assay

The plain `python3` environment in this workspace may not include RDKit. The
script will still run with SMILES n-gram models and records that RDKit models
were skipped.

## Notes

This is a baseline QSAR model, not a standalone drug-discovery decision engine.
Use its output as one term in a multi-objective ranking with docking/GNINA,
pose checks, ADMET alerts, route feasibility, and model uncertainty.

## Case Study

After training, generate diagnostic tables and a short report:

```bash
conda activate markush-dock
python activity_modeling/case_study_g12c.py \
  --run-dir activity_modeling/runs/g12c_expanded_20260612_122903
```

The report highlights split-specific winners, largest prediction errors,
model-disagreement cases, and nearest-neighbor SAR cliffs.

## Score New CSVs

Use the trained G12C activity model and nearest-neighbor SAR check on an analog
or docking CSV:

```bash
conda activate markush-dock
python activity_modeling/predict_g12c_activity.py \
  --input-csv path/to/analogs.csv \
  --output-csv path/to/analogs_g12c_activity.csv
```

The output adds `g12c_predicted_pIC50`, nearest-neighbor pIC50/similarity, and
an applicability-domain flag.

## Multi-Objective Ranking

Rank a candidate CSV directly. If the CSV has not already been scored, the
ranker will load the trained G12C model, add activity predictions and
nearest-neighbor SAR fields, compute RDKit property terms, and then rank:

```bash
conda activate markush-dock
python activity_modeling/rank_g12c_candidates.py \
  --input-csv path/to/candidates_or_docking.csv \
  --output-csv path/to/candidates_ranked.csv \
  --prefer-covalent-warhead
```

If the input includes docking columns such as `vina_score_kcal_mol`,
`docking_score`, `affinity`, or `score`, the ranker adds a docking component
where more negative values are better. A separate docking CSV can also be
merged:

```bash
python activity_modeling/rank_g12c_candidates.py \
  --input-csv path/to/candidates.csv \
  --docking-csv path/to/docking_results.csv \
  --merge-on compound_id \
  --output-csv path/to/candidates_ranked.csv
```

The main docking CLI can call this ranker automatically after docking:

```bash
python3 extract_and_dock.py dock \
  --csv path/to/reviewed_compounds.csv \
  --allow-unreviewed \
  --pdb-id 8UN5 \
  --chain-id A \
  --output-dir output/docking_work/results \
  --rank-g12c \
  --g12c-run-dir activity_modeling/runs/g12c_expanded_20260612_122903
```

Key output columns include `multi_objective_rank`,
`multi_objective_score`, `risk_flags`, `g12c_predicted_pIC50`,
`nearest_neighbor_similarity`, `g12c_applicability_domain`,
`docking_component`, and `property_component`.

## Assay-Family Follow-Up

Run assay-family stratification and focused subset experiments:

```bash
conda activate markush-dock
python activity_modeling/assay_family_experiments.py \
  --run-dir activity_modeling/runs/g12c_expanded_20260612_122903
```

This writes `ASSAY_FAMILY_REPORT.md`, family summaries, focused subset metrics,
and centered-label diagnostics under the run directory.

## Pipeline Evaluation

Run retrospective evaluation for regression, top-k enrichment, ablation, metadata
baselines, current candidate CSV quality, and optional docking-score validation:

```bash
conda activate markush-dock
python activity_modeling/evaluate_g12c_pipeline.py \
  --run-dir activity_modeling/runs/g12c_expanded_20260612_122903
```

Outputs are written to `<run-dir>/evaluation/`, including
`PIPELINE_EVALUATION.md`, `ablation_topk_metrics.csv`,
`model_topk_metrics.csv`, and `candidate_csv_quality_summary.json`.

If you have a retrospective docking CSV with known G12C molecules, pass it with:

```bash
python activity_modeling/evaluate_g12c_pipeline.py \
  --docking-csv path/to/docking_results.csv
```

## G12C Docking Benchmark CSV

Generate a ChEMBL-derived G12C benchmark CSV that can be passed directly to the
repo docking CLI and later used for docking enrichment metrics:

```bash
conda activate markush-dock
python activity_modeling/make_g12c_docking_benchmark.py
```

Default output:

- `<run-dir>/g12c_docking_benchmark.csv`: all cleaned molecules, including
  active, inactive, and intermediate labels.

For a binary active/inactive benchmark:

```bash
python activity_modeling/make_g12c_docking_benchmark.py \
  --exclude-intermediate \
  --output-csv activity_modeling/runs/g12c_expanded_20260612_122903/g12c_docking_benchmark_binary.csv
```

Labels default to:

- active: `pIC50 >= 7`
- inactive: `pIC50 <= 6`
- intermediate: between 6 and 7

Then dock the benchmark CSV:

```bash
python3 extract_and_dock.py dock \
  --csv activity_modeling/runs/g12c_expanded_20260612_122903/g12c_docking_benchmark_binary.csv \
  --allow-unreviewed \
  --pdb-id 8UN5 \
  --chain-id A \
  --work-dir output/docking_work \
  --output-dir output/docking_work/g12c_benchmark_binary \
  --exhaustiveness 4 \
  --num-modes 3
```

For the frozen standard experiment, use the prepared unique holdout molecules
and the parallel runner:

```bash
conda activate markush-dock
python activity_modeling/parallel_dock_g12c.py \
  --csv activity_modeling/runs/g12c_expanded_20260612_122903/standard_experiment/docking_inputs/all_unique_test.csv \
  --output-dir output/docking_work/g12c_standard/all_unique_test \
  --work-dir output/docking_work \
  --exhaustiveness 4 \
  --num-modes 1 \
  --workers 5 \
  --vina-cpus 1
```

Then evaluate QSAR, nearest-neighbor SAR, docking-only, and QSAR+docking on the
frozen splits:

```bash
python activity_modeling/evaluate_g12c_standard_experiment.py \
  --docking-csv output/docking_work/g12c_standard/all_unique_test/docking_results.csv \
  --bootstrap-iters 1000 \
  --output-dir activity_modeling/runs/g12c_expanded_20260612_122903/standard_experiment/evaluation_with_docking
```

After docking, evaluate docking enrichment:

```bash
python activity_modeling/evaluate_g12c_pipeline.py \
  --docking-csv output/docking_work/g12c_benchmark_binary/docking_results.csv
```

## Standard Retrospective Experiment

The stricter, reportable experiment is documented in
`activity_modeling/STANDARD_EXPERIMENT_PROTOCOL.md`.

Prepare frozen split manifests and docking inputs:

```bash
conda activate markush-dock
python activity_modeling/prepare_g12c_standard_experiment.py
```

This creates:

- `<run-dir>/standard_experiment/frozen_split_manifest.csv`
- `<run-dir>/standard_experiment/split_summary.csv`
- `<run-dir>/standard_experiment/docking_inputs/all_unique_test.csv`
- `<run-dir>/standard_experiment/run_docking_commands.sh`

Evaluate fixed QSAR/nearest-neighbor baselines with bootstrap confidence
intervals:

```bash
python activity_modeling/evaluate_g12c_standard_experiment.py \
  --bootstrap-iters 500
```

After full docking of `all_unique_test.csv`, evaluate the fixed docking and
QSAR+docking strategies:

```bash
python activity_modeling/evaluate_g12c_standard_experiment.py \
  --docking-csv output/docking_work/g12c_standard/all_unique_test/docking_results.csv \
  --bootstrap-iters 1000 \
  --output-dir activity_modeling/runs/g12c_expanded_20260612_122903/standard_experiment/evaluation_with_docking
```

Use `oracle_best_rmse_diagnostic` only as a diagnostic upper-bound; final claims
should use fixed strategies such as `qsar_primary_fixed`,
`qsar_plus_nn_fixed`, and `qsar_nn_docking_fixed`.
