# G12C QSAR Baseline Results

Run date: 2026-06-12

Data source:

- ChEMBL API target: `CHEMBL2189121` / KRAS
- Assay filter: description contains `G12C`
- Modeling endpoint: exact `IC50`, `standard_units = nM`
- Label: `pIC50 = 9 - log10(IC50_nM)`

Cleaned dataset:

- Assays matched: 271
- Raw activity rows: 12,668
- Unique cleaned modeling rows: 1,846 canonical SMILES

First run with RDKit enabled:

- Directory: `activity_modeling/runs/g12c_lightgbm_20260612_105542`
- Best model selected on scaffold split: `char_tfidf_linear_svr`
- Best model file: `best_model.joblib`

Key validation metrics:

| Model | Split | RMSE | MAE | R2 | Spearman |
|---|---|---:|---:|---:|---:|
| `char_tfidf_svd_extra_trees` | random | 0.636 | 0.470 | 0.514 | 0.685 |
| `morgan_lightgbm` | random | 0.640 | 0.480 | 0.506 | 0.698 |
| `morgan_hist_gradient_boosting` | random | 0.642 | 0.479 | 0.504 | 0.690 |
| `morgan_random_forest` | random | 0.645 | 0.468 | 0.499 | 0.686 |
| `char_tfidf_linear_svr` | scaffold | 0.633 | 0.506 | 0.515 | 0.728 |
| `char_tfidf_ridge` | scaffold | 0.644 | 0.529 | 0.497 | 0.734 |
| `morgan_random_forest` | scaffold | 0.645 | 0.512 | 0.496 | 0.724 |
| `morgan_lightgbm` | scaffold | 0.686 | 0.562 | 0.430 | 0.703 |

Interpretation:

- Use `char_tfidf_linear_svr` as the conservative first activity model because
  it performed best on scaffold split.
- Use `char_tfidf_svd_extra_trees` as an interpolation model for close analogs
  within similar chemistry, because it performed best on random split.
- LightGBM improved rank correlation on the random split for Morgan features,
  but did not beat the simpler baselines on scaffold split.
- Keep Morgan-fingerprint and LightGBM models as useful consensus baselines, but
  they did not win this first comparison.

Limitations:

- This is a ChEMBL-derived public-data baseline, not a prospective validation.
- Assay/document bias is likely because many rows come from patent-style SAR
  tables.
- G12C covalent inhibitor mechanism is not explicitly modeled beyond molecular
  structure features and simple warhead flags.
- Prospective ranking should combine this score with docking/GNINA, pose
  interactions, ADMET alerts, synthetic route feasibility, and uncertainty.

## Split-Stress Comparison

Run directory: `activity_modeling/runs/g12c_compare_20260612_121753`

Additional experiments added:

- Tanimoto nearest-neighbor SAR baselines: `morgan_tanimoto_knn_k3/k5/k10`
- Document holdout split by primary ChEMBL document
- Assay holdout split by primary ChEMBL assay

Best model by RMSE for each split:

| Split | Best model | RMSE | MAE | R2 | Spearman |
|---|---|---:|---:|---:|---:|
| random | `morgan_tanimoto_knn_k10` | 0.623 | 0.471 | 0.533 | 0.712 |
| scaffold | `char_tfidf_linear_svr` | 0.633 | 0.506 | 0.515 | 0.728 |
| document | `char_tfidf_ridge` | 0.690 | 0.535 | 0.215 | 0.583 |
| assay | `morgan_plus_desc_lightgbm` | 0.951 | 0.717 | 0.082 | 0.502 |

Dataset concentration:

- 1,846 cleaned rows
- 42 primary documents
- 49 primary assays
- Largest document contributes 664 rows
- Largest assay contributes 375 rows

Interpretation:

- Nearest-neighbor SAR is the strongest random-split baseline. For close
  analogs, a Tanimoto-neighbor score should be part of the final ranker.
- Document holdout is worse than scaffold split, which suggests document/patent
  series effects matter.
- Assay holdout is much worse. This strongly suggests the current IC50 label
  mixes non-equivalent assays; improving assay stratification should come before
  spending more effort on model architecture.
- LightGBM helped some rank-correlation metrics, but it did not solve
  assay-transfer generalization.

## Expanded Method Comparison and Case Study

Run directory: `activity_modeling/runs/g12c_expanded_20260612_122903`

Additional methods tested:

- Morgan LinearSVR and ElasticNet
- RDKit descriptor Ridge, ElasticNet, RBF-SVR, RandomForest, ExtraTrees,
  HistGradientBoosting, and LightGBM
- Morgan + RDKit descriptor RandomForest and ExtraTrees
- Simple ensemble regressors combining nearest-neighbor SAR, text Ridge, and
  Morgan RF/LightGBM

Best model by RMSE for each split:

| Split | Best model | RMSE | MAE | R2 | Spearman |
|---|---|---:|---:|---:|---:|
| random | `ensemble_nn_ridge_lgbm` | 0.608 | 0.452 | 0.555 | 0.717 |
| scaffold | `rdkit_desc_elastic_net` | 0.611 | 0.477 | 0.547 | 0.729 |
| document | `ensemble_nn_ridge_lgbm` | 0.685 | 0.526 | 0.228 | 0.625 |
| assay | `morgan_plus_desc_lightgbm` | 0.951 | 0.717 | 0.082 | 0.502 |

Case-study outputs:

- `CASE_STUDY.md`
- `best_by_split.csv`
- `case_study_prediction_cases.csv`
- `case_study_consensus_cases.csv`
- `case_study_nearest_neighbor_cases.csv`

Case-study findings:

- The simple ensemble is now the strongest random/document model, but it still
  fails to transfer cleanly across assay holdout.
- Scaffold split improved with simple RDKit descriptors plus ElasticNet, which
  suggests coarse physicochemical descriptors carry useful extrapolation signal.
- Nearest-neighbor analysis found activity cliffs where a close analog
  similarity above 0.8 still differs by nearly 4 pIC50 units. These cases should
  be flagged in prospective ranking instead of blindly trusting similarity.
- Assay holdout remains the main bottleneck. The next improvement should be
  assay-family stratification or per-assay normalization, not only more model
  architecture search.

## Assay-Family Follow-Up

Run directory:
`activity_modeling/runs/g12c_expanded_20260612_122903/assay_family`

Main assay-family sizes:

| Assay family | Rows | Median pIC50 | Std pIC50 | Assays | Documents |
|---|---:|---:|---:|---:|---:|
| `covalent_labeling` | 664 | 6.260 | 0.614 | 3 | 1 |
| `nucleotide_exchange_mixed_5min_2h` | 311 | 6.780 | 0.876 | 1 | 1 |
| `nucleotide_exchange_2h` | 310 | 5.639 | 0.690 | 2 | 1 |
| `biochemical_other` | 125 | 6.420 | 0.730 | 4 | 4 |
| `nucleotide_exchange_htrf_18h` | 103 | 7.886 | 0.612 | 1 | 1 |
| `nucleotide_exchange_trfret` | 87 | 6.876 | 0.644 | 1 | 1 |
| `competition_spa` | 77 | 7.097 | 0.858 | 1 | 1 |
| `cellular_pERK` | 64 | 6.481 | 0.741 | 1 | 1 |

Median-only baseline diagnostics:

| Baseline | RMSE | R2 |
|---|---:|---:|
| Global median | 0.930 | 0.000 |
| Assay-family broad median | 0.922 | 0.017 |
| Assay-family median | 0.781 | 0.294 |
| Document median | 0.737 | 0.371 |
| Primary-assay median | 0.718 | 0.403 |

Focused subset highlights:

| Experiment | Split | Best model | RMSE | Spearman |
|---|---|---|---:|---:|
| `family_broad_nucleotide_exchange` | random | `rdkit_desc_elastic_net` | 0.660 | 0.754 |
| `family_broad_nucleotide_exchange` | scaffold | `char_tfidf_ridge` | 0.690 | 0.679 |
| `family_nucleotide_exchange_2h` | random | `rdkit_desc_elastic_net` | 0.547 | 0.643 |
| `assay_CHEMBL5738987` | random | `morgan_tanimoto_knn_k5` | 0.523 | 0.635 |
| `assay_CHEMBL5737243` | random | `rdkit_desc_elastic_net` | 0.538 | 0.715 |
| `family_covalent_labeling` | scaffold | `char_tfidf_ridge` | 0.507 | 0.187 |

Interpretation:

- Assay/document identity explains a large amount of label variance. This is
  why the assay-holdout split stays poor even when random/scaffold metrics look
  acceptable.
- Focused subsets can reduce absolute RMSE, but some subsets have narrow
  activity ranges, so ranking quality can still be weak.
- For prospective screening, keep the primary score as a multi-objective rank
  and expose assay/domain risk flags instead of trusting a single predicted
  pIC50.

## Multi-Objective Ranker Smoke Test

Added script:

- `activity_modeling/rank_g12c_candidates.py`

It combines:

- G12C predicted pIC50
- nearest-neighbor pIC50 and Tanimoto similarity
- applicability-domain class
- optional docking score, where more negative Vina-like scores are better
- RDKit property desirability
- optional covalent-warhead reward for G12C workflows
- risk flags and penalties for weak domain, property alerts, docking failures,
  and model-neighbor disagreement

Smoke-test inputs and outputs:

- Input: `output/formula_workflow_outputs/reversible_kras_g13d_inhibitors_formula_to_smiles.csv`
- Activity-scored output:
  `activity_modeling/runs/g12c_expanded_20260612_122903/g13d_smoke_predictions.csv`
- Ranked output:
  `activity_modeling/runs/g12c_expanded_20260612_122903/g13d_direct_ranked.csv`

Top smoke-test rows:

| Rank | Compound | Score | pIC50 pred | NN sim | Risk flags |
|---:|---:|---:|---:|---:|---|
| 1 | 30 | 0.606 | 7.140 | 0.464 |  |
| 2 | 27 | 0.588 | 7.329 | 0.532 | `mol_wt_high` |
| 3 | 32 | 0.542 | 7.171 | 0.563 | `mol_wt_high` |
| 4 | 33 | 0.520 | 7.195 | 0.569 | `mol_wt_high` |
| 5 | 40 | 0.517 | 7.225 | 0.556 | `mol_wt_high;clogp_high` |

Caveat: this smoke test used G13D molecules only to verify the engineering
path. It should not be interpreted as a scientific G12C hit-ranking result.

## Retrospective Pipeline Evaluation

Evaluation directory:
`activity_modeling/runs/g12c_expanded_20260612_122903/evaluation`

Added script:

- `activity_modeling/evaluate_g12c_pipeline.py`

Key outputs:

- `PIPELINE_EVALUATION.md`
- `model_topk_metrics.csv`
- `ablation_topk_metrics.csv`
- `ablation_summary_key_topk.csv`
- `metadata_baseline_regression.csv`
- `candidate_csv_quality_summary.json`

Dataset QA:

- 12,668 raw ChEMBL activity rows
- 1,846 cleaned unique G12C IC50 molecules
- 49 primary assays and 42 primary documents
- pIC50 >= 7 active rate: 0.229
- Largest document contributes 0.360 of cleaned rows
- Largest assay contributes 0.203 of cleaned rows
- 0.535 of molecules have multiple activity records
- 0.233 have replicate/record pIC50 std >= 0.5

Top-5% hit enrichment, pIC50 >= 7:

| Split | Strategy | K | Hits | Precision | Enrichment | ROC-AUC | AP |
|---|---|---:|---:|---:|---:|---:|---:|
| random | random expected | 19 | 3.9 | 0.205 | 1.00 | 0.500 | 0.205 |
| random | QSAR best RMSE | 19 | 18 | 0.947 | 4.61 | 0.883 | 0.755 |
| random | QSAR + train-only NN | 19 | 19 | 1.000 | 4.87 | 0.876 | 0.743 |
| scaffold | random expected | 19 | 5.6 | 0.295 | 1.00 | 0.500 | 0.295 |
| scaffold | QSAR best RMSE | 19 | 12 | 0.632 | 2.14 | 0.853 | 0.609 |
| scaffold | QSAR + train-only NN | 19 | 15 | 0.789 | 2.68 | 0.859 | 0.645 |
| document | random expected | 25 | 1.7 | 0.068 | 1.00 | 0.500 | 0.068 |
| document | QSAR best RMSE | 25 | 7 | 0.280 | 4.09 | 0.874 | 0.329 |
| document | QSAR + train-only NN | 25 | 8 | 0.320 | 4.67 | 0.868 | 0.339 |
| assay | random expected | 21 | 8.5 | 0.406 | 1.00 | 0.500 | 0.406 |
| assay | QSAR best RMSE | 21 | 18 | 0.857 | 2.11 | 0.707 | 0.613 |
| assay | QSAR + train-only NN | 21 | 20 | 0.952 | 2.34 | 0.727 | 0.656 |

Interpretation:

- The workflow is clearly better than random at enriching strong G12C actives
  in retrospective holdouts.
- QSAR + train-only nearest-neighbor SAR is the strongest pure-activity
  ranking strategy among the tested no-docking variants.
- The risk/property penalized no-docking ranker is more conservative and can
  lower pure potency enrichment. Keep this as a balanced developability rank,
  and use QSAR + NN as the potency-focused rank.
- Property-only is not a useful active-enrichment strategy.
- Docking-only and QSAR+docking claims are still unmeasured because no
  retrospective G12C docking CSV was available in this run.

Current candidate CSV QA:

- 35 rows
- valid SMILES rate: 1.000
- formula match rate: 1.000
- duplicate canonical SMILES: 1
- activity nM present rate: 0.429
- corrected-note fraction: 0.571

## G12C Docking Benchmark CSV

Added script:

- `activity_modeling/make_g12c_docking_benchmark.py`

Generated benchmark files:

- `g12c_docking_benchmark.csv`: 1,846 molecules
- `g12c_docking_benchmark_binary.csv`: 1,166 active/inactive molecules

Default label thresholds:

- active: pIC50 >= 7
- inactive: pIC50 <= 6
- intermediate: 6 < pIC50 < 7

Full benchmark label counts:

| Label | Count |
|---|---:|
| active | 423 |
| inactive | 743 |
| intermediate | 680 |

Binary benchmark label counts:

| Label | Count |
|---|---:|
| active | 423 |
| inactive | 743 |

The CSV includes docking CLI columns (`compound_id`, `SMILES`,
`Activity_nM`) plus retrospective evaluation labels (`pIC50`,
`benchmark_label`, `active_pIC50_ge_7`, `inactive_pIC50_le_6`,
assay/document/family metadata). It can be docked with `extract_and_dock.py
dock`, then passed back into `evaluate_g12c_pipeline.py --docking-csv` for
docking-only enrichment metrics.

## Pilot Docking Benchmark

Pilot files:

- Input CSV:
  `activity_modeling/runs/g12c_expanded_20260612_122903/g12c_docking_benchmark_binary_12_interleaved.csv`
- Docking output:
  `output/docking_work/g12c_benchmark_binary_12_interleaved/docking_results.csv`
- Labeled docking output:
  `output/docking_work/g12c_benchmark_binary_12_interleaved/docking_results_with_labels.csv`
- QSAR+docking ranker output:
  `output/docking_work/g12c_benchmark_binary_12_interleaved/g12c_pilot_ranked_qsar_docking.csv`

Pilot setup:

- 12 molecules total
- 6 active, pIC50 >= 7
- 6 inactive, pIC50 <= 6
- AutoDock Vina against `8UN5`, chain `A`
- `exhaustiveness=4`, `num_modes=1`

Docking-only pilot metrics:

| Metric | Value |
|---|---:|
| Overlap with G12C benchmark | 12 |
| ROC-AUC | 0.861 |
| Average precision | 0.856 |
| Top-1 active precision | 1.000 |
| Top-2 active precision | 1.000 |
| Top-10 active precision | 0.600 |

Docking score by class:

| Class | N | Mean Vina score | Median Vina score |
|---|---:|---:|---:|
| active | 6 | -8.553 | -8.854 |
| inactive | 6 | -6.804 | -6.430 |

Interpretation:

- This small pilot shows a real docking signal: active molecules have more
  favorable Vina scores on average, and the top two docking-ranked molecules are
  active.
- One inactive molecule (`G12C_00016`) also docks very strongly, so docking
  score alone still produces false positives.
- The QSAR+docking ranker placed all six active molecules above all six
  inactive molecules in this pilot, but that ranker uses the full fitted QSAR
  model and is not a strict blind retrospective result.
- Runtime is the practical bottleneck: local single-process Vina on large G12C
  ligands is slow, with some inactive ligands taking several minutes each.
  Full benchmark docking should be parallelized or run remotely.

## Standard Retrospective Experiment

Protocol:

- `activity_modeling/STANDARD_EXPERIMENT_PROTOCOL.md`

Added scripts:

- `activity_modeling/prepare_g12c_standard_experiment.py`
- `activity_modeling/evaluate_g12c_standard_experiment.py`

Generated directory:

- `activity_modeling/runs/g12c_expanded_20260612_122903/standard_experiment`

Frozen binary holdout split sizes:

| Split | Rows | Active | Inactive | Active rate |
|---|---:|---:|---:|---:|
| assay | 262 | 165 | 97 | 0.630 |
| document | 318 | 33 | 285 | 0.104 |
| random | 224 | 76 | 148 | 0.339 |
| scaffold | 241 | 110 | 131 | 0.456 |

The standard experiment uses fixed strategies:

- `qsar_primary_fixed`: fixed model `ensemble_nn_ridge_lgbm`
- `nn_train_only`: nearest-neighbor SAR using train split only
- `qsar_plus_nn_fixed`
- `docking_only`
- `qsar_plus_docking_fixed`
- `qsar_nn_docking_fixed`
- `balanced_qsar_nn_docking_property`

`oracle_best_rmse_diagnostic` is diagnostic only and should not be used for
final claims.

No-docking standard baseline, top-5% enrichment, pIC50 >= 7:

| Split | Strategy | K | Hits | Precision | Enrichment | ROC-AUC | AP |
|---|---|---:|---:|---:|---:|---:|---:|
| assay | random expected | 14 | 8.8 | 0.630 | 1.00 | 0.500 | 0.630 |
| assay | QSAR primary | 14 | 14 | 1.000 | 1.59 | 0.808 | 0.864 |
| assay | QSAR + NN | 14 | 14 | 1.000 | 1.59 | 0.812 | 0.872 |
| document | random expected | 16 | 1.7 | 0.104 | 1.00 | 0.500 | 0.104 |
| document | QSAR primary | 16 | 7 | 0.438 | 4.22 | 0.933 | 0.575 |
| document | QSAR + NN | 16 | 10 | 0.625 | 6.02 | 0.926 | 0.579 |
| random | random expected | 12 | 4.1 | 0.339 | 1.00 | 0.500 | 0.339 |
| random | QSAR primary | 12 | 12 | 1.000 | 2.95 | 0.934 | 0.895 |
| random | QSAR + NN | 12 | 12 | 1.000 | 2.95 | 0.928 | 0.889 |
| scaffold | random expected | 13 | 5.9 | 0.456 | 1.00 | 0.500 | 0.456 |
| scaffold | QSAR primary | 13 | 13 | 1.000 | 2.19 | 0.954 | 0.935 |
| scaffold | QSAR + NN | 13 | 13 | 1.000 | 2.19 | 0.956 | 0.937 |

Bootstrap 95% CI highlights:

| Split | Strategy | Metric | Point | 95% CI |
|---|---|---|---:|---|
| document | QSAR primary | top-5% precision | 0.438 | 0.250 - 0.750 |
| document | QSAR + NN | top-5% precision | 0.625 | 0.342 - 0.812 |
| document | QSAR primary | enrichment | 4.216 | 2.667 - 7.868 |
| document | QSAR + NN | enrichment | 6.023 | 3.731 - 8.518 |
| random | QSAR primary | ROC-AUC | 0.934 | 0.893 - 0.967 |
| scaffold | QSAR + NN | ROC-AUC | 0.956 | 0.928 - 0.978 |

Pilot docking coverage in the frozen standard splits is too low for final
docking claims:

| Split | Rows | Docked rows | Docking coverage |
|---|---:|---:|---:|
| assay | 262 | 1 | 0.004 |
| document | 318 | 1 | 0.003 |
| random | 224 | 2 | 0.009 |
| scaffold | 241 | 0 | 0.000 |

Interpretation:

- The fixed QSAR/NN baseline is now reportable on a frozen benchmark with
  bootstrap confidence intervals.
- Full scientific claims about docking require docking `all_unique_test.csv`
  from the standard experiment, not the 12-molecule pilot.
- The current pilot validates the docking plumbing and signal direction, but
  not final docking improvement over QSAR + NN.

## Full Standard Docking Experiment

Docking input:

- `activity_modeling/runs/g12c_expanded_20260612_122903/standard_experiment/docking_inputs/all_unique_test.csv`

Docking outputs:

- `output/docking_work/g12c_standard/all_unique_test/docking_results.csv`
- `output/docking_work/g12c_standard/all_unique_test/docking_activity_joint_score.csv`
- `output/docking_work/g12c_standard/all_unique_test/docking_metadata.json`
- `output/docking_work/g12c_standard/all_unique_test/g12c_standard_ranked_qsar_docking.csv`

Evaluation outputs:

- `activity_modeling/runs/g12c_expanded_20260612_122903/standard_experiment/evaluation_with_docking/STANDARD_EXPERIMENT_REPORT.md`
- `activity_modeling/runs/g12c_expanded_20260612_122903/standard_experiment/evaluation_with_docking/standard_point_metrics.csv`
- `activity_modeling/runs/g12c_expanded_20260612_122903/standard_experiment/evaluation_with_docking/standard_bootstrap_ci.csv`
- `activity_modeling/runs/g12c_expanded_20260612_122903/standard_experiment/evaluation_with_docking/docking_coverage.csv`

Docking setup:

- AutoDock Vina against `8UN5`, chain `A`
- `exhaustiveness=4`, `num_modes=1`, `seed=42`
- Parallel local runner: `activity_modeling/parallel_dock_g12c.py`
- 755 unique frozen holdout molecules docked or attempted
- Final status: 750 scored rows, 1 ligand-prep failure, 4 dock failures
- Docking score range among successful rows: -12.86 to -3.638 kcal/mol

Docking coverage by frozen split:

| Split | Rows | Docked rows | Coverage | Docked active | Docked inactive |
|---|---:|---:|---:|---:|---:|
| assay | 262 | 260 | 0.992 | 163 | 97 |
| document | 318 | 318 | 1.000 | 33 | 285 |
| random | 224 | 221 | 0.987 | 75 | 146 |
| scaffold | 241 | 240 | 0.996 | 109 | 131 |

Docking failures:

| Compound | Status | Reason |
|---|---|---|
| `G12C_00204` | prep failed | ligand PDBQT preparation failed |
| `G12C_00208` | dock failed | Vina timed out after 3600 seconds |
| `G12C_00344` | dock failed | invalid AutoDock atom type `B` in ligand PDBQT |
| `G12C_00776` | dock failed | Vina timed out after 3600 seconds |
| `G12C_01123` | dock failed | Vina timed out after 3600 seconds |

Full standard top-5% enrichment, pIC50 >= 7:

| Split | Strategy | K | Hits | Precision | Enrichment | ROC-AUC | AP |
|---|---|---:|---:|---:|---:|---:|---:|
| assay | QSAR + NN | 14 | 14 | 1.000 | 1.59 | 0.812 | 0.872 |
| assay | docking only | 13 | 3 | 0.231 | 0.37 | 0.358 | 0.520 |
| assay | QSAR + NN + docking | 13 | 13 | 1.000 | 1.60 | 0.787 | 0.856 |
| document | QSAR + NN | 16 | 10 | 0.625 | 6.02 | 0.926 | 0.579 |
| document | docking only | 16 | 4 | 0.250 | 2.41 | 0.490 | 0.141 |
| document | QSAR + NN + docking | 16 | 9 | 0.562 | 5.42 | 0.912 | 0.477 |
| random | QSAR + NN | 12 | 12 | 1.000 | 2.95 | 0.928 | 0.889 |
| random | docking only | 12 | 7 | 0.583 | 1.72 | 0.488 | 0.351 |
| random | QSAR + NN + docking | 12 | 12 | 1.000 | 2.95 | 0.896 | 0.844 |
| scaffold | QSAR + NN | 13 | 13 | 1.000 | 2.19 | 0.956 | 0.937 |
| scaffold | docking only | 12 | 3 | 0.250 | 0.55 | 0.529 | 0.429 |
| scaffold | QSAR + NN + docking | 12 | 12 | 1.000 | 2.20 | 0.947 | 0.932 |

Bootstrap 95% CI highlights, top-5%, pIC50 >= 7:

| Split | Strategy | Metric | Point | 95% CI |
|---|---|---|---:|---|
| assay | docking only | ROC-AUC | 0.358 | 0.288 - 0.436 |
| document | docking only | ROC-AUC | 0.490 | 0.354 - 0.609 |
| random | docking only | ROC-AUC | 0.488 | 0.409 - 0.569 |
| scaffold | docking only | ROC-AUC | 0.529 | 0.456 - 0.604 |
| document | QSAR + NN | top-5% precision | 0.625 | 0.375 - 0.812 |
| document | QSAR + NN + docking | top-5% precision | 0.562 | 0.312 - 0.812 |
| random | QSAR + NN | ROC-AUC | 0.928 | 0.886 - 0.963 |
| random | QSAR + NN + docking | ROC-AUC | 0.896 | 0.843 - 0.942 |
| scaffold | QSAR + NN | ROC-AUC | 0.956 | 0.928 - 0.978 |
| scaffold | QSAR + NN + docking | ROC-AUC | 0.947 | 0.914 - 0.973 |

Final interpretation:

- The final frozen benchmark does not support the claim that Vina docking score
  improves hit enrichment over the QSAR + nearest-neighbor activity baseline.
- Docking-only ranking is weak on the full benchmark: ROC-AUC is near random on
  document/random/scaffold and actively poor on assay holdout.
- Adding docking to QSAR + NN preserves high top-5% precision on random and
  scaffold splits, but lowers ROC-AUC/AP relative to QSAR + NN alone.
- The recommended potency-ranking baseline is therefore `qsar_plus_nn_fixed`.
  Use docking as a secondary triage/filter for pose plausibility, warhead
  placement, steric clashes, and medicinal-chemistry review rather than as a
  primary scalar score.
- The next scientifically useful docking improvement is pose-level validation
  for the covalent G12C warhead geometry and interaction pattern, not simply
  changing the scalar Vina weight.
