# KRAS G12D Activity Model

This directory contains the reference activity model used by the
small-molecule task to predict KRAS G12D potency from SMILES.

## Files

| File | Purpose |
| --- | --- |
| `best_g12d_model.joblib` | Fitted scikit-learn/joblib regression artifact used for inference. |
| `best_g12d_model_metadata.json` | Training provenance, metrics, format, and expected SHA-256. |

Keep both files together. The scorer discovers the metadata sidecar from the
model filename and verifies the declared digest before deserializing the model.

## Model Summary

- Target: KRAS G12D direct-assay activity.
- Input: valid molecular SMILES, canonicalized with RDKit before inference.
- Output: predicted `p_activity = pIC50 = 9 - log10(IC50_nM)`; higher values
  indicate greater predicted potency.
- Training data: 3,044 exact-relation, direct-assay records.
- Selected model: `ensemble_nn_ridge_rf`.
- Selection criterion: lowest RMSE among deployable joblib models on the
  scaffold split.
- Scaffold-split performance: RMSE `0.8850`, MAE `0.6543`, R2 `0.7387`, and
  Spearman correlation `0.8651` on 609 test records.

See `best_g12d_model_metadata.json` for the complete split definitions,
per-assay counts, model ranking, and provenance.

## Use

From the repository root, point the task at the model with:

```bash
export G12D="$PWD/tasks/small_molecule/resources/models/best_g12d_model.joblib"
```

The same path can be supplied explicitly as `args.nn-model-path` in a task
configuration or with `--set args.nn-model-path=...` on the runner command.

## Integrity and Trust

The expected SHA-256 digest is:

```text
a4c15c1124eced2e8dc80a18fdf94752da106168209d804002b0defbf63986ed
```

Verify it after copying or downloading the artifact:

```bash
sha256sum tasks/small_molecule/resources/models/best_g12d_model.joblib
```

Joblib uses pickle-compatible deserialization and can execute arbitrary code.
A matching checksum confirms that the file matches the documented artifact; it
does not make an untrusted artifact safe. Load this model only when it came from
a trusted source.
