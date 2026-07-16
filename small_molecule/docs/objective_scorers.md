# Core Objective Scorers

This note summarizes the two production objective scorers used by the
BO loops in `strbo_v1`: `VinaScorer` and `NNScorer`. Both implement the
same callable contract:

For external callers that want structured Python SDK responses instead
of raw `list[float]` scorer outputs, see
[`docs/external_interfaces.md`](external_interfaces.md).

```python
scores = scorer(["CCO", "CCN"])
```

The return value is a `list[float]` aligned to the input SMILES. Invalid
or failed rows return `float("nan")`; the BO loop records the molecule
but excludes non-finite scores from GP training.

## VinaScorer

`VinaScorer` docks each SMILES with AutoDock Vina and returns the best
binding score in kcal/mol. Lower is better, so use `minimize=True`.

```python
from pathlib import Path

from strbo_v1 import VinaScorer, VinaScorerConfig

vina = VinaScorer(
    VinaScorerConfig(
        pdb_id="8UN5",
        chain_id="A",
        cache_dir=Path("/mnt/data0/shared/pdf2dock/vina_cache"),
        vina_bin="/path/to/vina",
        exhaustiveness=4,
        n_poses=3,
        max_workers=1,
        use_cache=True,
    )
)

scores = vina(["CCO", "CCN"])
```

Important settings:

- `vina_bin`: explicit Vina executable. If omitted, `$VINA_BIN` or
  `PATH` is used.
- `cache_dir`: receptor, ligand, pose, log, and result cache directory.
  On the cloud machine, keep this under `/mnt/data0/shared`.
- `pdb_id`, `chain_id`, `ligand_resname`: receptor selection.
- `exhaustiveness`, `n_poses`, `seed`: Vina search controls.
- `max_workers`: parallel docking workers.

## NNScorer

`NNScorer` loads a joblib regression model and predicts pIC50-like
activity. Higher is better, so use `minimize=False`.

```python
from strbo_v1 import NNScorer, NNScorerConfig

nn = NNScorer(
    NNScorerConfig(
        model_path="activity_modeling/best_g12d_model.joblib",
        metadata_path="activity_modeling/best_g12d_model_metadata.json",
        on_error="all_nan",
    )
)

scores = nn(["CCO", "CCN"])
```

Important settings:

- `model_path`: joblib artifact with a callable `.predict`; the shared
  default is `activity_modeling/best_g12d_model.joblib`.
- `metadata_path`: optional model metadata JSON. If omitted, the scorer
  looks for the standard sidecar names next to `model_path`.
- `on_error`: `"all_nan"` keeps a failed batch as NaN scores;
  `"raise"` makes inference failures explicit.

## BO Usage

Single-objective Vina search:

```python
from strbo_v1 import BayesianAnalogSearchConfig, bayesian_analog_search

history = bayesian_analog_search(
    seed_smiles=["CCO", "CCN"],
    scorer=vina,
    analog_fn=generate_analogs_as_smiles,
    config=BayesianAnalogSearchConfig(
        minimize=True,
        acquisition="ei",
        init_size=10,
        batch_size=1,
        n_iterations=10,
    ),
)
```

Two-objective Vina plus NN search:

```python
history = bayesian_analog_search(
    seed_smiles=["CCO", "CCN"],
    scorer=(vina, nn),
    analog_fn=generate_analogs_as_smiles,
    config=BayesianAnalogSearchConfig(
        minimize=(True, False),
        ref_point=(0.0, 5.0),
        ehvi_n_samples=128,
    ),
)
```

For repeated acquisition queries over a fixed history, construct
`AcquisitionEvaluator` once. It fits the GP at construction time and
reuses it for every call:

```python
from strbo_v1 import AcquisitionEvaluator

evaluator = AcquisitionEvaluator(
    history=[("CCO", -7.1), ("CCN", -6.4), ("CCC", -5.9)],
    config=BayesianAnalogSearchConfig(acquisition=("ei", "pi", "ucb")),
)

details = evaluator(["CCCO", "CCNO"])
```

`details` is keyed by queried SMILES. Each value contains posterior
`mean`, `std`, `variance`, and one acquisition value per requested
acquisition function.
