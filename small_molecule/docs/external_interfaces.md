# External Scoring and Acquisition Interfaces

This document describes the public interfaces for callers outside the
internal BO loop. Use the Python SDK when the caller runs in the same
Python environment. Use the JSON helpers or HTTP adapter when the caller
is a notebook, subprocess, service, or another language runtime.

Public layers:

- Python SDK: `strbo_v1.external_interfaces`
- JSON boundary: `bo_api.py`
- Framework-neutral HTTP adapter: `bo_api_http.py`

Provider settings such as the Vina binary, cache directory, NN model
file, and GP device are deployment settings. Pass them as Python kwargs
or environment configuration. Do not rely on external client JSON for
those values.

On the shared cloud machine, keep generated caches and copied working
artifacts under `/mnt/data0/shared`, not under the home directory.

## Python SDK Quick Start

Run from the repository root, or make sure the repository is on
`PYTHONPATH`.

```python
from strbo_v1.external_interfaces import (
    evaluate_acquisition,
    score_nn,
    score_vina,
)
```

All SDK functions return a Python `dict` with the same top-level shape:

```python
{
    "ok": True,
    "items": [],
    "errors": [],
}
```

Each result in `items` is aligned to one input SMILES:

```python
{
    "smiles": "CCO",
    "ok": True,
    "value": -7.1,
    "error": None,
    "details": {},
}
```

Per-SMILES failures are item-level failures. They do not fail the whole
request:

```python
failed = [row for row in response["items"] if not row["ok"]]
```

Request-level errors, such as an invalid history shape or a missing
provider file, raise in the Python SDK. The JSON wrappers catch those
exceptions and convert them into structured JSON error responses.

## Vina Scoring SDK

`score_vina` docks SMILES with AutoDock Vina and returns one structured
item per input molecule. Lower values are better for Vina, so BO callers
usually use `minimize=True`.

### Minimal call

```python
from strbo_v1.external_interfaces import score_vina

response = score_vina(
    ["CCO", "CCN"],
    vina_bin="/opt/bin/vina",
    vina_cache_dir="/mnt/data0/shared/pdf2dock/vina_cache",
    max_workers=1,
)

for row in response["items"]:
    if row["ok"]:
        print(row["smiles"], row["value"], row["details"]["pose_ref"])
    else:
        print(row["smiles"], row["error"])
```

Successful Vina items include:

- `value`: best docking score.
- `details.status`: docking status.
- `details.canonical_smiles`: canonicalized molecule when available.
- `details.pose_ref`: pose output reference when available.
- `details.cached`: whether the score came from cache.

### Passing a request object

Use `request` when a caller already builds JSON-like payloads. Hyphenated
keys and underscored keys are both accepted.

```python
response = score_vina(
    request={
        "smiles": ["CCO", "CCN"],
        "vina-pdb-id": "8UN5",
        "vina-chain-id": "A",
        "vina-exhaustiveness": 4,
        "vina-n-poses": 3,
        "vina-seed": 42,
    },
    vina_bin="/opt/bin/vina",
    vina_cache_dir="/mnt/data0/shared/pdf2dock/vina_cache",
    max_workers=1,
)
```

### Explicit config

Use `build_vina_config` when one deployment process wants to create and
inspect provider configuration before scoring.

```python
from strbo_v1.external_interfaces import build_vina_config, score_vina

config = build_vina_config(
    {
        "vina_pdb_id": "8UN5",
        "vina_chain_id": "A",
        "vina_exhaustiveness": 4,
        "vina_n_poses": 3,
    },
    vina_bin="/opt/bin/vina",
    cache_dir="/mnt/data0/shared/pdf2dock/vina_cache",
    max_workers=1,
)

response = score_vina(["CCO", "CCN"], config=config)
```

Common Vina settings:

| Setting | Where to pass | Meaning |
|---|---|---|
| `vina_bin` | Python kwarg | Explicit Vina executable. |
| `vina_cache_dir` / `cache_dir` | Python kwarg | Docking cache and generated files. |
| `max_workers` | Python kwarg | Parallel docking workers. |
| `vina_pdb_id` | request | Receptor PDB id, default `8UN5`. |
| `vina_chain_id` | request | Receptor chain, default `A`. |
| `vina_ligand_resname` | request | Optional ligand residue name. |
| `vina_exhaustiveness` | request | Vina search exhaustiveness. |
| `vina_n_poses` | request | Vina pose count. |
| `vina_seed` | request | Vina RNG seed. |
| `vina_no_cache` | request | Disable cache when true. |

## NN Scoring SDK

`score_nn` loads the activity model and returns one structured item per
input molecule. Higher values are better for this scorer, so BO callers
usually use `minimize=False`. If `model_path` is omitted, it loads the
committed KRAS G12D model at `activity_modeling/best_g12d_model.joblib`
and the matching metadata sidecar.

### Minimal call

```python
from strbo_v1.external_interfaces import score_nn

response = score_nn(
    ["CCO", "CCN"],
)

values = {
    row["smiles"]: row["value"]
    for row in response["items"]
    if row["ok"]
}
```

Successful NN items include:

- `value`: predicted activity score.
- `details.raw_value`: raw model output before structured conversion.

Invalid SMILES or non-finite model outputs become item-level failures:

```python
for row in response["items"]:
    if not row["ok"]:
        print(row["smiles"], row["error"])
```

### Explicit config

```python
from strbo_v1.external_interfaces import build_nn_config, score_nn

config = build_nn_config(
    {"nn_on_error": "all_nan"},
    model_path="activity_modeling/best_g12d_model.joblib",
    metadata_path="activity_modeling/best_g12d_model_metadata.json",
)

response = score_nn(["CCO", "CCN"], config=config)
```

Common NN settings:

| Setting | Where to pass | Meaning |
|---|---|---|
| `model_path` | Python kwarg | Optional joblib model artifact; defaults to `activity_modeling/best_g12d_model.joblib`. |
| `metadata_path` | Python kwarg | Optional model metadata JSON. |
| `nn_on_error` | request | `all_nan` for item failures, `raise` for explicit inference errors. |

## Acquisition SDK

`evaluate_acquisition` fits GP surrogate models from historical
observations and evaluates candidate query SMILES. It returns posterior
statistics and acquisition values for every queried molecule.

For repeated queries over the same history, the lower-level
`AcquisitionEvaluator` class can be constructed once and reused. See
[Reusable evaluator](#reusable-evaluator).

### Single-objective call

```python
from strbo_v1.external_interfaces import evaluate_acquisition

history = [
    {"smiles": "CCO", "score": -7.1},
    {"smiles": "CCN", "score": -6.4},
    {"smiles": "CCC", "score": -5.9},
]

response = evaluate_acquisition(
    history=history,
    query_smiles=["CCCO", "CCNO"],
    request={
        "method": "bo-tanimoto",
        "acquisition": ["ei", "pi", "ucb"],
        "minimize": True,
        "xi": 0.01,
        "kappa": 2.0,
        "gp_fit_itersteps": 20,
        "gp_learning_rate": 0.05,
        "gp_fp_radius": 2,
        "gp_fp_n_bits": 128,
    },
    gp_device="cpu",
)

for row in response["items"]:
    print(row["smiles"], row["details"])
```

Single-objective item details contain:

```python
{
    "mean": -6.8,
    "std": 0.3,
    "variance": 0.09,
    "acquisition_ei": 0.12,
    "acquisition_pi": 0.45,
    "acquisition_ucb": -6.2,
}
```

If `acquisition` is a single string, `value` is that acquisition value.
If multiple acquisition functions are requested, `value` is the first
acquisition value in the returned details, and all requested values are
still present in `details`.

### Single objective selected from multi-objective history

Use `objective_index` when the history stores multiple scores but the
caller wants the posterior and acquisition value for one objective.

```python
response = evaluate_acquisition(
    history=[
        {"smiles": "CCO", "scores": [-7.1, 5.1]},
        {"smiles": "CCN", "scores": [-6.4, 5.8]},
        {"smiles": "CCC", "scores": [-5.9, 4.9]},
    ],
    query_smiles=["CCCO", "CCNO"],
    request={
        "objective_index": 1,
        "acquisition": "ucb",
        "minimize": [True, False],
        "kappa": 2.0,
        "gp_fit_itersteps": 20,
    },
    gp_device="cpu",
)
```

`objective_index` is zero-based. In the example above, index `1`
selects the second score column.

### Two-objective EHVI call

When history entries contain `scores` with length two and no
`objective_index` is provided, the interface fits one GP per objective
and returns expected hypervolume improvement.

```python
response = evaluate_acquisition(
    history=[
        {"smiles": "CCO", "scores": [-7.1, 5.1]},
        {"smiles": "CCN", "scores": [-6.4, 5.8]},
        {"smiles": "CCC", "scores": [-5.9, 4.9]},
    ],
    query_smiles=["CCCO", "CCNO"],
    request={
        "method": "bo-tanimoto",
        "minimize": [True, False],
        "ref_point": [0.0, 4.0],
        "ehvi_n_samples": 128,
        "gp_fit_itersteps": 20,
    },
    gp_device="cpu",
)

for row in response["items"]:
    print(row["smiles"], row["value"], row["details"]["objectives"])
```

Two-objective item details contain:

```python
{
    "objectives": [
        {"index": 0, "mean": -6.8, "std": 0.3, "variance": 0.09},
        {"index": 1, "mean": 5.4, "std": 0.2, "variance": 0.04},
    ],
    "acquisition_ehvi": 0.18,
}
```

### Three-or-more-objective call

For `n_obj >= 3`, the interface uses Chebyshev scalarization and returns
`acquisition_chebyshev`.

```python
response = evaluate_acquisition(
    history=[
        {"smiles": "CCO", "scores": [-7.1, 5.1, 0.22]},
        {"smiles": "CCN", "scores": [-6.4, 5.8, 0.31]},
        {"smiles": "CCC", "scores": [-5.9, 4.9, 0.18]},
    ],
    query_smiles=["CCCO", "CCNO"],
    request={
        "minimize": [True, False, True],
        "che_alpha": 1.0,
        "gp_fit_itersteps": 20,
    },
    gp_device="cpu",
)
```

### Acquisition settings

| Setting | Type | Applies to | Meaning |
|---|---|---|---|
| `history` | list[dict] | all | Observed molecules and scores. |
| `query_smiles` | list[str] or comma string | all | Candidate molecules to evaluate. |
| `method` | str | all | `bo-tanimoto` or `bo-strkernel`; controls GP representation. |
| `acquisition` | str or list[str] | single objective | `ei`, `pi`, `ucb`, or a list of them. |
| `objective_index` | int | multi history, single objective | Selects one score column. |
| `minimize` | bool or list[bool] | all | Optimization direction per objective. |
| `ref_point` | list[float] or comma string | two objectives | EHVI reference point. |
| `ehvi_n_samples` | int | two objectives | Monte-Carlo samples for EHVI. |
| `che_alpha` | float | three or more objectives | Simplex weight concentration. |
| `xi` | float | EI / PI | Improvement threshold. |
| `kappa` | float | UCB | Exploration weight. |
| `gp_fit_itersteps` | int | all | GP training steps. |
| `gp_learning_rate` | float | all | GP optimizer learning rate. |
| `gp_fp_radius` | int | Tanimoto GP | Morgan fingerprint radius. |
| `gp_fp_n_bits` | int | Tanimoto GP | Fingerprint bit count. |
| `gp_device` | Python kwarg | all | GP device, for example `cpu` or `cuda`. |

`evaluate_acquisition` requires at least two finite historical scores
for the target objective columns that need GP fitting.

### Reusable evaluator

Use `AcquisitionEvaluator` directly when the history and acquisition
configuration are fixed and the caller needs many candidate queries. The
GP is fit during construction, then reused for every call.

```python
from strbo_v1 import AcquisitionEvaluator, BayesianAnalogSearchConfig

config = BayesianAnalogSearchConfig(
    acquisition=("ei", "pi", "ucb"),
    minimize=True,
)

evaluator = AcquisitionEvaluator(
    history=[
        ("CCO", -7.1),
        ("CCN", -6.4),
        ("CCC", -5.9),
    ],
    config=config,
)

first_batch = evaluator(["CCCO", "CCNO"])
second_batch = evaluator(["CCCl", "CCBr"])
```

`AcquisitionEvaluator.__call__` returns a dictionary keyed by queried
SMILES:

```python
{
    "CCCO": {
        "mean": -6.8,
        "std": 0.3,
        "variance": 0.09,
        "acquisition_ei": 0.12,
        "acquisition_pi": 0.45,
        "acquisition_ucb": -6.2,
    }
}
```

For a multi-objective history, pass `objective_index` to reuse the class
for one selected score column:

```python
evaluator = AcquisitionEvaluator(
    history=[
        ("CCO", (-7.1, 5.1)),
        ("CCN", (-6.4, 5.8)),
        ("CCC", (-5.9, 4.9)),
    ],
    config=BayesianAnalogSearchConfig(acquisition="ei", minimize=(True, False)),
    objective_index=0,
)
```

## Error Handling Pattern

The SDK intentionally keeps row-level and request-level failures
separate.

Row-level failures:

```python
response = score_nn(["CCO", "not-a-smiles"], model_path="model.joblib")

ok_rows = [row for row in response["items"] if row["ok"]]
bad_rows = [row for row in response["items"] if not row["ok"]]
```

Request-level failures:

```python
try:
    response = evaluate_acquisition(
        history=[{"smiles": "CCO", "score": -7.1}],
        query_smiles=["CCN"],
        gp_device="cpu",
    )
except ValueError as exc:
    print("Invalid request:", exc)
```

Use this pattern for external services:

```python
def score_batch(smiles):
    response = score_vina(
        smiles,
        vina_bin="/opt/bin/vina",
        vina_cache_dir="/mnt/data0/shared/pdf2dock/vina_cache",
        max_workers=1,
    )
    return {
        row["smiles"]: row["value"]
        for row in response["items"]
        if row["ok"]
    }
```

## JSON Functions

The JSON functions live in `bo_api.py` and return JSON strings. They are
useful across subprocess, notebook, or HTTP boundaries.

```python
import json
import bo_api

body = json.dumps({"smiles": ["CCO", "CCN"]})

response_json = bo_api.score_vina_json(
    body,
    vina_bin="/opt/bin/vina",
    vina_cache_dir="/mnt/data0/shared/pdf2dock/vina_cache",
    vina_max_workers=1,
)

response = json.loads(response_json)
```

Available JSON helpers:

```python
bo_api.score_vina_json(
    request_json,
    vina_bin="/opt/bin/vina",
    vina_cache_dir="/mnt/data0/shared/pdf2dock/vina_cache",
    vina_max_workers=1,
)

bo_api.score_nn_json(
    request_json,
    nn_model_path="activity_modeling/best_g12d_model.joblib",
    nn_metadata_path="activity_modeling/best_g12d_model_metadata.json",
)

bo_api.evaluate_acquisition_json(
    request_json,
    gp_device="cpu",
)
```

Example acquisition JSON request:

```json
{
  "history": [
    {"smiles": "CCO", "score": -7.1},
    {"smiles": "CCN", "score": -6.4},
    {"smiles": "CCC", "score": -5.9}
  ],
  "query_smiles": ["CCCO", "CCNO"],
  "acquisition": ["ei", "pi", "ucb"],
  "minimize": true,
  "gp_fit_itersteps": 20
}
```

If the request itself is invalid or provider configuration fails, the
JSON functions return:

```json
{
  "ok": false,
  "items": [],
  "errors": [{"type": "ValueError", "message": "..."}],
  "error": "...",
  "error_type": "ValueError",
  "traceback": "..."
}
```

## HTTP Adapter

`bo_api_http.handle_request(path, request_body, **provider_kwargs)` is a
framework-neutral adapter. It returns `(status_code, json_body)`.

Routes:

- `POST /score/vina`
- `POST /score/nn`
- `POST /acquisition/evaluate`

Example:

```python
import json
from bo_api_http import handle_request

status, body = handle_request(
    "/acquisition/evaluate",
    json.dumps({
        "history": [
            {"smiles": "CCO", "score": -7.1},
            {"smiles": "CCN", "score": -6.4},
            {"smiles": "CCC", "score": -5.9}
        ],
        "query_smiles": ["CCCO"],
        "acquisition": "ei",
        "gp_fit_itersteps": 20
    }),
    gp_device="cpu",
)

payload = json.loads(body)
```

`bo_api` error responses map to HTTP `502`; unknown routes map to `404`.

## End-to-End External Loop

The intended external pattern is:

1. Score seed or candidate molecules with `score_vina` and/or `score_nn`.
2. Store successful observations in `history`.
3. Call `evaluate_acquisition` to rank the next candidate list.
4. Evaluate the selected molecules with the real scorer.
5. Append the new observations and repeat.

Minimal single-objective loop:

```python
from strbo_v1.external_interfaces import evaluate_acquisition, score_vina

history = []
seed_response = score_vina(
    ["CCO", "CCN", "CCC"],
    vina_bin="/opt/bin/vina",
    vina_cache_dir="/mnt/data0/shared/pdf2dock/vina_cache",
)

for row in seed_response["items"]:
    if row["ok"]:
        history.append({"smiles": row["smiles"], "score": row["value"]})

candidate_pool = ["CCCO", "CCNO", "CCCl"]

acq_response = evaluate_acquisition(
    history=history,
    query_smiles=candidate_pool,
    request={"acquisition": "ei", "minimize": True},
    gp_device="cpu",
)

ranked = sorted(
    [row for row in acq_response["items"] if row["ok"]],
    key=lambda row: row["value"],
    reverse=True,
)

next_smiles = [row["smiles"] for row in ranked[:2]]
```

For two objectives, store `scores` instead of `score`:

```python
history.append({
    "smiles": "CCO",
    "scores": [vina_score, nn_score],
})
```

Then call:

```python
acq_response = evaluate_acquisition(
    history=history,
    query_smiles=candidate_pool,
    request={
        "minimize": [True, False],
        "ref_point": [0.0, 4.0],
        "ehvi_n_samples": 128,
    },
    gp_device="cpu",
)
```
