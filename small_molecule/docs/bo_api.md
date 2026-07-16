# `bo_api.py` — JSON-in/JSON-out API

`bo_api.py` exposes the `strbo_v1` search library to non-CLI callers
through two JSON-only functions:

* **`run_search_trajectory(req)`** — runs one full trajectory
  (analog generation + scoring + BO loop) end-to-end and returns
  the full record. Equivalent to `python run_search.py ...` but
  over JSON. Supports pure BO and LDM methods.
* **`recommend_next_smiles(req)`** — one-shot "advisor" step:
  given a pool and a history, returns the top-k SMILES to evaluate
  next. The caller drives the surrounding loop (analog generator,
  black-box scorer). Useful when the scorer is a remote API, a
  laboratory experiment, or any other process that doesn't fit
  a Python `Callable`. Supports pure BO and LDM-assisted one-step
  recommendation.
* **`score_vina_json(req)`**, **`score_nn_json(req)`**, and
  **`evaluate_acquisition_json(req)`** provide structured external
  scoring/acquisition utilities. See
  [`docs/external_interfaces.md`](external_interfaces.md).

Both functions return JSON strings (never Python objects), so the
boundary is safe across subprocess / HTTP / notebook contexts.

> **See also**:
> [`docs/bo.md`](bo.md) for the algorithm details (kernel choice,
> acquisition functions, multi-objective dispatch); the rest of this
> document focuses on the API surface.

## Two-layer setting model

Settings are split into three layers. The split is enforced
**inside `bo_api.py`** (`run_search.config_from_dict` is unaware
of it and still accepts every CLI flag):

* **Provider's setting** — Python kwargs only. Deployment wiring
  for binaries, models, GPU device, parallelism, and cache
  directory. The JSON body **never participates**: any JSON value
  for these keys is silently ignored (a single DEBUG log line is
  emitted per ignored key). Precedence is strictly
  `Python kwarg > env var > hard-coded default`.
* **bo_api's defaults** — :data:`bo_api.DEFAULT`, a module-level
  flat dict of argparse-dest-name → value that mirrors
  `run_search.sh`. Applied when the user's JSON omits a key
  entirely. The user's value (incl. `null`) always wins.
* **run_search.py argparse defaults** — applied only when
  `run_search.py` is invoked directly via CLI; bo_api users
  always see `bo_api.DEFAULT` first.

Precedence (highest → lowest):

1. Provider's setting kwarg (for provider-setting keys)
2. User's JSON value (for user's-request keys; `null` = explicit value)
3. `bo_api.DEFAULT` (when the user omits a user's-request key entirely)
4. `run_search.py` argparse default (only applies when CLI is invoked
   directly; bo_api always wins above this layer)

The 13 **provider-setting** keys are:

```
vina-bin            → kwarg vina_bin
vina-cache-dir      → kwarg vina_cache_dir
vina-max-workers    → kwarg vina_max_workers
gp-device           → kwarg gp_device
reasyn-repo         → kwarg reasyn_repo
reasyn-python-bin   → kwarg reasyn_python_bin
reasyn-model-path   → kwarg reasyn_model_path
reasyn-devices      → kwarg reasyn_devices
nn-model-path       → kwarg nn_model_path
nn-metadata-path    → kwarg nn_metadata_path
llm-model           → kwarg llm_model
llm-base-url        → kwarg llm_base_url
llm-api-key         → kwarg llm_api_key
```

The 13 keys are deliberately absent from `bo_api.DEFAULT`.
Their fallbacks come from argparse defaults (trajectory path)
and from hardcoded values (advisor path). See §1.1.1 for the
full kwarg / env / default table, and §2 for the advisor's
`gp_device` plus LDM LLM kwargs.

## Table of contents

- [§1. `run_search_trajectory`](#1-run_search_trajectory)
  - [1.1 Request schema](#11-request-schema)
  - [1.2 Response schema](#12-response-schema)
  - [1.3 Error response](#13-error-response)
  - [1.4 Worked example](#14-worked-example)
- [§2. `recommend_next_smiles`](#2-recommend_next_smiles)
  - [2.1 Request schema](#21-request-schema)
  - [2.2 Response schema](#22-response-schema)
  - [2.3 Error response](#23-error-response)
  - [2.4 Worked examples](#24-worked-examples)
- [§3. Method dispatch table](#3-method-dispatch-table)
- [§4. `minimize` and `ref_point` semantics](#4-minimize-and-ref_point-semantics)
- [§5. Common error types](#5-common-error-types)
- [§6. End-to-end example: external black-box loop](#6-end-to-end-example-external-black-box-loop)
- [§7. Quick reference](#7-quick-reference)

---

## §1. `run_search_trajectory`

Signature: `run_search_trajectory(request_json: str) -> str`.

This is the JSON equivalent of running `python run_search.py
--method <m> --seed <s> --objective <o> ... --output <path>`. It
runs one trajectory with the given configuration and returns the
same `{"config", "history"}` JSON that the CLI writes to disk,
**plus a `summary` field** with the best-so-far / hypervolume /
per-objective curve so callers don't have to recompute it.

### 1.1 Request schema

The request is a JSON object whose keys are the long-form CLI
flag names (with hyphens, e.g. `"num-evaluations"`) or their
underscore-attribute equivalents (e.g. `"num_evaluations"`).
Both forms are accepted. Values are passed through to the
underlying argparse parser.

| Key (hyphen OR underscore) | Type | Default | Notes |
|---|---|---|---|
| `method` | string | (required) | One of `"random"`, `"random-best"`, `"bo-tanimoto"`, `"bo-strkernel"`, `"bo-tanimoto-ldm"`, `"bo-strkernel-ldm"`. |
| `seed` | int | (required) | RNG seed. |
| `seed-smiles` | string | `"CCO,CCN,CCC"` | Comma-separated SMILES **or** a path to an existing file (one SMILES per line). Each entry is RDKit-validated and canonicalized. |
| `num-evaluations` | int | `80` | Total scorer evaluations. |
| `batch-size` | int | `5` | Candidates per BO round (or per random-search round). |
| `init-size` | int | `10` | BO initialization size (warm-up + init). Ignored for random methods. |
| `pool-min-size` | int | `9` | Random-search pool refill trigger. |
| `pool-max-size` | int or `null` | `18` | Random-search pool FIFO cap (`null` = unbounded). |
| `acquisition` | string | `"ei"` | One of `"ei"`, `"ucb"`, `"pi"`. Single-objective only. |
| `xi` | float | `0.01` | EI / PI improvement threshold. |
| `kappa` | float | `2.0` | UCB exploration weight. |
| `acq-budget` | int or `null` | `null` | Optional subsample size for the BO acquisition step. |
| `max-pool-size` | int or `null` | `1024` | BO pool FIFO cap. |
| `smiles-max-len` | int | `100` | SMILES length cap (also drives GP string-kernel padding). |
| `ehvi-n-samples` | int | `128` | Monte-Carlo samples per candidate in 2-objective EHVI. |
| `che-alpha` | float | `1.0` | Beta concentration for simplex-weight sampling (Chebyshev-ParEGO, `n_obj >= 3`). |
| `ref-point` | string or `null` | `null` | Comma-separated reference point for HV/EHVI (multi-objective only). |
| `objective` | string | `"vina+nn"` | `+`-joined backend names. Per-backend `minimize` is hard-coded. |
| `gp-fit-itersteps` | int | `100` | GP Adam iterations per fit. |
| `gp-learning-rate` | float | `0.05` | GP Adam learning rate. |
| `gp-min-jitter` | float | `1e-6` | Cholesky-jitter ladder minimum. |
| `gp-max-jitter` | float | `1e-1` | Cholesky-jitter ladder maximum. |
| `gp-standardize-y` | bool | `true` | Standardize targets before GP fit. |
| `gp-fp-radius` | int | `2` | Morgan fingerprint radius (Tanimoto kernel). |
| `gp-fp-n-bits` | int | `2048` | Morgan fingerprint bit count. |
| `vina-pdb-id` | string | `"8UN5"` | PDB id to fetch. |
| `vina-chain-id` | string | `"A"` | Receptor chain. |
| `vina-ligand-resname` | string | `null` | Ligand resname in the receptor PDB. |
| `vina-exhaustiveness` | int | `4` | Vina exhaustiveness. |
| `vina-n-poses` | int | `3` | Vina `--num_modes`. |
| `vina-seed` | int | `42` | Vina RNG seed. |
| `vina-allow-debug-receptor` | bool | `false` | Allow non-standard receptor residue names. |
| `vina-no-cache` | bool | `false` | Disable the disk cache. |
| `reasyn-search-width` | int | `5` | ReaSyn search width. |
| `reasyn-exhaustiveness` | int | `8` | ReaSyn exhaustiveness. |
| `reasyn-num-cycles` | int | `3` | ReaSyn cycles. |
| `reasyn-num-editflow-samples` | int | `10` | ReaSyn edit-flow samples. |
| `reasyn-num-editflow-steps` | int | `30` | ReaSyn edit-flow steps. |
| `reasyn-time-limit` | int | `20` | ReaSyn per-molecule time limit (seconds). |
| `reasyn-num-workers-per-gpu` | int | `1` | ReaSyn workers per GPU. |
| `reasyn-filter-sim` | float | `0.8` | ReaSyn similarity filter. |
| `reasyn-no-canonicalize` | bool | `false` | Disable ReaSyn output canonicalization. |
| `llm-trajectory-dir` | string | `""` | Optional sidecar directory for LDM trajectory JSONs. The main API response embeds the trajectory either way for `bo-*-ldm`. |
| `ldm-sys-prompt` | string | `""` | LDM system-prompt supplement. Existing file path = read file contents; otherwise inline text. |
| `output` | string | `"output/bo"` | Output directory or `.json` file path. (Ignored — the response is the JSON itself.) |
| `verbose` | bool | `false` | Verbose logging. |
| `log-level` | string | `"INFO"` | One of `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`. |

> **Note:** `vina-max-workers`, `vina-bin`, `vina-cache-dir`,
> `gp-device`, `reasyn-repo`, `reasyn-python-bin`, `reasyn-model-path`,
> `reasyn-devices`, `nn-model-path`, `nn-metadata-path`,
> `llm-model`, `llm-base-url`, and `llm-api-key` are the
> **13 provider-setting keys**. They are deliberately absent from this
> schema table — see §1.1.1 for the kwarg surface.

#### 1.1.1 Provider's setting (Python kwargs)

The 13 deployment-side settings are settable **only** via Python
keyword arguments on `run_search_trajectory`. The JSON body never
participates: any value passed for these keys is silently ignored
(a single DEBUG log line is emitted per ignored key). They are
also deliberately absent from `bo_api.DEFAULT`.

| Kwarg | Env var fallback | Hard-coded default | What it controls |
|---|---|---|---|
| `vina_bin` | `VINA_BIN` | `<repo>/../bin/vina` | AutoDock Vina binary. |
| `vina_cache_dir` | — | `output/bo/vina_cache/` | Disk cache directory. |
| `vina_max_workers` | — | `1` | Parallel Vina workers. |
| `gp_device` | — | `"cuda"` | GP device (`"cuda"` / `"cuda:0"` / `"cpu"`). |
| `reasyn_repo` | `REASYN_HOME` / `REASYN_REPO` | — (required) | ReaSyn checkout. |
| `reasyn_python_bin` | `REASYN_PYTHON` / `REASYN_BIN` | — | Python interpreter inside ReaSyn env. |
| `reasyn_model_path` | `REASYN_MODEL_PATH` | AR+EB checkpoints under `data/trained_model/` | Comma-separated checkpoint paths. |
| `reasyn_devices` | — | `"1,2"` | Comma-separated GPU ids for ReaSyn. |
| `nn_model_path` | — | `activity_modeling/best_g12d_model.joblib` | NN G12D pIC50 model. |
| `nn_metadata_path` | — | model-stem metadata | NN sidecar metadata JSON. |
| `llm_model` | — | `DeepSeek-V4-Flash` | LDM chat model name. |
| `llm_base_url` | `LLM_BASE_URL` | — (required for LDM methods) | OpenAI-compatible endpoint. |
| `llm_api_key` | `LLM_API_KEY` | — (required for LDM methods) | OpenAI-compatible API key. |

**Precedence for provider settings:** `Python kwarg > env var > hard-coded default`. The JSON body never participates; `bo_api.DEFAULT` does not contain these keys either.

**Signature:**

```python
bo_api.run_search_trajectory(
    request_json: str,
    *,
    vina_bin: Optional[str] = None,
    vina_cache_dir: Optional[str] = None,
    vina_max_workers: Optional[int] = None,
    reasyn_repo: Optional[str] = None,
    reasyn_python_bin: Optional[str] = None,
    reasyn_model_path: Optional[str] = None,
    reasyn_devices: Optional[str] = None,
    gp_device: Optional[str] = None,
    nn_model_path: Optional[str] = None,
    nn_metadata_path: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
) -> str
```

All kwargs are **keyword-only**; `None` means "no override" and
the env var / default chain applies.

**Example — provider-side wiring at the call site:**

```python
import json
import bo_api

response = bo_api.run_search_trajectory(
    json.dumps({
        "method": "bo-tanimoto",
        "seed": 0,
        "seed-smiles": "CCO,CCN,CCC",
        "num-evaluations": 10,
        "batch-size": 2,
        "objective": "vina",
    }),
    gp_device="cuda:0",
    vina_bin="/opt/vina/bin/vina",
    vina_cache_dir="/var/cache/vina",
    reasyn_repo="../ReaSyn",
    reasyn_python_bin="/path/to/conda/envs/reasyn/bin/python",
    reasyn_devices="0",
    nn_model_path="/models/best.joblib",
    nn_metadata_path="/models/best_metadata.json",
)
```

**Note:** the `config` echo in the response reflects the kwarg
values (since they are injected into the args namespace before
the echo is built), so e.g.
`response["config"]["reasyn"]["reasyn_repo"]` will be
`"../ReaSyn"` in this example. The JSON body never has this
information.

**`recommend_next_smiles` does not accept the Vina/ReaSyn/NN
kwargs** — it is an advisor step that does not invoke the scorers.
It does accept `gp_device`, plus `llm_model`, `llm_base_url`, and
`llm_api_key` for `bo-*-ldm` one-step recommendation (see §2).

### 1.2 Response schema

Success (single-objective):

```json
{
  "config": {
    "method": "bo-tanimoto",
    "seed": 0,
    "seed_smiles": ["CCO", "CCN", "CCC"],
"num_evaluations": 80,
        "batch_size": 5,
        "init_size": 12,
        "acquisition": "ei",
        "xi": 0.01,
        "kappa": 2.0,
        "minimize": true,
        "acq_budget": 500,
        "max_pool_size": 1024,
        "pool_min_size": 9,
        "pool_max_size": 18,
        "smiles_max_len": 100,
        "objective": "vina+nn",
    "n_objectives": 1,
    "objective_parts": ["vina"],
    "ehvi_n_samples": 128,
    "che_alpha": 1.0,
    "gp": {
      "gp_device": "cuda",
      "gp_fit_itersteps": 50,
      "gp_learning_rate": 0.1,
      "gp_min_jitter": 1e-6,
      "gp_max_jitter": 0.1,
      "gp_standardize_y": true,
      "gp_fp_radius": 2,
      "gp_fp_n_bits": 2048,
      "impl": "fingerprint+tanimoto",
      "smiles_maxlen": 50
    },
    "vina": { ... full Vina config echo ... },
    "reasyn": { ... full ReaSyn config echo ... }
  },
  "history": [
    {"index": 0, "smiles": "CCN", "score": -2.5},
    {"index": 1, "smiles": "CCO", "score": -1.5},
    ...
  ],
  "summary": {
    "bsf": [-2.5, -2.5, -2.5, ...]   // n_obj == 1
  }
}
```

Success (multi-objective, `n_obj == 2`):

```json
{
  "config": {
    ...
    "objective": "vina+nn",
    "n_objectives": 2,
    "objective_parts": ["vina", "nn"],
    "minimize": [true, false],
    "ref_point": [0.0, 5.0],
    ...
  },
  "history": [
    {"index": 0, "smiles": "CCO", "scores": [-7.5, 5.2]},
    ...
  ],
  "summary": {
    "hypervolume": [0.0, 0.0, 1.2, ...]   // n_obj == 2
  }
}
```

Success (multi-objective, `n_obj >= 3`):

```json
{
  "config": {
    ...
    "n_objectives": 3,
    "objective_parts": ["vina", "nn", "mock"],
    "minimize": [true, false, true],
    ...
  },
  "history": [
    {"index": 0, "smiles": "CCO", "scores": [-7.5, 5.2, -0.3]},
    ...
  ],
  "summary": {
    "bsf_per_objective": [[-7.5, -7.5, ...], [5.2, 5.2, ...], [-0.3, -0.3, ...]]
  }
}
```

The `summary` field uses:
- `bsf` for `n_obj == 1` (lower is better if `minimize=true`, higher if `false`)
- `hypervolume` for `n_obj == 2` (cumulative HV w.r.t. `ref_point`)
- `bsf_per_objective` for `n_obj >= 3` (graceful fallback since HV is not implemented in 3D+)

For `bo-tanimoto-ldm` and `bo-strkernel-ldm`, the success response
also includes:

```json
{
  "llm_trajectory": {
    "status": "completed",
    "rounds": [
      {
        "llm_interactions": {
          "stage_a1": { "...": "..." },
          "stage_a2": { "...": "..." },
          "stage_b": { "...": "..." }
        }
      }
    ]
  }
}
```

This is the same trajectory object written by the CLI sidecar and is
embedded so HTTP/notebook callers can debug the LDM's pool actions,
analog review, and BO-suggestion review without reading files.

### 1.3 Error response

```json
{
  "error": "argument --method: invalid choice: 'foo' (choose from random, random-best, bo-tanimoto, bo-strkernel)",
  "error_type": "ValueError",
  "traceback": "Traceback (most recent call last):\n  File \"bo_api.py\", line 42, in run_search_trajectory\n    ...\nValueError: ..."
}
```

The `traceback` is always included to make notebook / web-service
debugging easy. The `error` field is a one-line summary suitable
for logging or showing to a user.

### 1.4 Worked example

```python
import json
import bo_api

# CPU-only smoke run with the mock scorer
request = {
    "method": "random",
    "seed": 0,
    "seed-smiles": "CCO,CCN,CCC",
    "num-evaluations": 8,
    "batch-size": 2,
    "objective": "mock",
}
response_str = bo_api.run_search_trajectory(json.dumps(request))
response = json.loads(response_str)

if "error" in response:
    raise RuntimeError(response["error"])

print("history length:", len(response["history"]))
print("bsf curve:", response["summary"]["bsf"])
print("first entry:", response["history"][0])
```

The CLI equivalent (writes the same JSON to disk):

```bash
python run_search.py --method random --seed 0 \
    --seed-smiles CCO,CCN,CCC --num-evaluations 8 \
    --batch-size 2 --objective mock \
    --output output/api_smoke.json
```

---

## §2. `recommend_next_smiles`

Signature: `recommend_next_smiles(request_json: str, *, gp_device: Optional[str] = None, llm_model: Optional[str] = None, llm_base_url: Optional[str] = None, llm_api_key: Optional[str] = None) -> str`.

Pure advisor step (one BO round). The caller manages the
surrounding loop: they own the black-box scorer, the analog
generator, and the history. This function answers "given what we
have already evaluated, which SMILES from this pool should we
evaluate next?" using the same algorithm dispatch as the in-loop
[`bayesian_analog_search`](bo.md#22-the-bo-loop). For
`bo-tanimoto-ldm` / `bo-strkernel-ldm`, the one-step call first
runs LDM Stage A1/A2 pool review, then BO acquisition, then LDM
Stage B suggestion review. It still does **not** score the final
recommendations; the caller's black box remains responsible for
evaluation and history updates.

This is the right entry point when the black-box scorer is:

- a remote HTTP API (cloud docking, lab experiment, etc.)
- a process that doesn't fit a Python `Callable`
- a custom chemistry tool with its own pool generator

**Provider's setting.** The advisor does not invoke Vina, ReaSyn,
or the NN scorer — those deployment knobs are not applicable. The
provider-side settings that do apply are `gp_device`, plus
`llm_model`, `llm_base_url`, and `llm_api_key` for `bo-*-ldm`.
JSON values for provider settings are silently dropped; pass them
as Python kwargs. `llm_base_url` / `llm_api_key` fall back to
`LLM_BASE_URL` / `LLM_API_KEY`.

### 2.1 Request schema

```json
{
  "method": "bo-tanimoto" | "bo-strkernel" | "bo-tanimoto-ldm" | "bo-strkernel-ldm" | "random" | "random-best",
  "pool": ["CCO", "CCN", "CCC", ...],
  "history": [
    {"smiles": "CCO", "score": -7.5},
    {"smiles": "CCN", "score": -6.2}
  ],
  "batch_size": 3,
  "minimize": true,
  "ref_point": null,
  "ehvi_n_samples": 128,
  "che_alpha": 1.0,
  "acq_budget": null,
  "acquisition": "ei",
  "xi": 0.01,
  "kappa": 2.0,
  "gp_device": "cuda:0",
  "gp_fit_itersteps": 100,
  "gp_learning_rate": 0.05,
  "gp_min_jitter": 1e-6,
  "gp_max_jitter": 1e-1,
  "gp_standardize_y": true,
  "gp_fp_radius": 2,
  "gp_fp_n_bits": 2048,
  "seed": 0,
  "ldm_sys_prompt": "",
  "analog_pool": {"CCO": ["CCCO", "CCCN"]},
  "llm_max_retries": 3,
  "llm_use_rdkit": true
}
```

| Key | Type | Default | Notes |
|---|---|---|---|
| `method` | string | (required) | One of the 6 methods, including `bo-tanimoto-ldm` and `bo-strkernel-ldm`. |
| `pool` | list of strings | `[]` | Candidate SMILES. May be empty. |
| `history` | list of objects | `[]` | Prior `(smiles, score)` evaluations. |
| `batch_size` | int | `5` | How many to pick. |
| `minimize` | bool or list[bool] | `true` | Per-objective "smaller is better" flag. |
| `ref_point` | list[float] or `null` | `null` | Reference point for 2-obj EHVI. |
| `ehvi_n_samples` | int | `128` | Monte-Carlo samples per candidate (2-obj). |
| `che_alpha` | float | `1.0` | Beta concentration (3+ obj). |
| `acq_budget` | int or `null` | `null` | Pool subsample size before GP scoring. |
| `acquisition` | string | `"ei"` | One of `"ei"`, `"pi"`, `"ucb"` (1-obj only). |
| `xi` | float | `0.01` | EI/PI threshold. |
| `kappa` | float | `2.0` | UCB exploration weight. |
| `gp_device` | string | (provider's setting; `bo_api.DEFAULT` does **not** contain it) | One of `"cuda"` / `"cuda:0"` / `"cpu"`. JSON values are silently dropped; use the Python `gp_device` kwarg. |
| `gp_fit_itersteps` | int | `100` | GP Adam iterations per fit. |
| `gp_learning_rate` | float | `0.05` | GP Adam learning rate. |
| `gp_min_jitter` | float | `1e-6` | Cholesky-jitter ladder minimum. |
| `gp_max_jitter` | float | `1e-1` | Cholesky-jitter ladder maximum. |
| `gp_standardize_y` | bool | `true` | Standardize targets before GP fit. |
| `gp_fp_radius` | int | `2` | Morgan fingerprint radius (Tanimoto kernel). |
| `gp_fp_n_bits` | int | `2048` | Morgan fingerprint bit count. |
| `seed` | int or `null` | `null` | RNG seed (reproducible acq_budget subsample). |
| `ldm_sys_prompt` | string | `""` | LDM-only system-prompt supplement. Existing file path = read file contents; otherwise inline text. |
| `analog_pool` | dict or list | `null` | Optional JSON-backed analog provider for LDM Stage A1/A2. Dict form maps seed SMILES to candidate analogs; list form is used for every seed. |
| `llm_max_retries` | int | `3` | LDM-only retry count per LLM stage. |
| `llm_use_rdkit` | bool | `true` | LDM-only semantic validation switch. |

**History entry format** (matches the JSON schema of `run_search.py`):

- Single-objective: `{"smiles": "CCO", "score": -7.5}` (a number; failed evals use `null`)
- Multi-objective: `{"smiles": "CCO", "scores": [-7.5, 5.2]}` (a list; any element can be `null`)

The number of objectives is inferred from the first entry with a
`"scores"` field. An empty history defaults to `n_obj = 1`.

### 2.2 Response schema

```json
{
  "recommendations": ["CCO", "CCN", "CCC"],
  "method": "bo-tanimoto",
  "n_history": 5,
  "pool_size": 100,
  "pool_size_after_ldm": 104,
  "acquisition_values": [0.85, 0.72, 0.68],
  "n_objectives": 1,
  "llm": {
    "stage_a1": { "...": "..." },
    "stage_a2": { "...": "..." },
    "stage_b": { "...": "..." },
    "bo_suggestions": []
  }
}
```

| Field | Type | Notes |
|---|---|---|
| `recommendations` | list of strings | Top-`batch_size` SMILES, in order of decreasing acquisition. |
| `method` | string | Echo of the input `method`. |
| `n_history` | int | Number of entries in the request `history`. |
| `pool_size` | int | Number of SMILES in the request `pool` (after filtering). |
| `pool_size_after_ldm` | int | LDM methods only. Candidate-pool size after Stage A1/A2 pool edits and before BO acquisition. |
| `acquisition_values` | list of floats | The corresponding acquisition values. Empty list `[]` for `random` / `random-best` (uniform-random pick has no value). For `n_obj >= 3` (Chebyshev) the values are *smaller = better* (the opposite of the 1-obj / 2-obj direction). |
| `n_objectives` | int | `1`, `2`, or `3+`. |
| `llm` | object | LDM methods only. Contains Stage A1/A2/B attempts, fallback flags, final blocks, BO suggestions, final candidates, overrides, and the post-A1 pool. |

### 2.3 Error response

Same format as API 1 (`{"error", "error_type", "traceback"}`).
Common failure modes are listed in [§5](#5-common-error-types).

### 2.4 Worked examples

**Single-objective BO (Tanimoto kernel):**

```python
import json
import bo_api

request = {
    "method": "bo-tanimoto",
    "pool": ["CCO", "CCN", "CCC", "C1CCCCC1", "c1ccccc1", "CCCCC"],
    "history": [
        {"smiles": "CCO",   "score": -7.5},
        {"smiles": "CCN",   "score": -6.2},
        {"smiles": "CCC",   "score": -5.8},
        {"smiles": "CCCC",  "score": -7.0},
    ],
    "batch_size": 2,
    "minimize": True,
    "gp_fit_itersteps": 20,
    "gp_fp_n_bits": 128,
    "seed": 0,
}
response = json.loads(bo_api.recommend_next_smiles(
    json.dumps(request),
    gp_device="cpu",  # provider's setting; kwarg-only
))
print(response["recommendations"])
# e.g. ['C1CCCCC1', 'CCCCC']
print(response["acquisition_values"])
# e.g. [0.035, 0.033]   (higher = better)
```

**Multi-objective EHVI (2 objectives):**

```python
request = {
    "method": "bo-tanimoto",
    "pool": ["CCO", "CCN", "CCC", "C1CCCCC1", "c1ccccc1", "CCCCC"],
    "history": [
        {"smiles": "CCO",  "scores": [-7.5, 5.2]},
        {"smiles": "CCN",  "scores": [-6.2, 5.5]},
        {"smiles": "CCC",  "scores": [-5.8, 4.8]},
        {"smiles": "CCCC", "scores": [-7.0, 5.1]},
    ],
    "batch_size": 2,
    "minimize": [True, False],
    "ref_point": [0.0, 5.0],
    "ehvi_n_samples": 64,
    "gp_fit_itersteps": 20,
    "gp_fp_n_bits": 128,
    "seed": 0,
}
response = json.loads(bo_api.recommend_next_smiles(json.dumps(request)))
print(response["recommendations"], response["acquisition_values"])
# e.g. ['c1ccccc1', 'C1CCCCC1']  [0.124, 0.085]
```

**Random method:**

```python
request = {
    "method": "random",
    "pool": ["CCO", "CCN", "CCC", "C1CCCCC1", "c1ccccc1"],
    "history": [{"smiles": "CCO", "score": -5.0}],
    "batch_size": 3,
    "seed": 42,
}
response = json.loads(bo_api.recommend_next_smiles(json.dumps(request)))
print(response["recommendations"])
print(response["acquisition_values"])  # [] — random has no value
```

Note: for `random-best`, the "best" strategy only affects the
*expansion* (refill) target — which pool member to expand via
analog generation — not the *evaluation* pick. Both methods use
the same uniform-random advisor for the per-round scoring
decision.

**LDM-assisted one-step recommendation:**

```python
request = {
    "method": "bo-tanimoto-ldm",
    "pool": ["CCO", "CCN", "CCC", "CCCC", "CCCO"],
    "history": [
        {"smiles": "CCO", "score": -1.0},
        {"smiles": "CCN", "score": -1.2},
        {"smiles": "CCC", "score": -1.5},
    ],
    "batch_size": 2,
    "minimize": True,
    "gp_fit_itersteps": 20,
    "gp_fp_n_bits": 128,
    "pool_min_size": 2,
    "ldm_sys_prompt": "Prefer compact analogs; keep only valid KRAS-like motifs.",
    "analog_pool": {
        "CCCC": ["CCCCO", "CCCCN"]
    },
    "seed": 0,
}
response = json.loads(bo_api.recommend_next_smiles(
    json.dumps(request),
    gp_device="cpu",
    llm_base_url="https://llm.example/v1",
    llm_api_key="...",
))
print(response["recommendations"])
print(response["llm"]["stage_b"]["final_candidates"])
```

`analog_pool` is optional. If omitted, Stage A1 can still propose,
reject, or noop, but LDM `analog` actions have no backing generator in
the JSON-only API. In production trajectory runs, `run_search_trajectory`
uses ReaSyn as the analog generator.

---

## §3. Method dispatch table

| `method` | n_obj | Algorithm | Acquisition value direction | Notes |
|---|---|---|---|---|
| `bo-tanimoto` | 1 | EI / PI / UCB on the Tanimoto-kernel GP | higher = better | GP featurization = Morgan fingerprint (Tanimoto). |
| `bo-strkernel` | 1 | EI / PI / UCB on the string-kernel GP | higher = better | GP featurization = raw SMILES (subsequence kernel). |
| `bo-tanimoto-ldm` | 1 | LDM Stage A1/A2 + Tanimoto-kernel BO + LDM Stage B | higher = better | Same GP/acquisition as `bo-tanimoto`; LDM can edit the pool and review BO suggestions. |
| `bo-strkernel-ldm` | 1 | LDM Stage A1/A2 + string-kernel BO + LDM Stage B | higher = better | Same GP/acquisition as `bo-strkernel`; LDM can edit the pool and review BO suggestions. |
| `bo-tanimoto` | 2 | Expected Hypervolume Improvement (Monte Carlo) | higher = better | `ehvi_n_samples` Monte-Carlo draws per candidate. |
| `bo-tanimoto` | ≥ 3 | Chebyshev ParEGO scalarization | **smaller = better** | `che_alpha` controls the simplex-weight Beta distribution. |
| `bo-*-ldm` | 2 | LDM + EHVI | higher = better | Native multi-objective; no single-objective collapse. |
| `bo-*-ldm` | ≥ 3 | LDM + Chebyshev ParEGO scalarization | **smaller = better** | Native multi-objective; no single-objective collapse. |
| `random` | any | Uniform random | n/a | `acquisition_values` is always `[]`. |
| `random-best` | any | Uniform random (per-round); Chebyshev best (refill) | n/a | Same per-round pick as `random`; the "best" affects only which pool member is fed to the analog generator. |

GP training is shared between BO methods (same `gp_config`,
`fit_n_itersteps`, `learning_rate`, `min_jitter`/`max_jitter`,
`standardize_y`); the only difference is `gp.impl` and the
featurization (Tanimoto or string-kernel).

---

## §4. `minimize` and `ref_point` semantics

`minimize` is **hard-coded by backend** (the API mirrors
`run_search.py`):

| Backend | `minimize` | Units / meaning |
|---|---|---|
| `vina` | `true` | kcal/mol: more negative = stronger binding. |
| `nn` | `false` | pIC50: higher = more potent. |
| `mock` | `true` | Mock-scorer neutral. |

For multi-objective (`vina+nn` etc.), `minimize` is the
positionally-aligned tuple of per-objective directions. The API
accepts either a single bool (broadcast to all objectives) or a
list of bools of length `n_obj`. A length mismatch raises
`ValueError`.

`ref_point` is the HV/EHVI reference point. It is **only
honored for `n_obj == 2`**:

- `n_obj == 1`: ignored (HV is not defined).
- `n_obj == 2`: when the request omits `ref_point`, a conservative
  default of `[0.0, 0.0]` is used. Pass an explicit list to use
  a per-domain value.
- `n_obj >= 3`: ignored (Chebyshev ParEGO uses per-objective
  ideal points instead).

The per-backend `DEFAULT_REF` registry used by `run_search.py`
(see [§2.6.2 of `bo.md`](bo.md#262-reference-point-per-backend-default))
is **not** consulted by the advisor step — the advisor is a
single-shot call and the caller is expected to supply the
reference point they want.

---

## §5. Common error types

| `error_type` | When | Recovery |
|---|---|---|
| `ValueError` | Invalid request key (not a known flag) | Use a key from the schema above, or pass `null` to use the parser default. |
| `ValueError` | `minimize` length != `n_obj` (multi-obj) | Pass a list of length `n_obj` (e.g. `[true, false]` for `vina+nn`). |
| `ValueError` | `ref_point` length != 2 (n_obj=2) | Pass a 2-element list (e.g. `[0.0, 5.0]`). |
| `ValueError` | `batch_size < 1` | Pass a positive integer. |
| `ValueError` | Inconsistent history (`score` mixed with `scores`) | All entries must use either `score` (n_obj=1) or `scores` (n_obj>=2). |
| `SystemExit` (caught and re-raised as `ValueError`) | `argparse.error()` (e.g. unknown method) | Use a valid `method` value. |
| `RuntimeError` (from inner BO / GP) | GP fit failed at every jitter attempt | Loosen `gp-max-jitter`, reduce `gp-fit-itersteps`, or use a smaller pool (`acq_budget`). |
| `ValueError` | LDM method called without `llm_base_url` / `llm_api_key` kwarg or env fallback | Pass `llm_base_url` and `llm_api_key` kwargs, or set `LLM_BASE_URL` and `LLM_API_KEY`. |
| (no error raised) | Provider-setting key passed in JSON body | Silently ignored — use the Python kwarg instead. A DEBUG log line `bo_api: ignoring JSON provider-setting ...` is emitted. |

All error responses include `traceback` for debugging; the
`error` field is the single-line message.

---

## §6. End-to-end example: external black-box loop

A typical use case: a chemistry team has a proprietary remote
docking service that's not a Python `Callable`. They want to
drive BO over a pool of candidate molecules using
`recommend_next_smiles`, and let their service handle the
scoring.

```python
import json
import time
import requests  # or whatever calls the remote API
import bo_api

def remote_dock(smiles_list):
    """Call the remote docking service. Returns one score per input."""
    response = requests.post(
        "https://docking.example.com/api/score",
        json={"smiles": smiles_list},
    )
    response.raise_for_status()
    return response.json()["scores"]  # list[float], aligned with input

# 1. Initial seed SMILES.
seed = ["CCO", "CCN", "CCC"]
history = [{"smiles": s, "score": v} for s, v in zip(seed, remote_dock(seed))]

# 2. Initial candidate pool (from an external analog generator).
pool = requests.post("https://docking.example.com/api/expand",
                     json={"seeds": seed, "n": 50}).json()["smiles"]

# 3. BO loop.
N_ROUNDS = 5
BATCH = 3
for round_idx in range(N_ROUNDS):
    rec_response = json.loads(bo_api.recommend_next_smiles(json.dumps({
        "method": "bo-tanimoto",
        "pool": pool,
        "history": history,
        "batch_size": BATCH,
        "minimize": True,
        "gp": {"device": "cuda:0"},
    })))
    picks = rec_response["recommendations"]
    scores = remote_dock(picks)
    history.extend({"smiles": s, "score": v} for s, v in zip(picks, scores))
    pool = [s for s in pool if s not in picks]  # remove picks before next expand
    print(f"round {round_idx}: picks={picks} scores={scores}")

# 4. Best so far.
best = min(history, key=lambda h: h["score"] if h["score"] is not None else float("inf"))
print(f"best: {best['smiles']} = {best['score']}")
```

This pattern is the recommended way to use `bo_api.py` when
the black-box is a remote API or otherwise can't be wrapped as
a Python `Callable`.

---

## §7. Quick reference

| API | Use when | Input | Output |
|---|---|---|---|
| `run_search_trajectory` | You want the same thing as `python run_search.py ...` but over JSON (web service, notebook, batch driver). | Full `run_search.py` config dict (see §1.1). | `{"config", "history", "summary"}` plus `{"llm_trajectory"}` for LDM methods (see §1.2). |
| `recommend_next_smiles` | You have your own loop, your own black-box (remote API, lab experiment, custom chemistry tool), and you just need the algorithm to pick the next batch. | `{"method", "pool", "history", "batch_size", ...}` (see §2.1). | `{"recommendations", "method", "n_history", "pool_size", "acquisition_values", "n_objectives"}` plus `{"llm"}` for LDM methods (see §2.2). |

The external utility APIs `score_vina_json`, `score_nn_json`, and
`evaluate_acquisition_json` return `{"ok", "items", "errors"}` with
item-level success/failure. See
[`docs/external_interfaces.md`](external_interfaces.md).

All JSON functions:
- accept and return JSON **strings** (never Python objects),
- return `{"error", "error_type", "traceback"}` on failure,
- never raise to the caller (all exceptions are caught and
  formatted as JSON).

**Setting model:** JSON = user's request (algorithm, hyperparameters, run-shape). Provider's setting (paths, models, GPU device, cache) lives only on the Python signature as kwargs — JSON values are silently ignored.

The CLI (`python run_search.py ...`) is unchanged.
