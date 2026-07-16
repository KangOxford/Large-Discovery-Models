# `bo_api.py` HTTP Service — Deployment Resource Guide

> A guide for wrapping `PDF2Dock/bo_api.py` as an HTTP service.
> This document is **not** a tutorial on how to write an HTTP
> server; it is a reference of the resources (binaries, models,
> caches) and operational concerns (GPU, disk, concurrency) the
> wrapper needs.

---

## §1. What `bo_api.py` is

`PDF2Dock/bo_api.py` exposes two JSON-in/JSON-out functions for
programmatic access to the BO / random-search loops without going
through the CLI:

| Function | Purpose |
|---|---|
| `bo_api.run_search_trajectory(request_json, **provider_kwargs)` | Full BO/LDM loop (analog generation + scoring + selection). Latency ~30–300 s. |
| `bo_api.recommend_next_smiles(request_json, *, gp_device=None, llm_model=None, llm_base_url=None, llm_api_key=None)` | One-shot BO/LDM advisor step. Latency ~1–10 s for pure BO; LDM adds LLM latency. The caller owns the surrounding loop. |

Both functions return **JSON strings** (never Python objects) so
the boundary is safe across subprocess / HTTP / notebook contexts.
On any exception the response is `{"error": str, "error_type":
str, "traceback": str}` — `bo_api` never raises to the caller.

The full request/response schemas are documented in
[`PDF2Dock/docs/bo_api.md`](../PDF2Dock/docs/bo_api.md). The
implementation is [`PDF2Dock/bo_api.py`](../PDF2Dock/bo_api.py)
(see `__all__` at the bottom of that file for the public surface).

---

## §2. The 13 provider-setting keys (the critical part)

`bo_api` splits settings into three layers:

1. **Provider's setting** (Python kwargs only) — deployment wiring.
   13 keys total. **Not settable via JSON**; the JSON body silently
   drops them.
2. **bo_api's defaults** (`bo_api.DEFAULT`) — module-level flat
   dict of argparse-dest-name → value. Mirrors `run_search.sh`.
   Applied when the user's JSON omits a key.
3. **`run_search.py` argparse defaults** — used only when CLI is
   invoked directly.

Precedence (highest → lowest):

1. Provider's setting kwarg (for the 13 provider-setting keys)
2. User's JSON value (for user's-request keys; `null` is an
   explicit value)
3. `bo_api.DEFAULT` (when the user omits a key entirely)
4. `run_search.py` argparse default (only when CLI is invoked
   directly)

### The 13 keys the wrapper **must** inject server-side

The HTTP wrapper **cannot** accept provider settings from the client
(the JSON body is silently dropped). The wrapper must read these
13 values from process config (env vars, `.env`, or hard-code at
startup) and pass them as Python kwargs to every `bo_api` call.

| Python kwarg | Env var fallback | Hard-coded default | Absolute path on this host |
|---|---|---|---|
| `vina_bin` | `VINA_BIN` | `<repo>/../bin/vina` | `/mnt/data1/dock-project/bin/vina` |
| `vina_cache_dir` | — | `output/bo/vina_cache/` | `/mnt/data1/dock-project/PDF2Dock/output/bo/vina_cache/` |
| `vina_max_workers` | — | `1` | — (set to host's CPU count) |
| `gp_device` | — | `"cuda"` | e.g. `"cuda:0"` |
| `reasyn_repo` | `REASYN_HOME` / `REASYN_REPO` | — | `/mnt/data1/dock-project/ReaSyn` |
| `reasyn_python_bin` | `REASYN_PYTHON` / `REASYN_BIN` | — | `/mnt/data1/dock-project/ReaSyn/.venv/bin/python` |
| `reasyn_model_path` | `REASYN_MODEL_PATH` | AR+EB checkpoints under `data/trained_model/` | `/mnt/data1/dock-project/ReaSyn/data/trained_model/nv-reasyn-ar-166m-v2.ckpt,/mnt/data1/dock-project/ReaSyn/data/trained_model/nv-reasyn-eb-174m-v2.ckpt` |
| `reasyn_devices` | — | `"1,2"` | e.g. `"0"` |
| `nn_model_path` | — | `activity_modeling/best_model.joblib` | `/mnt/data1/dock-project/PDF2Dock/activity_modeling/best_model.joblib` |
| `nn_metadata_path` | — | model-stem metadata | `/mnt/data1/dock-project/PDF2Dock/activity_modeling/best_model_metadata.json` |
| `llm_model` | — | `DeepSeek-V4-Flash` | e.g. `"DeepSeek-V4-Flash"` |
| `llm_base_url` | `LLM_BASE_URL` | — (required for LDM methods) | OpenAI-compatible endpoint URL |
| `llm_api_key` | `LLM_API_KEY` | — (required for LDM methods) | secret; never log |

### Two interpreters — non-obvious

`bo_api` runs in the **`PDF2Dock/.venv`** Python 3.12 environment,
but **invokes `ReaSyn/.venv/bin/python` (Python 3.10) as a
subprocess** for analog generation. Both must be present and
correctly versioned. The `reasyn_python_bin` kwarg points at the
ReaSyn venv's Python, not at the wrapper's Python.

---

## §3. The HTTP endpoints to expose

| HTTP path | Wraps | Request shape | Response shape | Typical latency |
|---|---|---|---|---|
| `POST /v1/trajectory` | `bo_api.run_search_trajectory` | Full JSON: `method`, `seed`, `seed-smiles`, `num-evaluations`, `batch-size`, `objective`, … | `{"config": {...}, "history": [...], "summary": {...}}`; LDM methods also include `llm_trajectory` | 30–300 s |
| `POST /v1/recommend` | `bo_api.recommend_next_smiles` | JSON: `method`, `pool` (list of SMILES), `history`, `batch_size`, … | `{"recommendations": [...], "acquisition_values": [...], "n_objectives": ...}`; LDM methods also include `llm` diagnostics | 1–10 s plus LLM latency |
| `POST /score/vina` | `bo_api.score_vina_json` | JSON: `{"smiles": [...]}` plus Vina user knobs | Structured per-SMILES Vina scores/errors | seconds to minutes |
| `POST /score/nn` | `bo_api.score_nn_json` | JSON: `{"smiles": [...]}` | Structured per-SMILES NN scores/errors | sub-second to seconds |
| `POST /acquisition/evaluate` | `bo_api.evaluate_acquisition_json` | JSON: `{"history": [...], "query_smiles": [...]}` | Structured posterior and acquisition details | seconds |

Full request/response schemas:

- `POST /v1/trajectory` → `PDF2Dock/docs/bo_api.md §1` (request
  schema, response schema, worked example).
- `POST /v1/recommend` → `PDF2Dock/docs/bo_api.md §2` (same).
- External scoring/acquisition endpoints →
  `PDF2Dock/docs/external_interfaces.md`.

### Error format

On failure `bo_api` returns:

```json
{
  "error": "AutoDock Vina executable not found. Pass vina_bin, set $VINA_BIN, or add 'vina' to PATH. (resolved: /wrong/path)",
  "error_type": "ValueError",
  "traceback": "Traceback (most recent call last):\n  File ..."
}
```

The wrapper should map this to **HTTP 502 Bad Gateway** (the
upstream `bo_api` returned an error, not the wrapper). HTTP 400
should be reserved for wrapper-level errors (bad JSON request,
missing method, etc.).

---

## §4. Repository layout

```
/mnt/data1/dock-project/
├── bin/
│   └── vina                            # AutoDock Vina 1.2.7 binary
├── ReaSyn/                             # NVIDIA ReaSyn checkout
│   ├── .venv/bin/python                 # ReaSyn Python 3.10 (separate venv)
│   └── data/trained_model/
│       ├── nv-reasyn-ar-166m-v2.ckpt    # Autoregressive model
│       └── nv-reasyn-eb-174m-v2.ckpt    # Edit-bridge model
├── PDF2Dock/                           # The library
│   ├── bo_api.py                       # The HTTP target (do NOT modify)
│   ├── run_search.py                   # CLI driver (bo_api delegates to this)
│   ├── strbo_v1/                       # BO + GP implementation
│   ├── activity_modeling/
│   │   ├── best_model.joblib            # NN G12C pIC50 model (joblib)
│   │   └── best_model_metadata.json     # NN metadata (feature schema, scaler)
│   ├── output/bo/vina_cache/           # Vina disk cache (created on first run)
│   └── .venv/                          # Python 3.12 venv (bo_api's interpreter)
└── BO_API_HTTP_SERVICE.md              # This file
```

Key paths the wrapper must know:

- `bo_api.py` lives in `PDF2Dock/`. The wrapper runs from the
  `PDF2Dock/.venv` Python with `PYTHONPATH` set to `PDF2Dock/`.
- The **Vina binary** is at `/mnt/data1/dock-project/bin/vina`
  (NOT inside `PDF2Dock/`).
- The **ReaSyn repo** is at `/mnt/data1/dock-project/ReaSyn/`
  (NOT inside `PDF2Dock/`).
- The **ReaSyn Python interpreter** is `ReaSyn/.venv/bin/python`
  (a separate venv from `PDF2Dock/.venv/`).
- The **NN model** is `PDF2Dock/activity_modeling/best_model.joblib`
  with metadata in the sibling `.json`.
- The **Vina disk cache** is `PDF2Dock/output/bo/vina_cache/`
  (auto-created on first run; mounted on persistent storage in
  production).

---

## §5. Environment variables to set on the wrapper process

Recommended `.env` for the wrapper:

```bash
# Required: where bo_api.py lives.
PYTHONPATH=/mnt/data1/dock-project/PDF2Dock

# Required (if ReaSyn uses GPU): pin which GPU this worker sees.
CUDA_VISIBLE_DEVICES=0

# Optional — these are fallbacks. Python kwargs passed to bo_api win.
VINA_BIN=/mnt/data1/dock-project/bin/vina
REASYN_HOME=/mnt/data1/dock-project/ReaSyn
REASYN_PYTHON=/mnt/data1/dock-project/ReaSyn/.venv/bin/python
REASYN_MODEL_PATH=/mnt/data1/dock-project/ReaSyn/data/trained_model/nv-reasyn-ar-166m-v2.ckpt,/mnt/data1/dock-project/ReaSyn/data/trained_model/nv-reasyn-eb-174m-v2.ckpt

# Required for bo-tanimoto-ldm / bo-strkernel-ldm.
LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
LLM_API_KEY=...
```

> **Note**: env vars are a fallback. The recommended pattern is
> to **read the same values at startup and pass them as Python
> kwargs** to `bo_api.run_search_trajectory`. This makes the
> configuration explicit and avoids surprises when env vars
> change between calls.

---

## §6. How to construct the wrapper (skeleton)

Pseudocode (collaborators pick their framework — FastAPI, Flask,
aiohttp, Starlette, etc.):

```
PROCESS = {
    "vina_bin":         read_env("VINA_BIN") or "/home/.../bin/vina",
    "vina_cache_dir":   read_env("VINA_CACHE_DIR") or "output/bo/vina_cache/",
    "vina_max_workers": int(read_env("VINA_MAX_WORKERS") or 1),
    "gp_device":        read_env("GP_DEVICE") or "cuda:0",
    "reasyn_repo":      read_env("REASYN_HOME") or "/home/.../ReaSyn",
    "reasyn_python_bin":read_env("REASYN_PYTHON") or "/home/.../ReaSyn/.venv/bin/python",
    "reasyn_model_path":read_env("REASYN_MODEL_PATH") or "/home/.../nv-reasyn-ar-166m-v2.ckpt,/home/.../nv-reasyn-eb-174m-v2.ckpt",
    "reasyn_devices":   read_env("REASYN_DEVICES") or "0",
    "nn_model_path":    "/home/.../PDF2Dock/activity_modeling/best_model.joblib",
    "nn_metadata_path": "/home/.../PDF2Dock/activity_modeling/best_model_metadata.json",
    "llm_model":        read_env("WRAPPER_LLM_MODEL") or "DeepSeek-V4-Flash",
    "llm_base_url":     read_env("LLM_BASE_URL") or "",
    "llm_api_key":      read_env("LLM_API_KEY") or "",
}

def handler_trajectory(request_body: str) -> (status, response_body):
    # 1. Parse request_body as JSON. Reject non-objects with HTTP 400.
    # 2. Reject requests whose "method" is not in the bo_api-allowed
    #    set with HTTP 400 (early validation).
    # 3. response = bo_api.run_search_trajectory(request_body, **PROCESS)
    # 4. Parse response; if "error" key present, return (502, response).
    #    Otherwise return (200, response).

def handler_recommend(request_body: str) -> (status, response_body):
    # 1. Validate request JSON and method.
    # 2. response = bo_api.recommend_next_smiles(
    #        request_body,
    #        gp_device=PROCESS["gp_device"],
    #        llm_model=PROCESS["llm_model"],
    #        llm_base_url=PROCESS["llm_base_url"],
    #        llm_api_key=PROCESS["llm_api_key"],
    #    )
    # 3. Map {"error", ...} to 502; otherwise 200.
```

### Things the wrapper **should** do

- **Validate** the request body is a JSON object and `method` is in
  `bo_api.VALID_METHODS` before calling `bo_api`. Saves a wasted
  CPU/GPU cycle for malformed requests.
- **Map** `bo_api` errors (`{"error", ...}`) to HTTP 502.
- **Set** `Content-Type: application/json` on the response (the
  body is already a JSON string).
- **Configure** CORS for browser-based clients if needed.
- **Log** DEBUG-level messages from `bo_api` (which logs every
  silently-dropped provider-setting key) — useful for detecting
  misconfigured clients.

### Things the wrapper **should NOT** do

- **Don't** try to validate provider-setting keys (they're silently
  dropped by `bo_api`; passing them through is fine).
- **Don't** try to fill in defaults — that's `bo_api.DEFAULT`'s job.
- **Don't** parse the response body and re-wrap it — return the
  `bo_api` JSON string verbatim.
- **Don't** inject any of the 13 provider-setting keys into the
  JSON request body — `bo_api` will drop them. Inject only via
  Python kwargs.

---

## §7. Operational concerns

### Concurrency

`bo_api` is **synchronous** and CPU/GPU-bound. Do **not** share one
Python process across multiple HTTP requests (GIL + GPU contention).

Recommended scaling:

- **One process per CPU socket** (Vina parallelism).
- **One GPU per process** (ReaSyn + GP share the GPU).
- Scale **vertically** (more cores, more GPUs in one machine) before
  scaling horizontally. Horizontal scaling needs a job queue
  (Celery / RQ / Redis queue) in front of the wrapper.

### Latency budget

`run_search_trajectory` typically takes **30–300 seconds** (Vina is
the bottleneck: ~5–30 s per molecule × N evaluations). Set your
HTTP server's keep-alive timeout accordingly (e.g. uvicorn
`--timeout-keep-alive 600`).

`recommend_next_smiles` takes **1–10 seconds** (just GP fit + one
acquisition evaluation).

### Disk

- **Vina disk cache** (`vina_cache_dir`) can grow to ~10 GB per
  receptor (PDB ID + chain + ligand-resname combination).
- Mount on **persistent storage** (not tmpfs / emptyDir).
- Mount on **local NVMe** if possible — NFS adds ~50 ms latency
  per docking, which compounds over hundreds of evaluations.
- Monitor disk usage; consider a cache-cleanup cron job.

### GPU

- **GP fit** on a 1000-pool, n_obj=2 holds ~2 GB GPU memory.
  `bo-strkernel` is the heaviest (subsequence kernel matrices);
  `bo-tanimoto` is lighter.
- **One worker per GPU**. Set `CUDA_VISIBLE_DEVICES` per worker
  process (don't share a GPU across processes — CUDA context
  switching is expensive).
- **ReaSyn** also uses GPU; its peak memory is roughly proportional
  to the analog pool size and `num_editflow_samples`.
- Monitor with `nvidia-smi dmon` or equivalent.

### ReaSyn subprocess

`bo_api.run_search_trajectory` spawns
`ReaSyn/.venv/bin/python scripts/sample.py ...` as a subprocess
for each analog round. Each subprocess:

- Loads the two `.ckpt` checkpoints (~700 MB total).
- Has a per-molecule time limit (default 20 s; set via
  `reasyn-time-limit` in the JSON body).
- Generates N analogs per input SMILES.

Watch for **stuck subprocesses** — set a wall-time watchdog on
the subprocess invocation. ReaSyn doesn't have a clean
subprocess timeout; you may need to `kill -9` it from the
wrapper if it hangs.

---

## §8. Failure modes & recovery

| Symptom (in `bo_api` response `error` field) | Root cause | Fix |
|---|---|---|
| `"AutoDock Vina executable not found"` | `vina_bin` kwarg not set or wrong path | Set `VINA_BIN` env or pass `vina_bin` kwarg explicitly |
| `"Cannot locate ReaSyn repo"` | `reasyn_repo` kwarg not set | Set `REASYN_HOME` env |
| `"ReaSyn python interpreter not found"` | `reasyn_python_bin` kwarg wrong | Point at `ReaSyn/.venv/bin/python` (not the wrapper's Python) |
| `"CUDA out of memory"` | GP fit too large | Lower `max_pool_size` or `acq-budget` in the JSON request; or use a GPU with more memory |
| `"CUDA error: invalid device ordinal"` | `gp_device` / `reasyn_devices` references a non-existent GPU | Check `nvidia-smi`; set per-process `CUDA_VISIBLE_DEVICES` |
| All requests return `"GP fit failed at every jitter attempt"` | History contains constant scores (degenerate training data) | Caller bug — ensure `history` has non-constant scores |
| Subprocess timeout / hang | ReaSyn subprocess stuck | Increase `reasyn-time-limit` (default 20 s); add wall-time watchdog; check `CUDA_VISIBLE_DEVICES` is set inside ReaSyn venv |
| `"Vina: error writing output file"` | `vina_cache_dir` not writable | Ensure the directory exists and the wrapper has write permissions |

---

## §9. Testing the wrapper

The library's own tests (`PDF2Dock/tests/test_bo_api.py`) cover
the `bo_api` behaviours. The wrapper's own tests should:

- **Mock** `bo_api.run_search_trajectory` and
  `bo_api.recommend_next_smiles` so they don't need Vina / ReaSyn
  / NN.
- For `bo-*-ldm`, also mock the LLM client or point
  `llm_base_url` / `llm_api_key` at a test OpenAI-compatible
  endpoint.
- Cover HTTP status codes:
  - **200** on success.
  - **400** on bad JSON request (not an object, missing `method`).
  - **502** when `bo_api` returns `{"error", ...}`.
- Cover that **provider-setting keys in the JSON body are passed
  through** (the wrapper must not strip them — `bo_api` does that).

A **live smoke test** (separate from the unit tests) should call
the running HTTP endpoint with a tiny mock trajectory
(`num-evaluations=4`, `objective="vina+nn"`, `seed-smiles="CCO,CCN,CCC"`)
and assert ~30 s latency + non-empty `history`.

---

## §10. Reference

### Source-of-truth documentation

- [`PDF2Dock/docs/bo_api.md`](../PDF2Dock/docs/bo_api.md) — full
  request/response schemas for both endpoints, the 3-layer
  setting model, all kwargs, common error types, worked
  examples. **Read this first.**
- [`PDF2Dock/run_search.sh`](../PDF2Dock/run_search.sh) — the
  bash driver; canonical defaults that `bo_api.DEFAULT` mirrors.

### Implementation entry points

- [`PDF2Dock/bo_api.py`](../PDF2Dock/bo_api.py) — the module.
  Key landmarks:
  - `PROVIDER_SETTING_KEYS` — provider kwarg names.
  - `DEFAULT` — bo_api's user-request defaults.
  - `VALID_METHODS` — early HTTP method validation surface.
  - `run_search_trajectory` — full trajectory entry point.
  - `recommend_next_smiles` — one-step advisor entry point.
  - `__all__` — public surface.

### Resources

- **Vina binary**: `/mnt/data1/dock-project/bin/vina`
- **ReaSyn repo**: `/mnt/data1/dock-project/ReaSyn/`
- **ReaSyn checkpoints**: `/mnt/data1/dock-project/ReaSyn/data/trained_model/`
- **NN model**: `/mnt/data1/dock-project/PDF2Dock/activity_modeling/best_model.joblib`
- **NN metadata**: `/mnt/data1/dock-project/PDF2Dock/activity_modeling/best_model_metadata.json`
- **Vina disk cache**: `/mnt/data1/dock-project/PDF2Dock/output/bo/vina_cache/`

### Useful upstream references

- [AutoDock Vina documentation](https://autodock-vina.readthedocs.io/)
- [NVIDIA ReaSyn (NeurIPS 2024)](https://research.nvidia.com/labs/dvl/projects/reasyn/)

---

*Last updated: alongside the LDM-enabled `bo_api.py` API surface
(`bo-*-ldm` methods plus 13 provider-setting keys).*
