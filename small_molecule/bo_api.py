"""JSON-in/JSON-out API wrapping the ``strbo_v1`` search library.

This module exposes two public functions for programmatic access to
the BO / random-search loops without going through the CLI:

* :func:`run_search_trajectory`  — full trajectory runner, returns
  ``{"config": ..., "history": ..., "summary": ...}`` (same schema
  as the per-trajectory JSON file written by ``run_search.py``).
* :func:`recommend_next_smiles` — pure advisor step, returns
  ``{"recommendations": ..., "method": ..., "n_history": ...,
  "pool_size": ..., "acquisition_values": ..., "n_objectives": ...}``.
  The caller manages the surrounding loop (analog generation,
  black-box scoring).

Both functions accept and return JSON strings, never Python objects
(so they're safe to expose over HTTP, from a notebook, or as a
subprocess boundary). On any exception the response is
``{"error": str, "error_type": str, "traceback": str}`` so the
caller always gets a structured failure.

The full request/response schemas and worked examples are in
``docs/bo_api.md``.

Settings are split into three layers:

* **Provider's setting** (Python kwargs only) — deployment wiring:
  binary paths, model checkpoints, GPU device, cache directory,
  parallelism, and LLM endpoint credentials. Listed by
  :data:`PROVIDER_SETTING_KEYS`.
  These are deliberately **not** settable via the JSON body; any
  JSON value for them is silently ignored. Configure them via the
  Python kwarg, the relevant env var, or the hard-coded default.
* **bo_api's defaults** (:data:`DEFAULT`) — module-level flat dict
  of argparse-dest-name → value. Mirrors ``run_search.sh``. Applied
  when the user's JSON omits a key entirely. The user's value (incl.
  ``null``) always wins.
* **run_search.py argparse defaults** — used only when
  ``run_search.py`` is invoked directly via CLI; bo_api users
  always see :data:`DEFAULT` first.

Precedence (highest → lowest):

1. Provider's setting kwarg (for provider-setting keys)
2. User's JSON value (for user's-request keys; ``null`` = explicit value)
3. :data:`DEFAULT` (when the user omits a user's-request key entirely)
4. ``run_search.py`` argparse default (only applies when CLI is invoked
   directly; bo_api always wins above this layer)
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from run_search import config_from_dict, run_one_trajectory

from strbo_v1 import (
    BayesianAnalogSearchConfig,
    GPConfig,
    RNG,
    bayesian_select_candidates,
    evaluate_acquisition,
    random_select_next_batch,
    score_nn,
    score_vina,
)


LOGGER = logging.getLogger("bo_api")


VALID_METHODS_BO = ("bo-tanimoto", "bo-strkernel")
VALID_METHODS_LDM = ("bo-tanimoto-ldm", "bo-strkernel-ldm")
VALID_METHODS_RANDOM = ("random", "random-best")
VALID_METHODS = VALID_METHODS_BO + VALID_METHODS_LDM + VALID_METHODS_RANDOM
GP_IMPL = {
    "bo-tanimoto": "fingerprint+tanimoto",
    "bo-strkernel": "smiles-strkernel",
    "bo-tanimoto-ldm": "fingerprint+tanimoto",
    "bo-strkernel-ldm": "smiles-strkernel",
}

# Provider's setting: deployment wiring. JSON values for these keys
# are silently dropped by ``run_search_trajectory``; configure them
# via Python kwargs / env var / hard-coded default. The JSON body
# never participates in their resolution.
#
# Keys are stored in underscore form (the argparse ``dest``). The
# filter normalises hyphen-keys to underscore-keys before comparing,
# so callers may pass either form in JSON — both are ignored.
PROVIDER_SETTING_KEYS: frozenset = frozenset({
    "vina_bin",
    "vina_cache_dir",
    "vina_max_workers",
    "gp_device",
    "reasyn_repo",
    "reasyn_python_bin",
    "reasyn_model_path",
    "reasyn_devices",
    "nn_model_path",
    "nn_metadata_path",
    "llm_model",
    "llm_base_url",
    "llm_api_key",
})

# bo_api's defaults. Mirrors `run_search.sh`. The user's JSON request
# always wins (setdefault semantics for ``run_search_trajectory``,
# ``.get(k, DEFAULT[k])`` semantics for ``recommend_next_smiles``).
# ``run_search.py`` argparse defaults sit below this layer in the
# precedence chain (they apply only when ``run_search.py`` is invoked
# directly via CLI; bo_api users always see the bo_api defaults first).
#
# Keys are flat argparse-dest names. The same dict is consulted by
# both ``run_search_trajectory`` (via ``request.setdefault``) and
# ``recommend_next_smiles`` (via ``request.get``).
#
# IMPORTANT: Provider-setting keys (see :data:`PROVIDER_SETTING_KEYS`)
# are deliberately NOT in this dict. They are settable only via the
# Python kwarg (provider's setting). Their fallbacks come from
# argparse defaults in the trajectory path and from hardcoded values
# in the advisor's ``_build_gp_config``.
DEFAULT: Dict[str, Any] = {
    # --- Trajectory / advisor shared ---
    "num_evaluations": 80,
    "batch_size": 5,
    "init_size": 10,
    "objective": "vina+nn",

    # GP tuning (flat argparse-dest names; `gp_device` excluded —
    # it's a provider-setting key, not in DEFAULT).
    "gp_fit_itersteps": 100,
    "gp_learning_rate": 0.05,
    "gp_min_jitter": 1e-6,
    "gp_max_jitter": 1e-1,
    "gp_standardize_y": True,
    "gp_fp_radius": 2,
    "gp_fp_n_bits": 2048,

    # Acquisition / multi-obj (same as argparse; included for completeness).
    "acquisition": "ei",
    "xi": 0.01,
    "kappa": 2.0,
    "ehvi_n_samples": 128,
    "che_alpha": 1.0,

    # ReaSyn tuning (provider-setting keys like reasyn_repo / reasyn_python_bin
    # / reasyn_model_path / reasyn_devices excluded).
    "reasyn_search_width": 5,
    "reasyn_exhaustiveness": 8,
    "reasyn_num_cycles": 3,
    "reasyn_num_editflow_samples": 10,
    "reasyn_num_editflow_steps": 30,
    "reasyn_time_limit": 20,
    "reasyn_num_workers_per_gpu": 1,
    "reasyn_filter_sim": 0.8,

    # Pool sizing.
    "smiles_max_len": 100,
    "pool_min_size": 9,
    "pool_max_size": 18,
    "max_pool_size": 1024,
}


# ---------------------------------------------------------------------------
# External scoring/acquisition APIs
# ---------------------------------------------------------------------------


def score_vina_json(
    request_json: str,
    *,
    vina_bin: Optional[str] = None,
    vina_cache_dir: Optional[str] = None,
    vina_max_workers: Optional[int] = None,
) -> str:
    """Score SMILES with Vina through the JSON boundary."""
    try:
        request = _parse_external_request(request_json)
        _drop_json_provider_settings(request)
        return _json_dumps(score_vina(
            request=request,
            vina_bin=vina_bin,
            vina_cache_dir=vina_cache_dir,
            max_workers=vina_max_workers,
        ))
    except Exception as exc:
        return _external_error_json(exc)


def score_nn_json(
    request_json: str,
    *,
    nn_model_path: Optional[str] = None,
    nn_metadata_path: Optional[str] = None,
) -> str:
    """Score SMILES with the NN activity model through the JSON boundary."""
    try:
        request = _parse_external_request(request_json)
        _drop_json_provider_settings(request)
        return _json_dumps(score_nn(
            request=request,
            model_path=nn_model_path,
            metadata_path=nn_metadata_path,
        ))
    except Exception as exc:
        return _external_error_json(exc)


def evaluate_acquisition_json(
    request_json: str,
    *,
    gp_device: Optional[str] = None,
) -> str:
    """Evaluate acquisition/posterior details through the JSON boundary."""
    try:
        request = _parse_external_request(request_json)
        _drop_json_provider_settings(request)
        return _json_dumps(evaluate_acquisition(
            request=request,
            gp_device=gp_device,
        ))
    except Exception as exc:
        return _external_error_json(exc)


# ---------------------------------------------------------------------------
# API 1: full trajectory
# ---------------------------------------------------------------------------


def run_search_trajectory(
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
) -> str:
    """Run one full trajectory and return the JSON record.

    Request JSON mirrors the ``run_search.py`` CLI args as a flat
    dict — keys are the long-form flag names (with hyphens, e.g.
    ``"num-evaluations"``) or their underscore-attribute equivalents
    (e.g. ``"num_evaluations"``). See ``docs/bo_api.md §1`` for the
    full schema and worked example.

    Settings are split into two layers:

    * **User's request** (JSON body) — method, seed, num-evaluations,
      batch-size, init-size, objective, ref-point, hyper-parameters,
      search knobs (vina-exhaustiveness, reasyn-search-width, …).
    * **Provider's setting** (Python kwargs only) — deployment
      wiring. The keys listed below are settable only via Python
      kwargs / env var / hard-coded default. JSON values for them
      are silently dropped (one DEBUG log line per ignored key).

    **Provider's setting — ``kwarg > env var > default``** (JSON
    never participates):

    =======================  ============================  ============================================  ===========================================
    Kwarg                    Env var                       Hard-coded default                            Notes
    =======================  ============================  ============================================  ===========================================
    ``vina_bin``             ``VINA_BIN``                  ``<repo>/../bin/vina``                        AutoDock Vina binary.
    ``vina_cache_dir``       —                             ``output/bo/vina_cache/``                     Disk cache directory.
    ``vina_max_workers``     —                             ``1``                                         Parallel Vina workers.
    ``gp_device``            —                             ``"cuda"``                                    GP device (``"cuda"``/``"cuda:0"``/``"cpu"``).
    ``reasyn_repo``          ``REASYN_HOME`` / ``REASYN_REPO``  — (required)                              ReaSyn checkout.
    ``reasyn_python_bin``    ``REASYN_PYTHON`` / ``REASYN_BIN``  —                                         Python interpreter inside ReaSyn env.
    ``reasyn_model_path``    ``REASYN_MODEL_PATH``         AR+EB checkpoints under ``data/trained_model/``  Comma-separated checkpoint paths.
    ``reasyn_devices``       —                             ``"1,2"``                                     Comma-separated GPU ids for ReaSyn.
    ``nn_model_path``        —                             ``DEFAULT_NN_MODEL_PATH``                     NN G12D pIC50 model.
    ``nn_metadata_path``     —                             model-stem metadata (e.g. ``best_g12d_model_metadata.json``)  NN sidecar metadata JSON.
    ``llm_model``            —                             ``DeepSeek-V4-Flash``                          LDM chat model (LDM methods only).
    ``llm_base_url``         ``LLM_BASE_URL``              — (required for LDM methods)                   OpenAI-compatible endpoint.
    ``llm_api_key``          ``LLM_API_KEY``               — (required for LDM methods)                   OpenAI-compatible API key.
    =======================  ============================  ============================================  ===========================================

    Setting any of these via JSON has no effect. If you accidentally
    include, say, ``"vina-bin": "/x"`` in the request body, it is
    silently dropped; the kwarg / env var / default chain still
    applies. The DEBUG-level log message
    ``"bo_api: ignoring JSON provider-setting %r; configure via
    kwarg or env var"`` is emitted once per ignored key.

    For the user's-request layer (everything not in the table above),
    standard ``argparse`` semantics apply: the JSON value is passed
    through to the CLI parser as if the caller had typed
    ``--<key> <value>``.

    Args:
        request_json: A JSON object string mirroring the
            ``run_search.py`` CLI flag namespace. See ``docs/bo_api.md``
            §1.1 for the full schema.

        vina_bin: Path to the AutoDock Vina binary. Settable only
            via this kwarg, the ``VINA_BIN`` env var, or the
            hard-coded ``<repo>/../bin/vina`` default. **Not
            settable via JSON** — any ``"vina-bin"`` /
            ``"vina_bin"`` key in ``request_json`` is silently
            dropped. Example: ``"/opt/vina/bin/vina"``.

        vina_cache_dir: Path to the disk cache directory for
            prepared Vina receptors and ligands. Settable only via
            this kwarg or the matching argparse default
            (``output/bo/vina_cache/``). **Not settable via JSON** —
            any ``"vina-cache-dir"`` / ``"vina_cache_dir"`` key in
            ``request_json`` is silently dropped. Example:
            ``"/var/cache/vina"``.

        vina_max_workers: Number of parallel Vina workers (overrides
            the argparse default of ``1``). Settable only via this
            kwarg. **Not settable via JSON** — any
            ``"vina-max-workers"`` / ``"vina_max_workers"`` key in
            ``request_json`` is silently dropped. Example: ``4``.

        reasyn_repo: Path to the ReaSyn git repository. Settable
            only via this kwarg, the ``REASYN_HOME`` /
            ``REASYN_REPO`` env vars. **Not settable via JSON** —
            any ``"reasyn-repo"`` / ``"reasyn_repo"`` key in
            ``request_json`` is silently dropped. Example:
            ``"../ReaSyn"`` or ``"/home/user/ReaSyn"``.

        reasyn_python_bin: Path to the Python interpreter inside
            the ReaSyn environment. Settable only via this kwarg,
            the ``REASYN_PYTHON`` / ``REASYN_BIN`` env vars. **Not
            settable via JSON** — any ``"reasyn-python-bin"`` /
            ``"reasyn_python_bin"`` key in ``request_json`` is
            silently dropped.

        reasyn_model_path: Comma-separated path(s) to ReaSyn
            checkpoint files. Settable only via this kwarg, the
            ``REASYN_MODEL_PATH`` env var, or the hardcoded AR+EB
            checkpoints (under ``data/trained_model/``). **Not
            settable via JSON** — any ``"reasyn-model-path"`` /
            ``"reasyn_model_path"`` key in ``request_json`` is
            silently dropped.

        reasyn_devices: Comma-separated GPU ids (e.g. ``"0,1"``).
            Settable only via this kwarg or the hard-coded default
            (``"1,2"``). **Not settable via JSON** — any
            ``"reasyn-devices"`` / ``"reasyn_devices"`` key in
            ``request_json`` is silently dropped.

        gp_device: One of ``"cuda"``, ``"cuda:0"``, ``"cpu"`` —
            selects the device for the GP. Settable only via this
            kwarg or the matching argparse default (``"cuda"``).
            **Not settable via JSON** — any ``"gp-device"`` /
            ``"gp_device"`` key in ``request_json`` is silently
            dropped. Example: ``"cuda:0"``.

        nn_model_path: Path to the joblib model file for the NN
            scorer. Settable only via this kwarg or the hard-coded
            default (``DEFAULT_NN_MODEL_PATH``).
            **Not settable via JSON** — any ``"nn-model-path"`` /
            ``"nn_model_path"`` key in ``request_json`` is silently
            dropped.

        nn_metadata_path: Path to the sidecar JSON metadata for the
            NN scorer. Settable only via this kwarg or the
            model-stem metadata default (e.g.
            ``best_g12d_model_metadata.json``). **Not settable via JSON**
            — any ``"nn-metadata-path"`` / ``"nn_metadata_path"``
            key in ``request_json`` is silently dropped.

        llm_model: LDM chat model name. Settable only via this kwarg
            or the hardcoded LDM default. **Not settable via JSON**.

        llm_base_url: OpenAI-compatible LDM endpoint base URL.
            Settable only via this kwarg or ``LLM_BASE_URL``.
            Required for ``bo-tanimoto-ldm`` / ``bo-strkernel-ldm``.
            **Not settable via JSON**.

        llm_api_key: OpenAI-compatible LDM API key. Settable only
            via this kwarg or ``LLM_API_KEY``. Required for
            ``bo-tanimoto-ldm`` / ``bo-strkernel-ldm``.
            **Not settable via JSON**.

    Returns:
        A JSON string of the form::

            {
              "config": {...},      // full echo of every input knob
              "history": [          // ordered list of {index, smiles, score|scores}
                {"index": 0, "smiles": "CCO", "score": -7.5},
                ...
              ],
              "summary": {          // bsf / hypervolume / per-obj-bsf curve
                "bsf": [...],
                "hypervolume": [...],  // n_obj=2 only
                "bsf_per_objective": [...]  // n_obj>=3 only
              },
              "llm_trajectory": {...}  // LDM methods only
            }

        On failure, returns::

            {
              "error": "msg",
              "error_type": "ValueError",
              "traceback": "Traceback (most recent call last): ..."
            }

        The ``traceback`` field is always included (not stripped)
        to make notebook / web-service debugging easy.

    Example:
        .. code-block:: python

            import json, bo_api

            response = bo_api.run_search_trajectory(
                json.dumps({
                    "method": "bo-tanimoto",
                    "seed": 0,
                    "seed-smiles": "CCO,CCN,CCC",
                    "num-evaluations": 10,
                    "batch-size": 2,
                    "objective": "vina",
                }),
                # Provider's setting — kwarg only; the JSON body never
                # carries these:
                gp_device="cuda:0",
                vina_bin="/opt/vina/bin/vina",
                vina_cache_dir="/var/cache/vina",
                vina_max_workers=4,
                reasyn_repo="../ReaSyn",
                reasyn_python_bin="/path/to/conda/envs/reasyn/bin/python",
                reasyn_devices="0",
                nn_model_path="/models/best.joblib",
                nn_metadata_path="/models/best_metadata.json",
                llm_base_url="https://llm.example/v1",
                llm_api_key="...",
            )
    """
    try:
        request = json.loads(request_json)
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        # Provider's setting: drop deployment-wiring keys from the
        # request body. They are settable only via Python kwargs,
        # env vars, or hard-coded defaults; JSON never participates.
        _drop_json_provider_settings(request)
        # Fill bo_api defaults for keys the user did not supply.
        # User's explicit value (including ``null``) wins. Both the
        # underscore and the hyphen form of each key are checked
        # (callers use either form interchangeably).
        for k, v in DEFAULT.items():
            hyphen_key = k.replace("_", "-")
            if k not in request and hyphen_key not in request:
                request[k] = v
        args = config_from_dict(request)
        # Inject Python-level path overrides. Precedence for these 9
        # settings: kwarg > env var > default. The existing builders in
        # run_search.py (_build_vina_scorer, _build_reasyn_analog,
        # _build_nn_scorer) read args.<name> and fall back to env vars +
        # defaults via `or` chains; setting a non-None value here is
        # sufficient to make it win.
        if vina_bin is not None:
            args.vina_bin = vina_bin
        if vina_cache_dir is not None:
            args.vina_cache_dir = vina_cache_dir
        if vina_max_workers is not None:
            args.vina_max_workers = vina_max_workers
        if reasyn_repo is not None:
            args.reasyn_repo = reasyn_repo
        if reasyn_python_bin is not None:
            args.reasyn_python_bin = reasyn_python_bin
        if reasyn_model_path is not None:
            args.reasyn_model_path = reasyn_model_path
        if reasyn_devices is not None:
            args.reasyn_devices = reasyn_devices
        if gp_device is not None:
            args.gp_device = gp_device
        if nn_model_path is not None:
            args.nn_model_path = nn_model_path
        if nn_metadata_path is not None:
            args.nn_metadata_path = nn_metadata_path
        if llm_model is not None:
            args.llm_model = llm_model
        if llm_base_url is not None:
            args.llm_base_url = llm_base_url
        if llm_api_key is not None:
            args.llm_api_key = llm_api_key
        result = run_one_trajectory(args, include_summary=True)
        # ``result["history"]`` is the in-memory tuple list (matches
        # the CLI write_json contract); ``result["history_json"]`` is
        # the JSON-safe form we expose over the API.
        payload = {
            "config": result["config"],
            "history": result["history_json"],
            "summary": result["summary"],
        }
        if "llm_trajectory" in result:
            payload["llm_trajectory"] = result["llm_trajectory"]
        return _json_dumps(payload)
    except SystemExit as exc:
        # ``argparse.error`` and ``run_one_trajectory`` raise
        # ``SystemExit`` for friendly CLI errors; convert to JSON.
        return _error_json(ValueError(
            str(exc) or "argparse exited without a message"
        ))
    except Exception as exc:
        return _error_json(exc)


# ---------------------------------------------------------------------------
# API 2: advisor step
# ---------------------------------------------------------------------------


def recommend_next_smiles(
    request_json: str,
    *,
    gp_device: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
) -> str:
    """Return the top-k SMILES to evaluate next from a candidate pool.

    The caller drives the surrounding loop: they own the black-box
    scorer, the analog generator, and the history. This function
    answers "given what we have already evaluated, which SMILES from
    this pool should we evaluate next?" using the same algorithm
    dispatch as the in-loop :func:`bayesian_analog_search`.
    For ``bo-tanimoto-ldm`` / ``bo-strkernel-ldm``, the same one-step
    call also runs the LDM Stage A1/A2 pool-review phase, BO
    acquisition, and Stage B suggestion review, then returns the
    final recommendations without scoring them.

    Request JSON schema::

        {
          "method": "bo-tanimoto" | "bo-strkernel" |
                    "bo-tanimoto-ldm" | "bo-strkernel-ldm" |
                    "random" | "random-best",
          "pool": ["CCO", "CCN", ...],          // candidates
          "history": [                          // prior evaluations
            {"smiles": "CCO", "score": -7.5},   // n_obj=1
            {"smiles": "CCN", "scores": [-7.2, 4.8]}  // n_obj>=2
          ],
          "batch_size": 3,                      // how many to pick
          "minimize": true | [true, false, ...],// per-obj direction
          "ref_point": [0.0, 5.0] | null,       // n_obj=2 only
          "ehvi_n_samples": 128,                // n_obj=2 only
          "che_alpha": 1.0,                     // n_obj>=3 only
          "acq_budget": null,                   // optional subsample
          "acquisition": "ei" | "pi" | "ucb",   // n_obj=1 only
          "xi": 0.01,                           // EI/PI threshold
          "kappa": 2.0,                        // UCB exploration
          "gp_device": "cuda:0" | "cpu",        // provider's setting; kwarg wins
          "gp_fit_itersteps": 100,              // GP tuning (flat argparse-dest)
          "gp_learning_rate": 0.05,
          "gp_min_jitter": 1e-6,
          "gp_max_jitter": 1e-1,
          "gp_standardize_y": true,
          "gp_fp_radius": 2,
          "gp_fp_n_bits": 2048,
          "seed": 0,                            // RNG seed
          "ldm_sys_prompt": "",                // LDM methods only
          "analog_pool": {"CCO": ["CCCO"]}      // optional LDM analog provider
        }

    **Provider's setting.** The advisor does not invoke Vina,
    ReaSyn, or the NN scorer, so those deployment knobs are not
    applicable here. The provider-side settings that do apply are
    the **GP device** and, for LDM methods, LLM endpoint settings:
    ``gp_device``, ``llm_model``, ``llm_base_url``, and
    ``llm_api_key``. JSON values for these keys are silently
    dropped; pass them as Python kwargs. ``llm_base_url`` and
    ``llm_api_key`` fall back to ``LLM_BASE_URL`` / ``LLM_API_KEY``.

    All GP tuning fields (``gp_fit_itersteps``, ``gp_learning_rate``,
    …) are flat argparse-dest names in the JSON body. When the user
    omits a key, :data:`DEFAULT` fills in the value (mirrors
    ``run_search.sh``). User's explicit value, including ``null``,
    always wins.

    Args:
        request_json: A JSON object string describing the pool,
            history, and search configuration. See §2.1 of
            ``docs/bo_api.md``.

        gp_device: One of ``"cuda"``, ``"cuda:0"``, ``"cpu"`` —
            selects the device for the GP. Settable only via this
            kwarg. Falls back to the hardcoded ``"cuda"`` default.
            **Not settable via JSON** — any ``"gp-device"`` /
            ``"gp_device"`` key in ``request_json`` is silently
            dropped. Example: ``"cuda:0"``.

        llm_model: LDM chat model name for ``bo-*-ldm`` methods.
            Settable only via this kwarg. Falls back to the LDM
            default model.

        llm_base_url: OpenAI-compatible LDM endpoint for
            ``bo-*-ldm`` methods. Settable only via this kwarg or
            ``LLM_BASE_URL``.

        llm_api_key: OpenAI-compatible LDM API key for
            ``bo-*-ldm`` methods. Settable only via this kwarg or
            ``LLM_API_KEY``.

    Returns:
        A JSON string::

            {
              "recommendations": ["CCC", "CCO", "CCN"],  // top-k
              "method": "bo-tanimoto",
              "n_history": 5,
              "pool_size": 100,
              "pool_size_after_ldm": 120,                  // LDM only
              "acquisition_values": [0.85, 0.72, 0.68],  // [] for random
              "n_objectives": 1,                         // 1, 2, or 3+
              "llm": {...}                               // LDM only
            }

        On failure, returns ``{"error": ..., "error_type": ...,
        "traceback": ...}`` (same format as API 1).
    """
    try:
        request = json.loads(request_json)
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        _drop_json_provider_settings(request)
        return _advisor_response(
            request,
            gp_device=gp_device,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
        )
    except SystemExit as exc:
        return _error_json(ValueError(
            str(exc) or "argparse exited without a message"
        ))
    except Exception as exc:
        return _error_json(exc)


def _advisor_response(
    request: dict,
    *,
    gp_device: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
) -> str:
    method = str(request.get("method", ""))
    if method not in VALID_METHODS:
        raise ValueError(
            f"method must be one of {list(VALID_METHODS)}; got {method!r}"
        )

    pool = list(request.get("pool", []))
    history_raw = list(request.get("history", []))
    batch_size = int(request.get("batch_size", DEFAULT["batch_size"]))
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size!r}")
    seed = request.get("seed")
    rng = RNG(seed=seed) if seed is not None else RNG(seed=None)

    n_obj = _infer_n_obj(history_raw)
    minimize_raw = request.get("minimize", True)
    if isinstance(minimize_raw, bool):
        minimize_t: Tuple[bool, ...] = (minimize_raw,) * n_obj
    else:
        seq = list(minimize_raw)
        if not all(isinstance(x, bool) for x in seq):
            raise ValueError(f"minimize entries must all be bool; got {minimize_raw!r}")
        minimize_t = tuple(seq)
        if len(minimize_t) != n_obj:
            raise ValueError(
                f"minimize length ({len(minimize_t)}) != n_objectives ({n_obj})"
            )

    if method in VALID_METHODS_RANDOM:
        recs = random_select_next_batch(pool, batch_size=batch_size, rng=rng)
        return _json_dumps({
            "recommendations": recs,
            "method": method,
            "n_history": len(history_raw),
            "pool_size": len(pool),
            "acquisition_values": [],
            "n_objectives": n_obj,
        })

    if method in VALID_METHODS_LDM:
        return _ldm_advisor_response(
            request,
            pool=pool,
            history_raw=history_raw,
            batch_size=batch_size,
            seed=seed,
            rng=rng,
            n_obj=n_obj,
            minimize=minimize_t,
            gp_device=gp_device,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
        )

    history_pairs = _normalize_history(history_raw, n_obj)
    ref_point = _resolve_ref_point_for_advisor(request, n_obj)
    # Provider's setting: ``gp_device`` is kwarg-only (provider's
    # setting; not in DEFAULT). Falls back to hardcoded ``"cuda"``.
    device = gp_device if gp_device is not None else "cuda"
    gp_cfg = _build_gp_config(request, method, device=device)
    bo_cfg = BayesianAnalogSearchConfig(
        batch_size=batch_size,
        minimize=minimize_t,
        acq_budget=request.get("acq_budget"),
        ref_point=ref_point,
        ehvi_n_samples=int(request.get("ehvi_n_samples", DEFAULT["ehvi_n_samples"])),
        che_alpha=float(request.get("che_alpha", DEFAULT["che_alpha"])),
        acquisition=str(request.get("acquisition", DEFAULT["acquisition"])),
        xi=float(request.get("xi", DEFAULT["xi"])),
        kappa=float(request.get("kappa", DEFAULT["kappa"])),
        gp_config=gp_cfg,
    )
    recs, acq_values = bayesian_select_candidates(
        pool=pool, history=history_pairs, config=bo_cfg, rng=rng,
    )
    acq_list = (
        [float(v) for v in acq_values] if acq_values is not None else []
    )
    return _json_dumps({
        "recommendations": list(recs),
        "method": method,
        "n_history": len(history_raw),
        "pool_size": len(pool),
        "acquisition_values": acq_list,
        "n_objectives": n_obj,
    })


# ---------------------------------------------------------------------------
# LDM advisor step
# ---------------------------------------------------------------------------


def _ldm_advisor_response(
    request: dict,
    *,
    pool: List[str],
    history_raw: List[dict],
    batch_size: int,
    seed: Any,
    rng: RNG,
    n_obj: int,
    minimize: Tuple[bool, ...],
    gp_device: Optional[str],
    llm_model: Optional[str],
    llm_base_url: Optional[str],
    llm_api_key: Optional[str],
) -> str:
    """Run one LDM-assisted recommendation step without scoring.

    This mirrors one LDM round from ``run_search.py`` but stops before
    scoring. Stage A1/A2 may mutate the candidate pool, BO picks the
    top candidates, Stage B reviews them, and the final candidates are
    returned as recommendations.
    """
    import dataclasses

    from run_search import _resolve_ldm_sys_prompt
    from strbo_v1.llm_advisor.advisor import LLMAdvisor
    from strbo_v1.llm_advisor.orchestrator import (
        OrchestratorConfig,
        _action_state_from_snapshot,
        _apply_actions,
        _apply_review_analogs,
        _apply_review_suggestions,
        _build_review_analogs_state,
        _first_of_type,
        _run_bo_step,
        _snapshot_actions,
        _snapshot_suggestions,
        _suggestion_state_from_snapshot,
    )
    from strbo_v1.llm_advisor.parser import (
        SemanticError,
        format_error_for_prompt,
    )
    from strbo_v1.llm_advisor.reasyn_pool import ReasynConfigPool
    from strbo_v1.llm_advisor.trajectory import (
        serialize_attempts,
        serialize_blocks,
    )

    method = str(request.get("method", ""))
    history = _history_to_ldm_ordered_dict(history_raw, n_obj)
    seen_history = set(history.keys())
    pool_before = _dedupe_strings(pool)
    pool_working = [s for s in pool_before if s not in seen_history]

    device = gp_device if gp_device is not None else "cuda"
    gp_cfg = _build_gp_config(request, method, device=device)
    ref_point = _resolve_ref_point_for_advisor(request, n_obj)
    bo_cfg = BayesianAnalogSearchConfig(
        batch_size=batch_size,
        minimize=minimize,
        acq_budget=request.get("acq_budget"),
        ref_point=ref_point,
        ehvi_n_samples=int(request.get("ehvi_n_samples", DEFAULT["ehvi_n_samples"])),
        che_alpha=float(request.get("che_alpha", DEFAULT["che_alpha"])),
        acquisition=str(request.get("acquisition", DEFAULT["acquisition"])),
        xi=float(request.get("xi", DEFAULT["xi"])),
        kappa=float(request.get("kappa", DEFAULT["kappa"])),
        gp_config=gp_cfg,
    )

    pool_min_size = request.get("pool_min_size")
    if pool_min_size is None:
        pool_min_size = batch_size
    pool_min_size = max(batch_size, int(pool_min_size))
    max_pool_size = request.get("max_pool_size", DEFAULT["max_pool_size"])
    if max_pool_size is not None:
        max_pool_size = int(max_pool_size)

    raw_guidance = request.get("ldm_sys_prompt", request.get("guidance", ""))
    guidance = _resolve_ldm_sys_prompt(str(raw_guidance or ""))
    llm = _build_llm_client(
        model=llm_model,
        base_url=llm_base_url,
        api_key=llm_api_key,
    )
    advisor = LLMAdvisor(
        llm=llm,
        max_retries=int(request.get("llm_max_retries", 3)),
        use_rdkit=bool(request.get("llm_use_rdkit", True)),
        guidance=guidance,
    )
    analog_fn = _build_json_analog_fn(request)
    reasyn_pool = ReasynConfigPool.from_env() if analog_fn is not None else None

    orch_cfg = OrchestratorConfig(
        init_size=int(request.get("init_size", DEFAULT["init_size"])),
        batch_size=batch_size,
        n_iterations=1,
        smiles_max_len=int(request.get("smiles_max_len", DEFAULT["smiles_max_len"])),
        bo_config=bo_cfg,
        method=method,
        seed=int(seed) if seed is not None else 0,
        objective_legend=_objective_legend_from_request(request, n_obj, minimize),
        minimize=minimize,
        pool_max_size=max_pool_size,
        pool_min_size=pool_min_size,
        n_obj=n_obj,
        guidance=guidance,
    )

    # ---- Stage A1/A2: LLM pool mutation -------------------------------
    all_attempts_A1 = []
    all_attempts_A2 = []
    final_blocks_A1 = []
    final_blocks_A2 = []
    fb_A1 = False
    fb_A2 = False
    snap = _snapshot_actions(
        pool=pool_working,
        history=history,
        config=orch_cfg,
        round_idx=0,
        stagnation_counter=int(request.get("stagnation_counter", 0)),
    )
    action_state = _action_state_from_snapshot(snap)

    for iter_idx in range(orch_cfg.max_pool_size_iters):
        blocks_A1, attempts_A1, fb_A1 = advisor.decide_actions(action_state)
        all_attempts_A1.extend(attempts_A1)
        final_blocks_A1 = blocks_A1

        new_analogs = _apply_actions(
            blocks=blocks_A1,
            pool=pool_working,
            analog_fn=analog_fn,
            reasyn_pool=reasyn_pool,
        )
        _cap_pool(pool_working, max_pool_size)

        if new_analogs:
            review_state = _build_review_analogs_state(
                action_state=action_state,
                new_analogs=new_analogs,
                pool=pool_working,
                history=history,
                round_idx=0,
                config=orch_cfg,
            )
            blocks_A2, attempts_A2, fb_A2 = advisor.decide_review_analogs(review_state)
            all_attempts_A2.extend(attempts_A2)
            final_blocks_A2 = blocks_A2
            _apply_review_analogs(
                blocks=blocks_A2,
                pool=pool_working,
                new_analogs=new_analogs,
            )
            _cap_pool(pool_working, max_pool_size)

        pool_working[:] = [s for s in pool_working if s not in seen_history]
        if len(pool_working) >= pool_min_size or fb_A1:
            break

        pool_err = SemanticError(
            f"pool has {len(pool_working)} SMILES (< min {pool_min_size}); "
            f"you MUST emit `propose` with new SMILES or `analog` "
            f"to expand existing members. `noop` is rejected."
        )
        action_state = dataclasses.replace(
            action_state,
            pool=tuple(pool_working),
            previous_errors=tuple(
                list(action_state.previous_errors)
                + [format_error_for_prompt(pool_err)]
            ),
            attempt=iter_idx + 2,
        )

    stage_a1 = {
        "executed": True,
        "attempts": serialize_attempts(all_attempts_A1),
        "fallback_used": fb_A1,
        "final_blocks": serialize_blocks(final_blocks_A1),
        "pool_size_loop_final_pool_size": len(pool_working),
    }
    stage_a2 = {
        "executed": bool(all_attempts_A2),
        "attempts": serialize_attempts(all_attempts_A2),
        "fallback_used": fb_A2,
        "final_blocks": serialize_blocks(final_blocks_A2),
    }

    # ---- BO + Stage B review ------------------------------------------
    pick_records, _summary = _run_bo_step(
        pool=pool_working,
        history=history,
        bo_config=bo_cfg,
        rng=rng,
        top_k=batch_size,
        n_obj=n_obj,
    )
    post_snap = _snapshot_suggestions(
        pool=pool_working,
        history=history,
        bo_picks=pick_records,
        acq_function=str(getattr(bo_cfg, "acquisition", "ei")),
        config=orch_cfg,
        round_idx=0,
        stagnation_counter=int(request.get("stagnation_counter", 0)),
    )
    post_state = _suggestion_state_from_snapshot(post_snap)
    blocks_B, attempts_B, fb_B = advisor.decide_review_suggestions(post_state)
    review_bo_block = _first_of_type(blocks_B, "review_bo")
    final_candidates, overrides = _apply_review_suggestions(
        review_bo_block, pick_records,
    )
    recommendations = _dedupe_strings(final_candidates)

    acq_by_pick = {p.smiles: float(p.acq_value) for p in pick_records}
    acq_values = [float(acq_by_pick.get(s, 0.0)) for s in recommendations]
    stage_b = {
        "executed": True,
        "attempts": serialize_attempts(attempts_B),
        "fallback_used": fb_B,
        "final_blocks": serialize_blocks(blocks_B),
        "review_bo_block": (
            review_bo_block.to_dict() if review_bo_block is not None else None
        ),
        "final_candidates": list(recommendations),
        "overrides": overrides,
    }

    return _json_dumps({
        "recommendations": recommendations,
        "method": method,
        "n_history": len(history_raw),
        "pool_size": len(pool_before),
        "pool_size_after_ldm": len(pool_working),
        "acquisition_values": acq_values,
        "n_objectives": n_obj,
        "llm": {
            "model": getattr(llm, "model_name", "?"),
            "guidance": guidance,
            "stage_a1": stage_a1,
            "stage_a2": stage_a2,
            "stage_b": stage_b,
            "bo_suggestions": [p.to_dict() for p in pick_records],
            "pool_after_stage_a": list(pool_working),
        },
    })


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _drop_json_provider_settings(request: dict) -> None:
    """Drop kwarg-only provider settings from a parsed JSON request."""
    for key in list(request.keys()):
        if key.replace("-", "_") in PROVIDER_SETTING_KEYS:
            LOGGER.debug(
                "bo_api: ignoring JSON provider-setting %r; "
                "configure via kwarg or env var",
                key,
            )
            del request[key]


def _parse_external_request(request_json: str) -> dict:
    request = json.loads(request_json)
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")
    return request


def _build_llm_client(
    *,
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
):
    """Construct the production LDM chat client from kwargs/env."""
    from strbo_v1.llm_advisor import client as llm_client_module
    from strbo_v1.llm_advisor.config import (
        DEFAULT_LLM_MODEL,
        LLMClientConfig,
        load_env,
    )

    load_env()
    api_key_resolved = (api_key or os.environ.get("LLM_API_KEY", "") or "").strip()
    base_url_resolved = (
        base_url or os.environ.get("LLM_BASE_URL", "") or ""
    ).strip().rstrip("/")
    model_resolved = (model or DEFAULT_LLM_MODEL or "").strip()
    cfg = LLMClientConfig(
        api_key=api_key_resolved,
        base_url=base_url_resolved,
        model=model_resolved,
    )
    return llm_client_module.OpenAIChatClient(cfg)


def _history_to_ldm_ordered_dict(
    history_raw: List[dict], n_obj: int,
) -> "OrderedDict[str, Any]":
    """Convert API history JSON to the LDM orchestrator history shape."""
    history_pairs = _normalize_history(history_raw, n_obj)
    out: "OrderedDict[str, Any]" = OrderedDict()
    for smi, score in history_pairs:
        if n_obj == 1:
            out[smi] = None if score is None else float(score)  # type: ignore[arg-type]
        else:
            seq = list(score) if isinstance(score, (list, tuple)) else []
            out[smi] = [
                None if v is None else float(v)
                for v in seq[:n_obj]
            ]
    return out


def _objective_legend_from_request(
    request: dict, n_obj: int, minimize: Tuple[bool, ...],
) -> List[Dict[str, Any]]:
    """Build objective metadata for LDM prompts."""
    raw = request.get("objective_legend")
    if isinstance(raw, list) and len(raw) == n_obj:
        legend: List[Dict[str, Any]] = []
        for i, item in enumerate(raw):
            if isinstance(item, dict):
                legend.append({
                    "name": str(item.get("name", f"objective_{i}")),
                    "minimize": bool(item.get("minimize", minimize[i])),
                })
            else:
                legend.append({"name": str(item), "minimize": minimize[i]})
        return legend
    if n_obj == 1:
        return [{"name": "score", "minimize": minimize[0]}]
    return [
        {"name": f"objective_{i}", "minimize": minimize[i]}
        for i in range(n_obj)
    ]


def _build_json_analog_fn(request: dict):
    """Build an optional JSON-backed analog provider for one-step LDM.

    ``analog_pool`` may be either:
    * ``{"seed": ["analog1", "analog2"]}``
    * ``["analog1", "analog2"]`` (used for every seed)
    Dict analogue entries may use ``"smiles"`` or
    ``"analogue_smiles"`` and optional ReaSyn metadata fields.
    """
    raw = request.get("analog_pool")
    if raw is None:
        raw = request.get("analog_map")
    if raw is None:
        return None
    if not isinstance(raw, (dict, list)):
        raise ValueError(
            "analog_pool must be a dict seed->list or a list of SMILES/objects"
        )

    def analog_fn(seeds: Sequence[str]):
        records = []
        for seed in seeds:
            values = raw.get(seed, []) if isinstance(raw, dict) else raw
            records.extend(_analogue_records_from_json(values, seed=str(seed)))
        return records

    return analog_fn


def _analogue_records_from_json(values: Any, *, seed: str) -> List[Any]:
    from strbo_v1.llm_advisor.state import AnalogueRecord

    if values is None:
        return []
    if isinstance(values, (str, dict)):
        seq = [values]
    else:
        seq = list(values)
    out = []
    for item in seq:
        if isinstance(item, str):
            smi = item.strip()
            if smi:
                out.append(AnalogueRecord(seed_smiles=seed, analogue_smiles=smi))
            continue
        if not isinstance(item, dict):
            continue
        smi = str(item.get("analogue_smiles", item.get("smiles", ""))).strip()
        if not smi:
            continue
        out.append(AnalogueRecord(
            seed_smiles=str(item.get("seed_smiles", seed)),
            analogue_smiles=smi,
            reasyn_score=_optional_float(item.get("reasyn_score")),
            synthesis=item.get("synthesis"),
            num_steps=_optional_int(item.get("num_steps")),
            scf_sim=_optional_float(item.get("scf_sim")),
            pharm2d_sim=_optional_float(item.get("pharm2d_sim")),
        ))
    return out


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dedupe_strings(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _cap_pool(pool: List[str], max_pool_size: Optional[int]) -> None:
    if max_pool_size is None or max_pool_size < 1:
        return
    overflow = len(pool) - max_pool_size
    if overflow > 0:
        del pool[:overflow]


def _infer_n_obj(history_raw: List[dict]) -> int:
    """Infer the number of objectives from the first history entry.

    An empty history defaults to ``n_obj == 1`` (the user's pool is
    scored under single-objective defaults). For n_obj > 1, the
    caller must include at least one history entry with a ``"scores"``
    list to disambiguate.
    """
    for entry in history_raw:
        if not isinstance(entry, dict):
            continue
        if "scores" in entry:
            seq = entry["scores"]
            if not isinstance(seq, list):
                raise ValueError(
                    f"history entry 'scores' must be a list; got {type(seq).__name__}"
                )
            if len(seq) < 1:
                raise ValueError(
                    "history entry 'scores' has length 0; cannot infer n_obj."
                )
            return len(seq)
        if "score" in entry:
            return 1
    return 1


def _normalize_history(
    history_raw: List[dict], n_obj: int,
) -> List[Tuple[str, Union[float, Tuple[Optional[float], ...]]]]:
    """Convert JSON history entries to ``(smiles, score)`` tuples.

    For ``n_obj == 1``: ``score`` is a float (or ``None``).
    For ``n_obj >= 2``: ``score`` is a tuple of ``n_obj`` floats
    (or ``None`` for failed evaluations).
    """
    out: List[Tuple[str, Union[float, Tuple[Optional[float], ...]]]] = []
    for i, entry in enumerate(history_raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"history[{i}] must be an object, got {type(entry).__name__}"
            )
        if "smiles" not in entry:
            raise ValueError(f"history[{i}] is missing 'smiles'")
        smi = str(entry["smiles"])
        if "scores" in entry:
            seq = entry["scores"]
            if not isinstance(seq, list):
                raise ValueError(
                    f"history[{i}] 'scores' must be a list; got {type(seq).__name__}"
                )
            if len(seq) != n_obj:
                raise ValueError(
                    f"history[{i}] 'scores' length ({len(seq)}) != n_objectives ({n_obj})"
                )
            tup: List[Optional[float]] = []
            for j, v in enumerate(seq):
                if v is None:
                    tup.append(None)
                else:
                    try:
                        tup.append(float(v))
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"history[{i}] 'scores'[{j}] is not a number: {v!r}"
                        ) from exc
            out.append((smi, tuple(tup)))
        else:
            v = entry.get("score")
            if v is None:
                if n_obj == 1:
                    out.append((smi, None))
                else:
                    raise ValueError(
                        f"history[{i}] has 'score'=null but n_objectives={n_obj}; "
                        f"expected 'scores' (list of {n_obj} values, possibly with nulls)."
                    )
            else:
                try:
                    v_float = float(v)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"history[{i}] 'score' is not a number: {v!r}"
                    ) from exc
                if n_obj == 1:
                    out.append((smi, v_float))
                else:
                    raise ValueError(
                        f"history[{i}] has a bare 'score' but n_objectives={n_obj}; "
                        f"expected 'scores' (list of {n_obj} numbers)."
                    )
    return out


def _resolve_ref_point_for_advisor(
    request: dict, n_obj: int,
) -> Optional[Tuple[float, ...]]:
    """Resolve the ref point for the advisor step.

    For ``n_obj != 2`` the field is ignored (n_obj=1 doesn't use HV;
    n_obj>=3 uses Chebyshev scalarization with per-obj ideal
    points). For ``n_obj == 2`` the user-supplied list is returned
    verbatim, or ``(0.0, 0.0)`` as a conservative default.
    """
    if n_obj != 2:
        return None
    user_ref = request.get("ref_point")
    if user_ref is None:
        return (0.0, 0.0)
    if not isinstance(user_ref, list):
        raise ValueError(
            f"ref_point must be a list of 2 floats; got {type(user_ref).__name__}"
        )
    if len(user_ref) != 2:
        raise ValueError(
            f"ref_point must be length 2 for n_obj=2; got {len(user_ref)}"
        )
    return tuple(float(x) for x in user_ref)


def _build_gp_config(
    request: dict,
    method: str,
    *,
    device: str = "cuda",
) -> GPConfig:
    """Build a :class:`GPConfig` from flat request keys (argparse-dest names).

    The advisor calls this with the full request dict; the GP tuning
    fields are read directly via their flat argparse-dest names
    (``gp_fit_itersteps``, ``gp_learning_rate``, …) and fall back to
    :data:`DEFAULT`. The ``device`` parameter is a provider-setting
    (kwarg-only; not in ``DEFAULT``); the advisor passes its
    ``gp_device`` kwarg value through here. ``device`` defaults to
    ``"cuda"`` if the kwarg is not supplied.
    """
    if not isinstance(request, dict):
        raise ValueError(
            f"'request' must be a JSON object; got {type(request).__name__}"
        )
    return GPConfig(
        impl=GP_IMPL[method],
        device=device,
        fit_n_itersteps=int(request.get("gp_fit_itersteps", DEFAULT["gp_fit_itersteps"])),
        learning_rate=float(request.get("gp_learning_rate", DEFAULT["gp_learning_rate"])),
        min_jitter=float(request.get("gp_min_jitter", DEFAULT["gp_min_jitter"])),
        max_jitter=float(request.get("gp_max_jitter", DEFAULT["gp_max_jitter"])),
        standardize_y=bool(request.get("gp_standardize_y", DEFAULT["gp_standardize_y"])),
        fp_radius=int(request.get("gp_fp_radius", DEFAULT["gp_fp_radius"])),
        fp_n_bits=int(request.get("gp_fp_n_bits", DEFAULT["gp_fp_n_bits"])),
    )


def _json_dumps(obj: Any) -> str:
    """JSON dump with numpy-type fallbacks."""
    return json.dumps(obj, default=_json_default)


def _json_default(obj: Any) -> Any:
    """JSON encoder fallback for numpy types."""
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


def _error_json(exc: BaseException) -> str:
    """Build the canonical error JSON response."""
    return _json_dumps({
        "error": str(exc),
        "error_type": type(exc).__name__,
        "traceback": traceback.format_exc(),
    })


def _external_error_json(exc: BaseException) -> str:
    return _json_dumps({
        "ok": False,
        "items": [],
        "errors": [{"type": type(exc).__name__, "message": str(exc)}],
        "error": str(exc),
        "error_type": type(exc).__name__,
        "traceback": traceback.format_exc(),
    })


__all__ = [
    "score_vina_json",
    "score_nn_json",
    "evaluate_acquisition_json",
    "score_vina",
    "score_nn",
    "evaluate_acquisition",
    "run_search_trajectory",
    "recommend_next_smiles",
    "VALID_METHODS",
    "VALID_METHODS_BO",
    "VALID_METHODS_LDM",
    "VALID_METHODS_RANDOM",
    "PROVIDER_SETTING_KEYS",
    "DEFAULT",
]
