# Adaptive LLM KV-cache quantization

This task searches complete `AdaptiveKVQuantizer` implementations against the
pinned [MLS-Bench task](https://github.com/Imbernoulli/MLS-Bench/tree/cfd57a7e0139c72753e32e31bca593719b098717/harbor/tasks/mls-bench__llm-kv-adaptive-quantization).
The immutable source contract is recorded in `resources/upstream_contract.json`.
Candidates may replace only lines 41-172 of
`transformers-kv-lab/custom_quant_eval.py`; datasets, model, decode loop,
parser, score specification, and the rest of the harness remain fixed.

The experiment contract remains `draft`. Registration, deterministic mock
execution, and isolated tensor checks do not qualify a benchmark claim. Moving
to `qualified` requires both an official-budget seed evaluation and a real
endpoint-generated, acquisition-selected candidate entering the evaluator.

## Candidate and search contract

Deterministic profiles emit complete `AdaptiveKVQuantizer` classes directly.
Endpoint profiles ask the model for a schema-constrained JSON reservoir of
`bit_cap`, key/value group sizes, and residual lengths. The task materializes
each model-selected parameter set into the pinned seed class, then records both
the requested and materialized specifications. Duplicate or already-evaluated
parameter tuples are advanced within the declared finite design space, with
the repair recorded in candidate metadata; this keeps each requested reservoir
full without substituting deterministic proposals for the endpoint call.

Admission enforces the seven exact method signatures, the 64 KiB limit, and an
AST safety policy before deduplicating on a normalized AST hash. The evaluator
then runs a credential-free tensor subprocess and, only for real profiles,
copies the pristine harness and replaces that class.

Every round builds a finite reservoir. `QuantizerSourceEncoder` maps every
admitted class to the versioned 18-dimensional `quantizer_ast_v1`
representation, and the shared exact-RBF GP-UCB selector scores every candidate
before selecting the configured evaluation count.

Mock, tensor, and one-workload tiny runs optimize the explicitly non-official
`selection_score`. Only a successful run of all five workloads exposes and
optimizes `official_score`. The official aggregate is computed by the scoring
implementation bundled in the same pinned MLS-Bench task checkout.

## Verification stages

Registration and service-free mock:

```bash
python scripts/validate_tasks.py --task llm_kv_adaptive_quantization
uv run --locked --project tasks/llm_kv_adaptive_quantization \
  python -m pytest tasks/llm_kv_adaptive_quantization/tests
uv run --locked --project tasks/llm_kv_adaptive_quantization \
  python scripts/check_task_dependencies.py \
  config/llm_kv_adaptive_quantization/mock.yaml --no-optional
uv run --locked --project tasks/llm_kv_adaptive_quantization \
  python scripts/run_ldm_tts.py config/llm_kv_adaptive_quantization/mock.yaml
```

The tensor contract needs a Python environment containing PyTorch. Select it
with `--set args.evaluator-python=/path/to/python` when it differs from the task
environment:

```bash
uv run --locked --project tasks/llm_kv_adaptive_quantization \
  python scripts/run_ldm_tts.py \
  config/llm_kv_adaptive_quantization/preflight.yaml \
  --set args.evaluator-python=/path/to/python
```

## External evaluator layout

Real runs require two paths:

- `upstream-root`: a Git checkout of MLS-Bench at
  `cfd57a7e0139c72753e32e31bca593719b098717` containing the pinned task and its
  `tests/mlsbench_src` scoring package.
- `package-dir`: the prepared `transformers-kv-lab` directory containing
  `custom_quant_eval.py` and `src/`.

The dependency checker verifies the Git commit, critical task-file hashes,
fixed harness digest, evaluator Python modules, endpoint settings when needed,
and configured CUDA devices.

## Real profiles

`official_seed.yaml` evaluates the exact upstream seed on all five workloads.
It requires five concurrently available GPUs, one per workload, and permits up
to 9 hours for the hidden GSM8K job. This is the seed qualification gate, not an
LDM campaign.

`tiny_real.yaml` requests one four-candidate endpoint reservoir, selects one
candidate with GP-UCB, and evaluates one example from HotpotQA on one GPU. Its
`selection_score` is a qualification signal and is not benchmark-comparable.

`extended_tiny_real_20.yaml` repeats that diagnostic topology for twenty
endpoint-generated reservoirs and twenty one-example HotpotQA evaluations. It
uses a separately enforced profile with 20 LLM requests, 80 valid candidates,
20 GP-UCB selections, and 20 benchmark jobs. The extended budget improves
operational coverage but remains non-official and does not change the task's
`draft` qualification.

`official_campaign.yaml` requests one four-candidate endpoint reservoir,
selects one candidate, and evaluates all five official workloads on five GPUs.
The fair comparison axis is one expensive candidate evaluation; that evaluation
contains five benchmark jobs.

All credentials come from `LDM_LLM_URL`, `LDM_LLM_MODEL`, and
`LDM_LLM_API_KEY`. They are not written to configs or run artifacts. A failed
endpoint preflight writes `paused_endpoint_unavailable` and can be continued
with `--resume-from` after connectivity is restored.

Campaign directories contain `campaign.json`, `status.json`, `budget.json`,
`events.jsonl`, `checkpoint.json`, `experiment_contract.json`,
`search_manifest.json`, `selection_record.json`, evaluator manifests and logs,
and `summary.json`.
