# Registering the Adaptive LLM KV-Cache Quantization Task

## Purpose

This document summarizes the implementation experience and reproducible
workflow used to register `llm_kv_adaptive_quantization` as an LDM task, prove
that it can execute through the shared runner, and run a 20-iteration real-LLM
diagnostic campaign.

The work produced three different kinds of evidence:

1. **Registration evidence:** the manifest, adapter layout, experiment
   contract, dependency hook, configs, and tests satisfy the repository's task
   interface.
2. **Operational evidence:** the task completed a real LDM campaign with 20
   Qwen-generated reservoirs, GP-UCB selection, and 20 successful evaluator
   jobs.
3. **Scientific evidence:** the campaign measured one HotpotQA example per
   selected candidate. It is useful diagnostic evidence, but it is explicitly
   non-official, remains `draft`, and is not benchmark-comparable.

The distinction matters. A task can be correctly registered and operationally
sound without having passed the complete official benchmark qualification.

## Final Outcome

The registered task is available through the normal shared-runner interface:

```text
config YAML
  -> task.json manifest discovery
  -> tasks.llm_kv_adaptive_quantization.ldm_task.procedure:main
  -> task-local workflow and dependency adapters
  -> shared LDMEngine campaign runtime
  -> durable run artifacts
```

The completed 20-iteration diagnostic campaign recorded the following exact
counters:

| Counter | Value |
| --- | ---: |
| Outer iterations | 20 |
| LLM requests | 20 |
| Proposal attempts | 20 |
| Valid search candidates | 80 |
| Selected candidates | 20 |
| External evaluations | 20 |
| Expensive evaluation attempts | 20 |
| Successful evaluations | 20 |
| Benchmark jobs | 20 |

Every iteration produced a four-candidate reservoir, selected one candidate,
and completed one one-example HotpotQA evaluation. There were no failed
evaluations and no dropped reservoir candidates.

The best observed candidate was selected in iteration 1:

| Field | Value |
| --- | --- |
| Candidate | `quantizer-dd19f6c383f5` |
| Bit cap | 3 |
| Key group size | 64 |
| Value group size | 64 |
| Residual length | 32 |
| Selection score | 0.4979345 |
| HotpotQA final score | 40.0 |
| Effective KV bits | 3.101562 |
| FP16 KV compression ratio | 5.15869 |

The full derived trajectory is in [progress.csv](./progress.csv), with a
machine-readable campaign summary in [summary.json](./summary.json).

## Registration Architecture

The task follows the repository's manifest-based convention. No shared task
registry or dependency dispatch table had to be edited.

```text
tasks/llm_kv_adaptive_quantization/
|-- task.json                 Manifest and dependency-hook registration
|-- experiment.json           Benchmark, metric, budget, and profile contract
|-- README.md                 Domain and execution documentation
|-- QUICKSTART.md             Staged validation workflow
|-- pyproject.toml            Isolated lightweight task dependencies
|-- ldm_task/
|   |-- procedure.py          Stable shared-runner adapter
|   `-- dependencies.py       Dependency-check adapter
|-- core/
|   |-- candidate.py          Admission, normalization, and materialization
|   |-- proposals.py          Deterministic and endpoint reservoir expansion
|   |-- surrogate.py          Versioned AST feature encoder
|   |-- evaluator.py          Mock, tensor, and MLS-Bench evaluators
|   |-- contract_worker.py    Isolated tensor contract execution
|   `-- workflow.py           LDMEngine assembly and durable runtime
|-- resources/
|   |-- upstream_contract.json
|   `-- seed_quantizer.py
`-- tests/test_procedure.py

config/llm_kv_adaptive_quantization/
|-- mock.yaml
|-- preflight.yaml
|-- tiny_real.yaml
|-- extended_tiny_real_20.yaml
|-- official_seed.yaml
`-- official_campaign.yaml
```

The stable external adapter is intentionally small. It delegates argument
parsing, task description, and execution to `core/workflow.py`; candidate,
proposal, surrogate, and evaluator details remain private to the task.

Relevant source files:

- [Task manifest](../../../tasks/llm_kv_adaptive_quantization/task.json)
- [Experiment contract](../../../tasks/llm_kv_adaptive_quantization/experiment.json)
- [Task guide](../../../tasks/llm_kv_adaptive_quantization/README.md)
- [Quickstart](../../../tasks/llm_kv_adaptive_quantization/QUICKSTART.md)
- [Procedure adapter](../../../tasks/llm_kv_adaptive_quantization/ldm_task/procedure.py)
- [Workflow implementation](../../../tasks/llm_kv_adaptive_quantization/core/workflow.py)
- [Task tests](../../../tasks/llm_kv_adaptive_quantization/tests/test_procedure.py)

## Workflow That Proved Reliable

### 1. Pin the upstream benchmark before designing the search

The first important decision was to treat MLS-Bench as an immutable external
contract rather than copying its behavior loosely.

The task records:

- Source repository: `https://github.com/Imbernoulli/MLS-Bench`
- Commit: `cfd57a7e0139c72753e32e31bca593719b098717`
- Upstream task: `harbor/tasks/mls-bench__llm-kv-adaptive-quantization`
- Editable file: `transformers-kv-lab/custom_quant_eval.py`
- Editable region: lines 41 through 172, the `AdaptiveKVQuantizer` class
- Benchmark model: `Qwen/Qwen2.5-3B-Instruct`
- Seed: 42
- Official workloads: three LongBench tasks, NeedleBench, and GSM8K

`resources/upstream_contract.json` stores hashes for critical task metadata,
the pristine harness, the fixed harness region, and the seed class. The
dependency checker refuses a real run when the checkout commit or fixed
harness digest differs from the pin.

This made the ownership boundary explicit: LDM may change the quantizer class,
but it may not change datasets, scoring, model, decode loop, or the surrounding
evaluation harness.

### 2. Register the conventional adapter, not a custom launcher

The minimal registration manifest is:

```json
{
  "schema_version": 1,
  "task_id": "llm_kv_adaptive_quantization",
  "description": "Optimize adaptive low-bit KV-cache quantization policies against MLS-Bench quality and compression objectives.",
  "dependency_checker": "tasks.llm_kv_adaptive_quantization.ldm_task.dependencies:check_dependencies"
}
```

The shared runner discovers this manifest and calls:

```text
tasks.llm_kv_adaptive_quantization.ldm_task.procedure:main
```

That kept registration independent of task implementation. It also meant the
same configs could use the normal dependency checker, dry-run resolution,
campaign runtime, and artifact conventions.

### 3. Encode the evidence boundary in `experiment.json`

The experiment contract separates reported, optimized, and diagnostic metrics:

- `official_score` is the official five-workload aggregate and is only exposed
  when all five workloads complete.
- `selection_score` is an explicitly non-official objective for mock, tensor,
  and tiny diagnostic runs.
- `final_score`, `effective_kv_bits`, `kv_compression_ratio`, and
  `runtime_seconds` remain visible per workload.
- `state_tensor_elements` checks that candidate code does not retain an
  unreasonable amount of tensor state during contract evaluation.

Named profiles lock arguments and budget ledgers. The important profiles are:

| Profile | Purpose | Evaluation shape | Qualification |
| --- | --- | --- | --- |
| `tiny_real_qualification` | First real proposal/evaluator check | 1 generated reservoir, 1 selected candidate, 1 HotpotQA example | Non-official |
| `extended_tiny_real_20` | Operational campaign coverage | 20 reservoirs, 20 selections, 20 one-example HotpotQA jobs | Non-official |
| `official_seed` | Seed qualification gate | Exact upstream seed, all 5 workloads | Official-budget gate |
| `official_campaign` | Full LDM-selected evaluation | 1 generated reservoir, 1 selected candidate, all 5 workloads | Official campaign |

The task remains `qualification: draft`. Neither successful registration nor
the 20-run diagnostic is a reason to change it to `qualified`.

### 4. Make endpoint output structured, finite, and auditable

The endpoint does not return an unrestricted codebase. It returns a
schema-constrained reservoir of quantizer specifications:

```json
{
  "bit_cap": 3,
  "key_group_size": 64,
  "value_group_size": 64,
  "residual_length": 32
}
```

Each accepted specification is materialized into the pinned seed quantizer
class. Candidate metadata records both the requested specification and the
materialized specification, plus whether repair was needed.

The finite design space allows a duplicate or previously evaluated proposal to
advance deterministically to an unused tuple. This preserves the requested
four-candidate reservoir without hiding repair or substituting a random
fallback.

Admission then checks:

- The complete `AdaptiveKVQuantizer` class is present.
- All seven required methods have exact signatures.
- Source is at most 64 KiB.
- The abstract syntax tree passes the safety policy.
- Normalized AST hashes are used for deduplication.

The resulting boundary is narrow enough to validate but still lets the search
change quantization bits, grouping, and residual policy through real model
proposals.

### 5. Use a versioned surrogate representation and shared acquisition

`QuantizerSourceEncoder` maps every admitted class into the versioned
18-dimensional `quantizer_ast_v1` feature vector. The shared exact-RBF GP-UCB
selector then scores every candidate in the finite reservoir and records:

- Predicted mean.
- Predicted standard deviation.
- UCB acquisition score.
- Fit state (`prior`, `fallback`, or `fitted`).
- History size and best observed score.
- The selected candidate ID.

The first campaign iteration used the prior, the second used a one-observation
fallback, and iterations 3 through 20 used the fitted surrogate. Persisting
these predictions in `selection_record.json` made it possible to verify that
the evaluator received the acquisition-selected candidate rather than merely
the first generated candidate.

### 6. Validate with progressively more expensive evaluator lanes

The task implements four useful validation levels:

1. **Registration and dry-run:** validates manifests, profile arguments, and
   resolved runner plans without importing expensive scientific dependencies.
2. **Mock campaign:** exercises reservoir construction, encoding, GP-UCB,
   evaluation accounting, checkpoints, and artifacts without external
   services.
3. **Tensor preflight:** executes candidate code in an isolated subprocess with
   PyTorch tensors and checks shape, dtype, device, finite output, bit
   accounting, and retained state.
4. **MLS-Bench evaluation:** first runs the tensor contract, then copies the
   pristine harness, replaces only the quantizer class, and launches the
   configured benchmark workloads.

This ordering prevented model downloads and GPU jobs from becoming the first
place that class-signature, tensor-shape, or artifact bugs appeared.

### 7. Make runtime state durable and resumable

The task uses `CampaignRuntime` and the shared `LDMEngine` instead of writing a
task-local loop. A run persists:

- `campaign.json`
- `status.json`
- `budget.json`
- `events.jsonl`
- `checkpoint.json`
- `experiment_contract.json`
- `ldm_task_spec.json`
- `search_manifest.json`
- `selection_record.json`
- Per-candidate evaluation manifests and logs
- `summary.json`

Tests verify that resuming a completed run does not repeat its expensive
evaluation. Endpoint preflight failures become
`paused_endpoint_unavailable`, allowing a later `--resume-from` rather than
losing or double-spending campaign state.

## Recommended Verification Sequence

Run these from the repository root. `uv` is used so the task's checked-in lock
file controls its lightweight Python environment.

```bash
python scripts/validate_tasks.py --task llm_kv_adaptive_quantization

uv run --locked --project tasks/llm_kv_adaptive_quantization \
  python -m pytest tasks/llm_kv_adaptive_quantization/tests

uv run --locked --project tasks/llm_kv_adaptive_quantization \
  python scripts/check_task_dependencies.py \
  config/llm_kv_adaptive_quantization/mock.yaml --no-optional

uv run --locked --project tasks/llm_kv_adaptive_quantization \
  python scripts/run_ldm_tts.py \
  config/llm_kv_adaptive_quantization/mock.yaml --dry-run

uv run --locked --project tasks/llm_kv_adaptive_quantization \
  python scripts/run_ldm_tts.py \
  config/llm_kv_adaptive_quantization/mock.yaml
```

Then select an evaluator Python that imports PyTorch and run the isolated
tensor contract:

```bash
uv run --locked --project tasks/llm_kv_adaptive_quantization \
  python scripts/run_ldm_tts.py \
  config/llm_kv_adaptive_quantization/preflight.yaml \
  --set args.evaluator-python=/path/to/python
```

Before any real campaign, run the dependency check and dry-run against the
exact config and overrides that will be launched:

```bash
uv run --locked --project tasks/llm_kv_adaptive_quantization \
  python scripts/check_task_dependencies.py \
  config/llm_kv_adaptive_quantization/tiny_real.yaml \
  --set args.upstream-root=/path/to/pinned/mls-bench \
  --set args.package-dir=/path/to/transformers-kv-lab \
  --set args.evaluator-python=/path/to/evaluator/python

uv run --locked --project tasks/llm_kv_adaptive_quantization \
  python scripts/run_ldm_tts.py \
  config/llm_kv_adaptive_quantization/tiny_real.yaml --dry-run \
  --set args.upstream-root=/path/to/pinned/mls-bench \
  --set args.package-dir=/path/to/transformers-kv-lab \
  --set args.evaluator-python=/path/to/evaluator/python
```

Provider credentials must be supplied only through environment variables:

```text
LDM_LLM_URL
LDM_LLM_MODEL
LDM_LLM_API_KEY
```

Do not put credentials in YAML, command arguments, run artifacts, logs, or this
document.

## Delta Sandbox Campaign Workflow

This section documents the second 20-iteration demo run. It used Delta CLI to
host both the private proposal endpoint and the benchmark process, while the
task still ran through the same registered adapter and shared `LDMEngine`
described above. No external model API or `api_credential.json` was used.

The two models had separate roles and must not be conflated:

| Role | Model |
| --- | --- |
| Proposal generation | Local shared `Qwen3.5-9B` checkpoint served by vLLM |
| Benchmark evaluation | `Qwen/Qwen2.5-3B-Instruct` fixed by MLS-Bench |

### 1. Check Delta configuration before allocating resources

Run both read-only checks first:

```bash
delta-cli config show
delta-cli auth status
```

Continue only when both commands return exit code zero with top-level
`ok: true`. The displayed credential must remain masked.

### 2. Create exactly one GPU sandbox

The working image needs the `r3l` vLLM environment used by the Qwen launcher
and enough GPU memory to keep the 9B proposal model resident while the 3B
benchmark model runs:

```bash
delta-cli sandbox create \
  --image image.yangtzeailab.com/opensandbox/vllm_0.27.1:juicefs \
  --cpu 16 \
  --memory 64Gi \
  --gpu 1 \
  --gpu-mem 80000 \
  --max-life 180 \
  --no-auto-cleanup
```

Record `data.sandbox_id` from the response. Do not invent an ID or submit a
second create request if the first request has an uncertain result; use a
read-only `sandbox list` call to reconcile server state first.

Resolve the authoritative working directory once:

```bash
delta-cli sandbox working-directory <sandbox-id>
```

Every later `<working-directory>` placeholder means the exact `data.path`
returned by this command.

### 3. Stage bounded inputs with explicit destinations

Upload the repository components needed by the runner rather than transferring
an unrelated working tree. The pinned MLS-Bench checkout remains a separate
input:

```bash
delta-cli sandbox upload <sandbox-id> \
  --source ldm_tts \
  --target <working-directory>/repo/ldm_tts

delta-cli sandbox upload <sandbox-id> \
  --source config \
  --target <working-directory>/repo/config

delta-cli sandbox upload <sandbox-id> \
  --source scripts \
  --target <working-directory>/repo/scripts

delta-cli sandbox upload <sandbox-id> \
  --source tasks/llm_kv_adaptive_quantization \
  --target <working-directory>/repo/tasks/llm_kv_adaptive_quantization

delta-cli sandbox write <sandbox-id> \
  --source tasks/__init__.py \
  --path <working-directory>/repo/tasks/__init__.py

delta-cli sandbox upload <sandbox-id> \
  --source /path/to/pinned/mls-bench \
  --target <working-directory>/mls-bench
```

Use individual `sandbox write` calls for important small files when exact
placement matters. During the plotting follow-up, relative `write-multiple`
targets were reported as successful but did not land under the current
sandbox's working directory.

### 4. Prepare and validate both Python environments

The evaluator and proposal server intentionally use different Python
environments. Install benchmark dependencies into `/opt/conda/bin/python`:

```bash
delta-cli sandbox run-bg <sandbox-id> \
  --command "bash <working-directory>/repo/scripts/delta_llm_kv_eval_setup.sh" \
  --timeout 1800 \
  --wait
```

Then validate the shared Qwen checkpoint with the vLLM environment:

```bash
delta-cli sandbox run <sandbox-id> \
  --command "/opt/conda/envs/r3l/bin/python <working-directory>/repo/scripts/delta_qwen35_env_probe.py" \
  --timeout 60
```

The actual shared checkpoint was
`/workspace/577908796194689024/models/Qwen3.5-9B`. The probe verifies the
safetensors index, every referenced shard, absence of partial downloads,
runtime imports, CUDA visibility, and the GPU name. If the shared workspace
root differs, update the two Qwen helper scripts to the real path before they
are uploaded; do not download a second copy by default.

### 5. Start and verify the private Qwen endpoint

Start vLLM as a background job because it must remain alive for the entire
campaign:

```bash
delta-cli sandbox run-bg <sandbox-id> \
  --command "bash <working-directory>/repo/scripts/delta_launch_qwen35.sh" \
  --timeout 21600
```

Retain the returned server `execution_id` for log inspection and cancellation.
The launcher serves `Qwen3.5-9B` on sandbox-local
`http://127.0.0.1:8000/v1` with:

```text
--language-model-only
--max-model-len 4096
--gpu-memory-utilization 0.80
--enforce-eager
--reasoning-parser qwen3
```

Probe model discovery and a deterministic `OK` response from inside the same
sandbox:

```bash
delta-cli sandbox run <sandbox-id> \
  --command "/opt/conda/envs/r3l/bin/python <working-directory>/repo/scripts/delta_probe_qwen35.py" \
  --timeout 180
```

The endpoint is not exposed to the host. `LDM_LLM_API_KEY=EMPTY` is only a
non-secret placeholder required by the local OpenAI-compatible client.

### 6. Gate the full run with one real iteration

Use `tiny_real.yaml` with the same paths and local endpoint that the full run
will use:

```bash
delta-cli sandbox run-bg <sandbox-id> \
  --command "cd <working-directory>/repo && env PYTHONPATH=<working-directory>/repo LDM_LLM_URL=http://127.0.0.1:8000/v1 LDM_LLM_MODEL=Qwen3.5-9B LDM_LLM_API_KEY=EMPTY /opt/conda/bin/python scripts/run_ldm_tts.py config/llm_kv_adaptive_quantization/tiny_real.yaml --set args.upstream-root=<working-directory>/mls-bench --set args.package-dir=<working-directory>/mls-bench/harbor/tasks/mls-bench__llm-kv-adaptive-quantization/environment/_scaffold/transformers-kv-lab --set args.evaluator-python=/opt/conda/bin/python --set args.out-dir=<working-directory>/runs --set args.run-name=local_qwen35_9b_tiny_real" \
  --timeout 7200 \
  --wait
```

Authorize the 20-iteration run only after this job has exit code zero and its
`status.json`, `budget.json`, and evaluation manifest all report one successful
real evaluation.

### 7. Launch the locked 20-iteration campaign

The campaign helper performs the dependency preflight, sets the local endpoint
environment, generates a timestamped run name, invokes the registered task,
and prints a bounded JSON summary at the end:

```bash
delta-cli sandbox run-bg <sandbox-id> \
  --command "/opt/conda/bin/python <working-directory>/repo/scripts/delta_llm_kv_local_campaign.py --repo <working-directory>/repo --upstream-root <working-directory>/mls-bench --package-dir <working-directory>/mls-bench/harbor/tasks/mls-bench__llm-kv-adaptive-quantization/environment/_scaffold/transformers-kv-lab --output-root <working-directory>/runs --evaluator-python /opt/conda/bin/python" \
  --timeout 21600 \
  --wait
```

The effective profile remains `extended_tiny_real_20`: 20 iterations, four
valid candidates per reservoir, one GP-UCB selection per iteration, one
HotpotQA example per selected candidate, and exactly 20 benchmark jobs. The
helper sets proposal timeout to 600 seconds and proposal output to at most 1024
tokens.

### 8. Monitor durable state, not only the event stream

Keep the campaign `execution_id` from the `run-bg` init frame. Delta logs show
process health, while task artifacts establish scientific and accounting
progress:

```bash
delta-cli sandbox logs <sandbox-id> \
  --execution-id <campaign-execution-id>

delta-cli sandbox read <sandbox-id> \
  --path <working-directory>/runs/<run-name>/status.json

delta-cli sandbox read <sandbox-id> \
  --path <working-directory>/runs/<run-name>/budget.json
```

Do not submit the campaign again after a client timeout or lost stream. First
inspect the execution state and durable files; a remote process may still be
running.

### 9. Pull and validate the completed evidence

After the helper reports `status: completed`, pull the exact run directory:

```bash
delta-cli sandbox pull <sandbox-id> \
  --source <working-directory>/runs/<run-name>/ \
  --target tasks/llm_kv_adaptive_quantization/runs/<run-name>/ \
  --recursive
```

The accepted trial-2 run name was
`local_qwen35_9b_extended_tiny_real_20_20260816_163130`. Acceptance required
agreement across `status.json`, `budget.json`, `events.jsonl`,
`selection_record.json`, all 20 evaluation manifests, and the helper's final
JSON. Exit code zero alone was not sufficient.

The plots were derived from those pulled artifacts, without rerunning the
campaign:

```bash
python scripts/plot_llm_kv_campaign.py \
  --run-dir tasks/llm_kv_adaptive_quantization/runs/local_qwen35_9b_extended_tiny_real_20_20260816_163130 \
  --output-dir data/generated/llm_kv_quantization_trial2 \
  --source-label tasks/llm_kv_adaptive_quantization/runs/local_qwen35_9b_extended_tiny_real_20_20260816_163130
```

When the host lacks Matplotlib, run only this display step in a short CPU
sandbox, pull the eight generated files, and destroy that sandbox. Do not
submit another LDM campaign to make the plots.

### 10. Stop the server and destroy the task-owned sandbox

Cancel the long-running vLLM execution, then destroy the exact sandbox created
for this workflow:

```bash
delta-cli sandbox cancel <sandbox-id> \
  --execution-id <server-execution-id>

delta-cli sandbox kill <sandbox-id>

delta-cli sandbox list --sandbox-id <sandbox-id>
```

Cleanup is complete only when the kill command returns top-level `ok: true`
and the sandbox is no longer running. A silent or hung kill/finish client is
not success; reconcile server state and escalate the lifecycle failure instead
of assuming the resource was released.

## What the 20 Runs Showed

The accepted trial-2 campaign completed all 20 iterations in roughly 8 minutes
23 seconds from `status.json` start to finish. Proposal latency stayed between
5.02 and 5.95 seconds. The first evaluation took 153.04 seconds because it
included model download and cache warmup; the median of iterations 2 through
20 was 5.76 seconds.

Observed ranges were:

| Metric | Minimum | Maximum |
| --- | ---: | ---: |
| Selection score | 0.266008086 | 0.528205148 |
| HotpotQA final score | 10.0 | 40.0 |
| Effective KV bits | 2.054688 | 4.375 |
| FP16 compression ratio | 3.657143 | 7.787072 |
| Evaluation runtime | 5.435379 s | 153.035805 s |

The final iteration produced the best selection objective:

| Field | Value |
| --- | --- |
| Candidate | `quantizer-fcaa157c443e` |
| Bit cap | 2 |
| Key group size | 16 |
| Value group size | 64 |
| Residual length | 128 |
| Selection score | 0.528205148 |
| HotpotQA final score | 33.333333 |
| Effective KV bits | 2.4375 |
| FP16 KV compression ratio | 6.564103 |

This is useful operational evidence: the campaign generated, encoded, scored,
selected, evaluated, and persisted all 20 rounds with no failed evaluation and
no dropped candidate. It is not evidence that 20 one-example measurements
provide a statistically reliable ranking of quantizers.

The generated plots make the behavior inspectable:

- [Objective progress](../../data/generated/llm_kv_quantization_trial2/progress.png)
- [HotpotQA, compression, and runtime diagnostics](../../data/generated/llm_kv_quantization_trial2/metrics.png)
- [GP-UCB acquisition trace](../../data/generated/llm_kv_quantization_trial2/acquisition.png)
- [Per-iteration data](../../data/generated/llm_kv_quantization_trial2/progress.csv)
- [Machine-readable summary](../../data/generated/llm_kv_quantization_trial2/summary.json)

PDF versions are stored beside each PNG. The reusable plotting utility is
[plot_llm_kv_campaign.py](../../scripts/plot_llm_kv_campaign.py). All outputs
retain `draft_non_official_not_benchmark_comparable`; the run remains a
one-example HotpotQA diagnostic rather than an official benchmark.

## Practical Lessons

### Registration is an interface contract, not a benchmark claim

The most important lesson was to keep registration success, runnable LDM
behavior, and scientific qualification separate. Encoding `draft` and the
non-official metric directly into the experiment contract was safer than
relying on documentation alone.

### Pin both source identity and the fixed harness region

A Git commit check alone is not sufficient when a prepared evaluator package
is copied or modified outside the checkout. Checking critical upstream hashes
and the fixed harness digest catches accidental benchmark drift while still
allowing the declared quantizer class region to change.

### Use the same engine path for mock and real runs

Mock-only bespoke code would have provided weak evidence. Driving mock,
preflight, tiny, and extended runs through the same adapter and LDMEngine
exercised budgets, events, checkpoints, selection records, and resume behavior
before spending real evaluator jobs.

### Treat endpoint responses as untrusted structured input

Schema-constrained JSON reduced parsing ambiguity, but admission, exact method
signatures, AST safety, source-size limits, and deduplication were still
necessary. Recording repair decisions kept the search trace auditable.

### Lock campaign topology in named profiles

The profile should enforce iteration count, reservoir size, selected count,
workloads, devices, model ID, example limit, and timeout. Without locked args,
a run can retain the same name while silently changing its scientific cost or
meaning.

### Persist state before trusting terminal output

Streaming logs are useful for monitoring, but `status.json`, `budget.json`,
`events.jsonl`, selection records, and evaluation manifests are the durable
evidence. Completion was accepted only after those files agreed on 20
successful iterations and jobs.

### Separate cold-start runtime from steady-state runtime

The first evaluator job included download and warmup overhead and was more than
eight times slower than cached runs. Reporting it as ordinary per-candidate
runtime would distort capacity planning. The plot and summary therefore label
it explicitly.

### Be explicit about sandbox paths and artifact transfer

Delta CLI returns authoritative sandbox paths and IDs. Relative staging paths
can behave differently across images, and a directory pull can fail partially.
Use explicit paths for important files, validate local sizes and formats after
pulling, and destroy only the sandbox created for the current task after all
required artifacts are safely local.

## What I Would Do Next

The next scientifically meaningful step is not a larger one-example campaign.
It is to execute the qualification gates already represented by the task:

1. Run `official_seed.yaml` on all five workloads and verify the exact upstream
   seed under the official budget.
2. Run `tiny_real.yaml` as the minimal endpoint-generated,
   acquisition-selected evaluator gate if a clean independent rerun is desired.
3. Run `official_campaign.yaml` only with five available GPUs and the full
   datasets, retaining every workload manifest and the contract snapshot.
4. Change `qualification` from `draft` only after both required gates pass and
   their evidence is reviewed.

For future task registrations, I would also establish the artifact schema and
resume tests immediately after scaffolding. Those tests provided unusually
high leverage here because they caught accounting and lifecycle issues before
the real model and benchmark environments entered the picture.

## Reproducibility Checklist

- [ ] `task.json` validates and resolves the conventional procedure module.
- [ ] `experiment.json` pins benchmark provenance and honestly states
      qualification.
- [ ] Mock, tensor, tiny-real, official-seed, and official-campaign meanings are
      separate.
- [ ] Provider secrets are environment-only.
- [ ] Endpoint responses pass schema, admission, safety, and deduplication.
- [ ] Surrogate feature version and acquisition predictions are persisted.
- [ ] Real evaluation replaces only the declared quantizer class region.
- [ ] Profile-locked counters match the final budget ledger.
- [ ] Resume does not repeat completed expensive evaluations.
- [ ] Generated runs remain under ignored task `runs/` directories.
- [ ] Pulled campaign artifacts are validated before sandbox destruction.
- [ ] Non-official results are not presented as official benchmark results.
