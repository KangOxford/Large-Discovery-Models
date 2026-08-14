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

The real 20-run diagnostic used Delta CLI for the heavy environment. The
proposal model and benchmark model were different and should not be confused:

| Role | Model |
| --- | --- |
| Proposal generation | Shared `Qwen3.5-9B` checkpoint |
| Benchmark evaluation | `Qwen/Qwen2.5-3B-Instruct` fixed by MLS-Bench |

The practical lifecycle was:

1. Check Delta configuration and authentication.
2. Create one task-owned sandbox with the required GPU and memory.
3. Resolve and retain the real `sandbox_id` and working directory returned by
   Delta CLI.
4. Stage the repository task, pinned MLS-Bench checkout, prepared
   `transformers-kv-lab`, and evaluator environment.
5. Start the Qwen proposal endpoint as a long-running background execution.
6. Run registration, dependency, mock, and endpoint checks inside the same
   heavy environment.
7. Run a one-iteration confirmation before authorizing the 20-iteration
   profile.
8. Launch `extended_tiny_real_20` with one GPU, one HotpotQA example per
   selected candidate, and the locked 20-job budget.
9. Monitor durable events, status, checkpoints, evaluation manifests, and
   counters rather than relying only on streaming logs.
10. Pull campaign artifacts to the host and verify terminal counts before
    destroying the task-owned sandbox.

The endpoint remained private to the sandbox. The campaign process called it
through the task's OpenAI-compatible proposal client; no credentials were
stored in the checked-in configs or pulled run artifacts.

## What the 20 Runs Showed

The campaign completed in roughly 12 minutes from its first to last durable
event. Proposal latency stayed near five seconds per iteration. The first
benchmark evaluation took 153.2 seconds because it included model download and
cache warmup; the median of iterations 2 through 20 was 18.4 seconds.

Observed ranges were:

| Metric | Minimum | Maximum |
| --- | ---: | ---: |
| Selection score | 0.268271636 | 0.4979345 |
| HotpotQA final score | 9.52381 | 40.0 |
| Effective KV bits | 2.4375 | 4.75 |
| FP16 compression ratio | 3.368421 | 6.564103 |
| Evaluation runtime | 17.450805 s | 153.204105 s |

The best-so-far line is flat because the first selected candidate remained the
incumbent. That is still useful operational evidence: the campaign generated,
encoded, scored, selected, evaluated, and persisted all 20 rounds correctly.
It is not evidence that 20 one-example measurements provide a statistically
reliable ranking of quantizers.

The generated plots make the behavior inspectable:

- [Objective progress](./progress.png)
- [HotpotQA, compression, and runtime diagnostics](./metrics.png)
- [GP-UCB acquisition trace](./acquisition.png)

PDF versions are stored beside each PNG. The reusable plotting utility is
[plot_llm_kv_campaign.py](../../../scripts/plot_llm_kv_campaign.py).

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

