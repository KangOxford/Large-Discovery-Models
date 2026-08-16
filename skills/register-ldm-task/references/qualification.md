# LDM Task Qualification Stages

Use this reference after registration scaffolding and before any production
claim or long-running campaign.

Record the current stage in
`tasks/<task_id>/resources/qualification_evidence.json`. Each completed gate
must cite existing repository-relative evidence paths. Validate a claim with
`scripts/validate_tasks.py --task <task_id> --require-stage <stage>`.

## 1. Registered

Required evidence:

- `task.json` is discoverable and the adapter exports `main(argv)`.
- `experiment.json` parses with schema version 1 and matches the task ID.
- The contract is `draft` when benchmark provenance or budgets remain unknown.
- No domain implementation lives in `ldm_task/`.
- No secrets, generated runs, environments, or downloaded assets are tracked.

## 2. Mock Verified

Required evidence:

- Mock generation and evaluation need no service, GPU, dataset, or secret.
- The shared config runner resolves the conventional module and working tree.
- At least one accepted action crosses the same parser used by real inference.
- Collection tests validate canonical IR and prevent provenance/outcome leakage.
- Search and evaluation counters match the mock topology exactly.

## 3. Contract Verified

Required evidence:

- Candidate validation rejects unsafe, malformed, duplicate, and over-budget
  candidates before expensive evaluation.
- Fixed benchmark code remains unchanged outside the editable region.
- Tensor shapes, dtypes, finite outputs, parameter count, and requested devices
  pass the cheap contract evaluator.
- Parallel benchmark jobs map to the intended devices and per-job timeouts.
- Reported, optimized, and diagnostic metrics are distinct and documented. A
  reported metric may also be optimized when it provides a continuous signal.

## 4. Seed Evaluated

Required evidence:

- Benchmark URL, immutable commit, and task path come from a primary source.
- One seed candidate runs with official datasets, hyperparameters, random seed,
  checkpoint selection, epoch cap, training-hour cap, and parameter limit.
- The summary records every reported and diagnostic metric, source candidate,
  evaluator logs, and parameter count.
- The seed observation is explicitly outside or inside future campaign budget.

Only now change `qualification` from `draft` to `qualified`.

## 5. Tiny Campaign Verified

Required evidence:

- `experiment.json` declares the proposal-provider kind and capabilities.
- When `requires_endpoint_preflight` is true, a short authenticated preflight
  validates connectivity, provider/model identity, response shape, and latency
  before search begins. Endpoint checks are not gates for deterministic,
  dataset-backed, or simulator providers that declare the capability false.
- One configured test-time-search reservoir is generated and cheaply validated.
- Acquisition scores every valid candidate and selects exactly the configured
  number for expensive evaluation.
- The selected LDM candidate, not an unmodified benchmark baseline or standalone
  benchmark agent, enters the evaluator.
- `experiment_contract.json`, `budget.json`, `status.json`, search manifest,
  selection record, evaluation manifest, and summary are durable.
- Required-service failures open a circuit and produce a resumable paused state
  without consuming expensive evaluation budget.

## 6. Campaign Qualified

Required evidence:

- The real config selects a named `contract_profile`; dry-run validation rejects
  changes to official settings or campaign budget.
- `scripts/validate_tasks.py --task <task_id> --require-qualified` succeeds.
- `budget.json` separately limits and reports outer iterations, LLM requests,
  valid search candidates, expensive attempts, benchmark jobs, and completions;
  every declared counter is present even when its value is zero.
- Resume reconstructs state from terminal manifests and never repeats a
  completed expensive evaluation.
- Detached launch returns a durable execution handle, unbuffered log, heartbeat
  status, and unique run directory without copying credentials. The handle may
  be a local PID or a remote execution ID.
- Monitoring reports search phase, selected candidate, evaluator phase, device
  assignment, completed/remaining budget, best optimized metric, and official
  reported metric.
- Baseline and LDM comparisons declare the same primary expensive-evaluation
  budget. Extended-budget results are labeled separately.
- Artifact references are run-relative, and completed campaigns provide a
  portable `result.json` plus `trajectory.csv` when the task reports a scalar
  trajectory.

For Delta or another remote execution backend, archive pull, task-aware upload,
blocking cancellation (`kill --wait` or equivalent), and repeated terminal
status/cancel calls must be idempotent. These are backend requirements rather
than repository-local lifecycle implementations.

## Incident Rules

- Endpoint outage: pause and resume after a successful preflight; do not loop
  through the full search space with identical timeouts.
- Candidate contract failure: record a cheap rejection and preserve expensive
  budget.
- Evaluator failure after launch: count an expensive attempt, preserve logs, and
  resume at the next unevaluated selection unless the benchmark says otherwise.
- Contract/profile mismatch: stop before importing the task procedure.
- Missing provenance or official budget: keep qualification at `draft` and do
  not present results as benchmark-comparable.
