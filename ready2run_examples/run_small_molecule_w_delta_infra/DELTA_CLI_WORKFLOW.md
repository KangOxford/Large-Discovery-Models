# Delta CLI Small-Molecule Preflight and Campaign Runbook

## Purpose

This document records the practical workflow, results, failures, recovery
steps, and engineering lessons from running the repository's real KRAS G12D
small-molecule example through Delta CLI.

The work had two successful stages:

1. A one-evaluation preflight that exercised real Qwen inference, real
   AutoDock Vina docking against KRAS G12D structure `8UN5`, and the updated
   G12D activity model.
2. A seed-42 EHVI campaign configured for 100 evaluations. The campaign was
   stopped at the user's request after 30 complete evaluations, and its
   incremental checkpoint was recovered and plotted.

The primary artifacts are:

- [Partial campaign result](./result.json)
- [Reconstructed evaluation history](./history.json)
- [Compact objective trajectory](./progress.csv)
- [Cumulative hypervolume plot](./progress.png)
- [Cumulative hypervolume data](./progress_hypervolume.csv)
- [Raw incremental checkpoint](./rounds.jsonl)
- [Resolved campaign configuration](./config.json)

## Evidence Boundary

The final artifacts represent a **partial 30/100 campaign**, not a completed
100-evaluation campaign. `result.json` records:

```json
{
  "status": "partial",
  "completion_reason": "stopped_at_user_request",
  "evaluations": 30,
  "budget": 100
}
```

The following components were real rather than mocked:

- Qwen3.5-9B inference from the shared Delta workspace.
- AutoDock Vina scoring against receptor PDB `8UN5`, chain `A`.
- The updated `best_g12d_model.joblib` activity model.
- The repository's SMILES validation, canonicalization, Gaussian-process,
  EHVI, reservoir construction, and trace-recording code.

All remote execution, file transfer, process inspection, artifact retrieval,
and sandbox lifecycle operations went through `delta-cli sandbox`. No
`delta-cli science` operation was used for this run.

## Final Configuration

| Setting | Value |
|---|---|
| Sandbox image | `image.yangtzeailab.com/opensandbox/vllm_0.27.1:juicefs` |
| Sandbox resources | 16 CPU, 64 GiB RAM, 1 GPU, 80,000 MiB GPU memory |
| Maximum sandbox life | 360 minutes |
| Shared workspace | `/workspace/577908796194689024` |
| Model checkpoint | `/workspace/577908796194689024/models/Qwen3.5-9B` |
| Served model name | `Qwen3.5-9B` |
| vLLM context length | 16,384 tokens |
| vLLM GPU utilization | `0.38` |
| Receptor | PDB `8UN5`, chain `A` |
| Docking engine | AutoDock Vina 1.1.2 |
| Activity model SHA-256 | `a4c15c1124eced2e8dc80a18fdf94752da106168209d804002b0defbf63986ed` |
| Search method | `m1_stratified_direct_llm_oversample_sir` |
| Acquisition | EHVI, 128 samples |
| Objective directions | Minimize Vina; maximize predicted activity |
| Reference point | `[0.0, 5.0]` |
| Seed | `42` |
| Requested budget | 100 |
| Completed budget | 30 |
| Batch size | 1 |
| Maximum candidates per round | 32 |
| Direct LLM proposals requested per round | 128 |
| GP device | CPU |

The updated activity model was verified before both preflight and campaign
execution. Its checksum was never inferred from a filename or metadata alone;
the uploaded binary was hashed inside the sandbox.

## System Architecture

```text
Local repository
  |
  | delta-cli sandbox write/upload
  v
Delta GPU sandbox
  |
  +-- Qwen3.5-9B served by vLLM on sandbox localhost:8011
  |
  +-- Repository search loop
  |     |
  |     +-- Qwen JSON proposal chunks
  |     +-- isolated RDKit canonicalization workers
  |     +-- candidate reservoir and deduplication
  |     +-- CPU Gaussian process and EHVI selection
  |
  +-- Real objective evaluation
  |     |
  |     +-- AutoDock Vina against 8UN5
  |     +-- updated G12D activity model
  |
  +-- rounds.jsonl incremental checkpoint
  |
  | delta-cli sandbox pull
  v
data/generated/delta_small_molecule_campaign_30
```

The vLLM endpoint was private to the sandbox. The host did not call it
directly. The campaign launcher and proposal subprocesses accessed the local
OpenAI-compatible endpoint from inside the same sandbox.

## End-to-End Workflow

### 1. Allocate one task-owned sandbox

The sandbox was created with the requested image, one GPU, and a six-hour
maximum lifetime. The returned sandbox ID and working directory were treated
as authoritative.

Conceptually, the create operation was:

```bash
delta-cli sandbox create \
  --image image.yangtzeailab.com/opensandbox/vllm_0.27.1:juicefs \
  --cpu 16 \
  --memory 64Gi \
  --gpu 1 \
  --gpu-mem 80000 \
  --max-life 360 \
  --no-auto-cleanup
```

The shared model was not copied. It was consumed in place from:

```text
/workspace/577908796194689024/models/Qwen3.5-9B
```

### 2. Transfer the repository in bounded components

One monolithic repository upload was approximately 228 MB and failed because
the proxy returned HTML instead of the JSON envelope expected by Delta CLI.
The successful strategy was to split the upload into independently verifiable
components:

- `ldm_tts/`
- `config/`
- `scripts/`
- `tasks/small_molecule/`
- root `pyproject.toml`
- root `README.md`
- `tasks/__init__.py`

The 46.4 MB small-molecule task upload contained 89 files, including the
45.1 MB updated model artifact. Delta's upload integrity check reported:

```text
Upload integrity OK: 89 files, 46368184 bytes
```

Small root files were added with explicit absolute destinations using
`write-multiple`. This avoided ambiguity about the sandbox working directory.

Practical rule: use a directory upload for cohesive trees and explicit
`write` or `write-multiple` calls for small entry-point files. Do not retry a
large opaque upload unchanged after a proxy parsing failure.

### 3. Validate remote inputs before starting the GPU

The first remote check verified all three critical inputs:

1. The uploaded G12D model checksum matched the expected SHA-256.
2. The launcher contained `--max-model-len 16384`.
3. The shared Qwen checkpoint directory existed.

The checksum result was:

```text
a4c15c1124eced2e8dc80a18fdf94752da106168209d804002b0defbf63986ed
```

The launcher additionally parsed `model.safetensors.index.json`, verified
every referenced shard, and rejected the checkpoint if any `.partial` file
was present.

### 4. Provision a real Vina CLI

The new sandbox did not inherit the previous sandbox's local `vina-env`, so
the first preflight launcher exited at:

```bash
test -x "$vina_bin"
```

Installing the PyPI package `vina==1.2.7` was not sufficient. That package
provided the Python API but did not install the expected `bin/vina` command.

The first shared executable found under `/mnt/data0/shared` was also unusable.
Its first bytes were a Mach-O header, and Linux returned:

```text
cannot execute binary file: Exec format error
```

The successful executable was a Linux ELF build:

```text
/mnt/data0/shared/ldm_tilted_case2_three_methods/
  tools/autodock_vina_1_1_2_linux_x86/bin/vina
```

It was copied to the launcher-owned path and verified:

```text
AutoDock Vina 1.1.2 (May 11, 2011)
```

Practical rule: verify both executability and binary format. A matching
filename and executable mode do not imply that the binary targets Linux or
the sandbox CPU architecture.

### 5. Start Qwen with a bounded context and KV cache

The launcher started vLLM as a child process and polled its model-list endpoint
until a deterministic health prompt returned exactly `OK`.

```bash
/opt/conda/envs/r3l/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --model /workspace/577908796194689024/models/Qwen3.5-9B \
  --served-model-name Qwen3.5-9B \
  --host 127.0.0.1 \
  --port 8011 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.38 \
  --enforce-eager \
  --reasoning-parser qwen3
```

The launcher set:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0
CUDA_VISIBLE_DEVICES=0
LLM_DISABLE_THINKING=1
```

It also applied an idempotent deferred-annotation compatibility patch to the
installed FlashInfer module when needed.

### 6. Use a real one-evaluation preflight as a hard gate

The preflight did more than check imports. It ran four layers of validation:

1. Static file, checksum, model-shard, Vina, and environment checks.
2. A repository contract dry run with budget 1.
3. Four real Qwen proposal chunks.
4. One real selected molecule scored by both Vina and the G12D model.

The command was launched as a long background operation and waited through
the Delta SSE stream:

```bash
delta-cli sandbox run-bg <sandbox-id> \
  --command "bash <remote-repo>/scripts/delta_small_molecule_campaign.sh preflight" \
  --timeout 7200 \
  --wait
```

The accepted preflight result was:

```json
{
  "status": "ok",
  "mode": "preflight",
  "model": "Qwen3.5-9B",
  "evaluations": 1,
  "smiles": "CC(C)Sc1ccc(CN2C(=O)c3ccccc3C2=O)cc1",
  "vina": -8.8,
  "predicted_activity": 5.047809978149913,
  "final_hypervolume": 0.42072780771923496
}
```

The full launcher refused to run unless this marker existed and contained one
finite Vina/activity pair.

### 7. Launch the real campaign only after preflight success

The full run used CPU for the Gaussian process so Qwen retained sole use of
the GPU:

```bash
delta-cli sandbox run-bg <sandbox-id> \
  --command "bash <remote-repo>/scripts/delta_small_molecule_cpu_full.sh" \
  --timeout 21600 \
  --wait
```

The effective search arguments included:

```text
seed=42
budget=100
init_size=5
batch_size=1
m1_k_direct_llm=128
max_candidates_per_round=32
acquisition=ehvi
ehvi_n_samples=128
llm_max_tokens=512
gp_device=cpu
allow_early_stop=false
```

The receptor cache produced by the preflight was reused by the campaign.

### 8. Keep long remote executions observable

Receptor preparation and docking can be quiet for minutes. The launcher
printed a heartbeat every five seconds so the provider and Delta SSE path did
not interpret silence as an abandoned command:

```text
[heartbeat] campaign still running at 2026-08-14T13:50:04Z
```

Useful progress lines were:

```text
round=N reservoir: candidates=... drops=... elapsed=...
round=N selection: ... scores=[[vina, activity]] ...
round=N done history=M/100
```

Monitoring focused on committed `history=M/100` lines. A completed proposal
reservoir was not treated as a completed evaluation until the selection and
round-record lines appeared.

### 9. Recover incremental state rather than assuming final files exist

This search implementation writes `rounds.jsonl` incrementally. It writes
`history.json` and `summary.json` only after normal campaign completion.

When the user requested a stop, the campaign process group was terminated and
the remote run directory contained:

```text
config.json
rounds.jsonl
```

The final committed record was round index 29, giving 30 completed
evaluations. The raw checkpoint was pulled with integrity verification:

```bash
delta-cli sandbox pull <sandbox-id> \
  --source <remote-run-dir>/ \
  --target data/generated/delta_small_molecule_campaign_30/ \
  --recursive
```

`history.json`, `progress.csv`, `summary.json`, and `result.json` were then
reconstructed from each record's `selection_results.selected_smiles` and
`selection_results.selected_scores`. Every recovered objective pair was
checked for finiteness.

### 10. Plot with the repository-native plotter

The repository plotter accepts `rounds.jsonl` directly, so it was run inside
the sandbox's installed small-molecule environment:

```bash
python tasks/small_molecule/scripts/plot_pareto_hv.py \
  <remote-run-dir> \
  --budget 30 \
  --output-dir <remote-plot-dir> \
  --prefix progress \
  --title "Qwen3.5-9B KRAS G12D Campaign Progress (30/100)"
```

The generated PNG, PDF, and CSV files were pulled through Delta CLI. Local
validation confirmed:

- 30 history rows.
- 30 compact progress rows.
- 30 finite objective pairs.
- 31 hypervolume rows, including evaluation zero.
- Final hypervolume `15.310060291545108`.
- Nonempty PNG and PDF files.

### 11. Finalize the sandbox explicitly

The active search process group and vLLM server were stopped before artifact
retrieval. Immediate sandbox destruction then exposed a provider-side cleanup
failure:

```text
kill remote sandbox failed; database record retained for reconcile retry
```

Two immediate-kill requests either timed out or failed at the provider. The
successful cleanup path was:

```bash
delta-cli sandbox finish <sandbox-id> \
  --results '{
    "status":"partial",
    "evaluations":30,
    "completion_reason":"stopped_at_user_request",
    "artifacts_downloaded":true
  }'
```

Delta returned `status: finished`. A task is not complete while its GPU
sandbox is still running, even if all desired files have already been pulled.

## Failures and Root Causes

### Context limit at 4,096 tokens

The first campaign configuration used a 4,096-token context. It failed at
history 7 with the exact boundary condition:

```text
3585 input tokens + 512 output tokens >= 4097
```

This was not a malformed prompt or service outage. The requested prompt plus
completion exceeded the configured vLLM maximum sequence length.

### Context limit at 8,192 tokens

After raising the context to 8,192, an earlier campaign attempt reached 55
finite evaluations and then failed at the next exact boundary:

```text
7681 input tokens + 512 output tokens >= 8193
```

The durable fix was `--max-model-len 16384`. The earlier 55-row run could not
be recovered after its sandbox exceeded provider lifetime, so it is not mixed
with or reported as part of the final 30-row artifact set.

### HAMI allocation failure despite free aggregate memory

With `--gpu-memory-utilization 0.80`, Qwen weights loaded, but vLLM attempted
to create a very large KV cache. HAMI rejected a 5.01 GiB allocation while
48.49 GiB was reported free:

```text
cuMemoryAllocate failed res=2
torch.OutOfMemoryError: Tried to allocate 5.01 GiB
```

Reducing utilization to `0.50` still produced 2.08 GiB cache tensors and HAMI
rejected one of those allocations while 46.04 GiB was free.

The successful setting was `0.38`. It retained enough KV capacity for the
single-request 16,384-token campaign while keeping individual allocations
below the observed HAMI failure size.

The important distinction is aggregate memory versus allocator-compatible
contiguous allocations. `nvidia-smi` showing free memory did not prove that
vLLM's proposed KV tensor layout would be accepted.

### RDKit crashes on individual Qwen SMILES

Some generated SMILES caused the RDKit canonicalization process to exit with
return code `-11`. Running canonicalization in the main campaign process would
have made one model-generated candidate capable of terminating the entire
run.

The implemented defense was:

- Canonicalize candidates in isolated subprocesses.
- Cache successful and failed results.
- Recursively split failed batches.
- Drop only a singleton candidate that still crashes.
- Continue reservoir construction with valid candidates.

Representative campaign output was:

```text
RDKit canonicalization worker rejected candidate returncode=-11 smiles='...'
```

These warnings demonstrated containment, not campaign failure.

### Duplicate-heavy proposal reservoirs

Qwen increasingly exploited the current best scaffold and often returned
close analogs, previously evaluated molecules, or invalid informal SMILES such
as `Me`, `CF3`, and other non-SMILES abbreviations.

The reservoir code handled this by:

- Canonical deduplication.
- Excluding prior evaluations.
- Filtering invalid candidates.
- Issuing refill proposal batches when fewer than 32 candidates survived.

This preserved correctness but became the dominant runtime cost in several
rounds.

### Long structured-output retry tails

The full campaign requested 16 concurrent chunks and allowed 10 retries per
chunk with a 10-second retry wait. Later rounds sometimes required 30 to 39
total attempts to obtain 16 usable JSON responses. One proposal phase took
more than 200 seconds.

The campaign remained live and checkpointed, but latency variance increased
as history and molecular complexity grew. The five-second heartbeat was
important for distinguishing a long retry tail from a dead process.

### Large upload proxy failure

The initial 228 MB upload returned HTML where Delta CLI expected JSON. Splitting
the payload into repository components made failures local, retries cheap, and
integrity results readable.

### Sandbox destruction failure

Provider-side immediate deletion failed even after the process group was
stopped. Retrying `kill` was insufficient. `sandbox finish --results ...`
successfully moved the resource to `finished` state.

## Results at the Stop Point

| Metric | Value |
|---|---:|
| Completed evaluations | 30/100 |
| Finite objective pairs | 30/30 |
| Final cumulative hypervolume | 15.310060291545108 |
| Best Vina score | -10.6 |
| Best predicted activity | 6.478514430123958 |

Best Vina molecule:

```text
CS(=O)(=O)N1CCC(C(=O)Nc2cccc(C(F)(F)c3cccc(F)c3)c2)CC1
```

Best predicted-activity molecule:

```text
CS(=O)(=O)N1CCC(C(=O)Nc2cccc(C(F)(F)C#N)c2)CC1
```

The 30th evaluation improved docking to `-10.6` with predicted activity
`6.045722951488777`. The cumulative hypervolume increased from
`14.833839270804367` at evaluation 29 to `15.310060291545108` at evaluation
30.

## Recommendations for the Next Run

### Preserve the proven infrastructure settings

Use these values unless a dedicated preflight demonstrates a better setting:

```text
max_model_len=16384
gpu_memory_utilization=0.38
gp_device=cpu
```

Do not return to `0.80` merely because aggregate GPU memory appears free.

### Increase full-run completion allowance carefully

The full run used `llm_max_tokens=512`, while the successful preflight used
1,024. The later campaign's high structured-output retry count suggests that
some multi-molecule JSON responses may have been truncated or malformed at
512 tokens.

A focused preflight should compare 512 and 1,024 output tokens using the same
16-chunk request pattern. If 1,024 materially reduces retries without causing
context pressure, use it for the resumed campaign. The 16,384-token context
provides room, but the combined prompt plus completion must still be measured
near the end of the 100-evaluation history.

### Consider lower proposal concurrency

The search launched 16 Qwen chunks concurrently. Testing 8 to 12 workers may
reduce request-tail latency and allocator pressure, even if peak throughput is
lower. Compare total time to obtain 32 valid unique candidates, not only raw
tokens per second.

### Pull checkpoints periodically

Because `rounds.jsonl` is the durable incremental artifact, pull it after
fixed milestones such as every 10 evaluations. This prevents a provider
lifetime event from erasing hours of valid scientific work.

Also consider changing the trace recorder to write `history.json` and a
partial `summary.json` after every committed round. That would simplify
external monitoring and artifact recovery.

### Resume from the recovered trajectory

The launcher supports:

```bash
CAMPAIGN_RESUME=1 bash scripts/delta_small_molecule_cpu_full.sh
```

For a new sandbox, upload the recovered `rounds.jsonl`, `history.json`, and
`config.json` to the exact trajectory directory before launching with resume
enabled. Verify that startup reports:

```text
history=30/100
```

The resumed search will preserve evaluated molecules and round numbering. It
will not necessarily be bit-for-bit identical to an uninterrupted run because
process-local random-number-generator state is reinitialized; scientific
continuation and deterministic replay are different requirements.

### Treat preflight as a release gate

The full run should start only when all of the following are true:

- Uploaded activity-model checksum matches.
- Every Qwen checkpoint shard exists and no `.partial` file exists.
- Vina is a compatible Linux executable and responds to `--help` or
  `--version`.
- Qwen returns the exact health response.
- Repository dependency checks pass.
- Contract dry run passes.
- One real molecule receives two finite objective values.
- Preflight hypervolume is finite.

## Operational Checklist

### Before allocation

- Confirm the requested image, workspace, checkpoint, GPU, and maximum life.
- Estimate upload size and split large trees before transfer.
- Record the expected activity-model checksum.

### Before preflight

- Verify remote model shards and absence of partial files.
- Verify uploaded model checksum inside the sandbox.
- Verify the Vina binary format and version.
- Use a conservative KV-cache allocation under HAMI.
- Set a context length that includes both prompt and maximum completion.

### Before the full campaign

- Require a valid real preflight marker.
- Reuse the preflight receptor cache.
- Keep the GP on CPU while vLLM owns the GPU.
- Confirm the intended seed, budget, batch size, and objective directions.
- Confirm incremental trajectory paths.

### During the campaign

- Monitor committed history count, not proposal count.
- Treat isolated RDKit `-11` warnings as candidate drops unless the parent
  process exits.
- Watch LLM attempt counts and refill frequency.
- Pull `rounds.jsonl` at milestones.
- Track sandbox age against provider maximum life.

### At completion or user-requested stop

- Stop the campaign process group, including vLLM and proposal subprocesses.
- Verify the last complete `rounds.jsonl` record.
- Pull raw artifacts with integrity checking.
- Reconstruct partial history only from committed selection results.
- Validate evaluation counts and score finiteness.
- Generate and inspect plots.
- Finalize or destroy the sandbox and verify terminal state.

## Closing Assessment

Delta CLI provided a workable boundary for a real GPU-backed molecular
campaign, including large-file transfer, private model serving, long-running
execution, checkpoint retrieval, and resource cleanup. The main difficulties
were not the high-level Delta commands. They were the interactions between
provider lifetime, proxy limits, binary compatibility, HAMI allocation
behavior, growing model context, model-generated chemistry, and incremental
checkpoint semantics.

The most important workflow lesson is to make every expensive stage prove the
next stage is safe: verify bytes before loading models, verify the model server
before proposing chemistry, perform one real dual-objective evaluation before
launching the campaign, checkpoint every committed round, and recover only
what the trace proves was completed.
