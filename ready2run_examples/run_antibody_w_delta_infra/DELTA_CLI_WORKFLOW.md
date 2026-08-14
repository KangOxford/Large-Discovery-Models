# Delta CLI Antibody Preflight and Campaign Runbook

## Purpose

This document summarizes the practical workflow, results, failures, recovery
steps, and engineering lessons from running real antibody CDRH3 inference and
evaluation through Delta CLI.

The work covered two successful stages:

1. A two-candidate preflight that verified real Qwen inference and real
   AntBO/Absolut scoring for antigen `1ADQ_A`.
2. A 20-evaluation campaign implemented as four history-aware Qwen proposal
   rounds of five candidates, with every candidate scored by the live
   Delta Science AntBO service.

The main campaign artifacts are:

- [Campaign result](./result.json)
- [Evaluation trajectory](./progress.csv)
- [Progress plot](./progress.png)
- [Preflight result](../delta_antibody_preflight/result.json)

## Evidence Boundary

The run used Delta CLI for all sandbox and online scientific operations:

- GPU inference operations used `delta-cli sandbox ...`.
- Antibody service discovery and real evaluations used
  `delta-cli science invoke --tool antbo ...`.
- No direct HTTP requests were sent to the Delta infrastructure or AntBO
  service.
- No local or random score was substituted for an AntBO result.
- Every reported binding energy came from a successful `antbo/evaluate`
  response with top-level `ok: true`.

The successful 20-evaluation campaign was a custom history-aware proposal
loop. It was not the repository's full Gaussian-process/acquisition AntBO
implementation. Qwen proposed sequences from accumulated history, local code
enforced the repository's sequence and developability constraints, and the
real AntBO service evaluated the admitted candidates with Absolut.

## System Architecture

```text
Host workspace
  |
  | delta-cli sandbox create/write/run-bg/cancel/kill
  v
Delta GPU sandbox: vllm_0.27.1:juicefs
  |
  | Qwen3.5-9B OpenAI-compatible endpoint on sandbox localhost
  v
History-aware CDRH3 proposal client
  |
  | validated 11-residue candidate batches returned through Delta CLI
  v
Host campaign controller
  |
  | delta-cli science invoke --tool antbo --endpoint evaluate
  v
Delta Science AntBO service
  |
  v
Real Absolut binding-energy evaluator
```

The model endpoint remained private to the sandbox. The host did not call the
vLLM HTTP endpoint directly. Instead, proposal scripts ran inside the sandbox
through `delta-cli sandbox run-bg` and emitted structured JSON through the
Delta execution stream.

## Shared Configuration

| Setting | Value |
|---|---|
| Antigen | `1ADQ_A` |
| Objective | Absolut binding energy |
| Direction | Minimize; lower is better |
| CDRH3 length | 11 |
| Canonical alphabet | `ACDEFGHIKLMNPQRSTVWY` |
| Requested model family | Qwen 9B |
| Available shared checkpoint | `Qwen3.5-9B` |
| Sandbox image | `image.yangtzeailab.com/opensandbox/vllm_0.27.1:juicefs` |
| Sandbox resources | 16 CPU, 64 GiB RAM, 1 GPU, 80 GB GPU memory |
| Successful campaign budget | 20 real evaluations |
| Campaign batching | 4 rounds x 5 candidates |

The shared workspace contained `Qwen3.5-9B`, not a checkpoint named exactly
`Qwen3-9B`. Results therefore record the exact checkpoint name that actually
ran.

## Standard Delta CLI Lifecycle

### 1. Verify configuration and authentication

Always confirm configuration and authentication before allocating resources:

```bash
delta-cli config show
delta-cli auth status
```

The output must be inspected as a JSON envelope. A usable response has
top-level `ok: true` and a configured authentication method. Tokens or API
keys must never be printed in full or embedded in scripts.

### 2. Verify the scientific backend

Before spending GPU time, verify the current AntBO contract:

```bash
delta-cli science invoke --tool antbo --endpoint ldm-health
```

For this run, the service confirmed:

- `1ADQ_A` was supported.
- CDRH3 length was 11.
- Lower Absolut binding energy was better.
- The real Absolut executable existed at the service side.
- The canonical amino-acid alphabet matched the repository constraint.

This check was important because it established that subsequent scientific
scores would come from a real evaluator rather than a mock.

### 3. Create exactly one task-owned sandbox

The campaign requested one sandbox with the known image and resources:

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

The returned `data.sandbox_id` is authoritative. It must be recorded and used
for all later operations.

One create request returned HTTP 504 even though the provider had created the
sandbox. The safe recovery was a read-only list operation:

```bash
delta-cli sandbox list --status running --days 1 --provider opensandbox
```

The newly created sandbox was identified by its creation time, image, and
exact resource configuration. A second create was not submitted after the
side effect was discovered.

### 4. Resolve the authoritative working directory

```bash
delta-cli sandbox working-directory <sandbox-id>
```

Use the returned path for explicit remote file destinations. During this run,
relative targets supplied to `write-multiple` did not appear under the assumed
working directory even though the operation returned success. Individual
`sandbox write` calls with explicit absolute `--path` values resolved the
ambiguity.

Example:

```bash
delta-cli sandbox write <sandbox-id> \
  --source campaign-propose.py \
  --path <working-directory>/campaign-propose.py
```

### 5. Start vLLM as a long-running background execution

Long jobs must use `run-bg`. The model server was started once and reused by
all proposal rounds:

```bash
delta-cli sandbox run-bg <sandbox-id> \
  --command "bash <working-directory>/campaign-server.sh" \
  --timeout 7200
```

The server execution ID was retained for later cancellation. The startup
script performed these checks and compatibility adjustments:

1. Parsed `model.safetensors.index.json`.
2. Verified that every referenced shard existed.
3. Rejected the checkpoint if any `.partial` file remained.
4. Set the Conda environment's library directory in `LD_LIBRARY_PATH`.
5. Applied an idempotent FlashInfer deferred-annotation compatibility patch.
6. Set `VLLM_ENABLE_V1_MULTIPROCESSING=0`.
7. Launched Qwen with `--language-model-only`.
8. Used `--max-model-len 4096`, `--gpu-memory-utilization 0.80`, and
   `--enforce-eager`.

The `--language-model-only` flag was necessary because this vLLM release
resolved Qwen3.5 as a multimodal architecture and otherwise attempted to load
an image processor.

### 6. Run proposal jobs through the sandbox

Each round uploaded a small `campaign_input.json` containing:

- Round index.
- Requested proposal count.
- All previously evaluated sequences and real scores.

The proposal client ran through Delta CLI and waited for the private model
endpoint:

```bash
delta-cli sandbox run-bg <sandbox-id> \
  --command "python <working-directory>/campaign-propose.py" \
  --timeout 1200 \
  --wait
```

Successful proposal jobs printed one independent JSON object at the end of
stdout. Delta CLI carried that object in the complete frame as
`result_summary`.

### 7. Validate candidates before consuming evaluator budget

Every sequence had to satisfy all of the following:

- Exactly 11 residues.
- Only canonical amino-acid characters.
- At most one cysteine.
- No hydrophobic run longer than four residues.
- At most two aromatic residues from `F`, `W`, and `Y`.
- Approximate net charge between -1 and 2.
- No `N-X-S` or `N-X-T` motif where `X` is not proline.
- Not previously evaluated.
- Not duplicated within the current proposal batch.

No random candidate fallback was used. Failed or duplicate Qwen outputs were
rejected before calling AntBO, so they did not consume the 20-evaluation
scientific budget.

The first multi-candidate prompt caused Qwen to repeat the same sequence. The
effective recovery was to request one candidate per completion, immediately
add accepted and rejected sequences to `do_not_repeat`, return explicit
rejection reasons to the next attempt, and vary seed and temperature across
model retries.

### 8. Evaluate each admitted batch exactly once

The host submitted each five-sequence batch to the live service:

```bash
delta-cli science invoke \
  --tool antbo \
  --endpoint evaluate \
  --data '{"designs":["SEQUENCE1","SEQUENCE2"],"antigen":"1ADQ_A"}'
```

`evaluate` is a heavy operation. If the response had been unknown due to a
timeout or disconnect, it would not have been retried automatically. A batch
was accepted only when all of these conditions held:

- CLI process completed successfully.
- Top-level envelope contained `ok: true`.
- Business response contained `ok: true`.
- Every requested sequence had an item with `ok: true`.
- Score count matched design count.
- Every score was finite.

### 9. Persist trajectory and plot progress

After the fourth batch, the ordered evaluations were written to
`progress.csv`. The plot shows:

- Individual evaluated binding energies.
- The running best value.
- Boundaries between the four proposal rounds.
- The best sequence and its evaluation index.

Matplotlib was unavailable on the host but already present in the vLLM image.
The plotting script and CSV were sent into the sandbox, the PNG was rendered
there, and the output was retrieved with:

```bash
delta-cli sandbox pull <sandbox-id> \
  --source <working-directory>/progress.png \
  --target data/generated/delta_antibody_campaign_20/progress.png
```

The PNG was validated by magic bytes, dimensions, and visual inspection. The
final image is 1904 x 1104 pixels.

### 10. Stop executions and destroy the sandbox

The model server was cancelled using its execution ID. The sandbox was then
destroyed explicitly:

```bash
delta-cli sandbox cancel <sandbox-id> --execution-id <server-execution-id>
delta-cli sandbox kill <sandbox-id>
```

The task was not considered complete until `sandbox kill` returned
`status: killed`. A sandbox must be destroyed even when setup, inference, or
evaluation fails.

## Preflight Experience

### Original repository preflight attempt

The first approach tried to execute the repository's locked real CPU smoke
configuration end to end. It included:

- A task-local CPython 3.9.23 interpreter.
- `uv sync --locked` for the Antibody project.
- The repository dependency check and dry-run contract.
- One real model proposal and one real Absolut evaluation.

The dependency resolver successfully resolved 71 packages, but the pinned
Torch 1.9 package was approximately 792.9 MiB and unpacked very slowly. The
command reached its 3600-second limit while still materializing Torch and was
killed before inference. Earlier `uv` attempts also exposed a managed-Python
download/promotion deadlock, which was avoided by providing the task-local
Python 3.9 interpreter and disabling managed-Python downloads.

This attempt established an important boundary: a correct locked environment
can still be unsuitable for a short preflight when one legacy dependency
dominates the execution window.

### Successful real preflight

The recovery path separated inference from scientific scoring:

1. Load the shared Qwen3.5-9B checkpoint with the image's existing runtime.
2. Generate constrained CDRH3 proposals inside the GPU sandbox.
3. Verify AntBO service authenticity through `ldm-health`.
4. Submit the exact Qwen proposals to `antbo/evaluate` once.
5. Store the concise result and destroy the sandbox.

The successful preflight result was:

| Sequence | Real Absolut binding energy |
|---|---:|
| `MSTVQTEVTLK` | -68.18 |
| `MSTVQTEKSLI` | -74.96 |

The evaluator processed both candidates in 14.705227 seconds. The best
preflight sequence was `MSTVQTEKSLI` at -74.96.

## 20-Evaluation Campaign Workflow

The successful campaign used four sequential rounds. Each round followed the
same state transition:

```text
validated history
  -> Qwen proposes five new CDRH3 sequences
  -> local admission checks reject invalid or duplicate outputs
  -> one real AntBO/Absolut batch evaluation
  -> append exact sequence/score pairs to history
  -> compute running minimum
  -> next round
```

### Round summary

| Round | Evaluations | AntBO request ID | Evaluator time (s) | Round best | Global best after round |
|---:|---:|---|---:|---:|---:|
| 1 | 1-5 | `req_b0b11c398c35421fbecdc3d7f26583a6` | 19.370335 | -82.24 | -82.24 |
| 2 | 6-10 | `req_a50be11d41e04c7aa22b53db341e5926` | 19.128739 | -86.85 | -86.85 |
| 3 | 11-15 | `req_77df4e03ec6b4191a713b9af1ca7afb8` | 22.115259 | -81.06 | -86.85 |
| 4 | 16-20 | `req_62166c4bf0c14070905db4be98afa1ce` | 18.378871 | -87.72 | -87.72 |

Total AntBO evaluation time was 78.993204 seconds for 20 candidates. This is
the sum of the four service-reported batch timings and excludes model startup,
proposal generation, file transfer, and troubleshooting time.

### Optimization result

| Metric | Value |
|---|---|
| Initial five-evaluation best | -82.24 |
| Final campaign best | -87.72 |
| Binding-energy improvement | 5.48 lower |
| Best sequence | `MKSTLEAVLGM` |
| Best evaluation | 18 |
| Improvement evaluations | 1, 2, 4, 7, 18 |

Round 3 did not improve the global best. Round 4 recovered and found the final
best at evaluation 18. This behavior is visible in the best-so-far step line
in [progress.png](./progress.png).

## Issues Encountered and Recoveries

| Issue | Observable symptom | Recovery |
|---|---|---|
| Exact requested model name unavailable | Shared workspace contained `Qwen3.5-9B`, not `Qwen3-9B` | Used and reported the available 9B checkpoint exactly |
| Locked project required Python 3.9 | Image runtime was Python 3.11 | Bootstrapped task-local CPython 3.9.23 |
| `uv` managed-Python promotion stalled | Repeated download/promotion deadlock | Passed the task-local interpreter and disabled managed downloads |
| Legacy Torch dependency was too slow | 792.9 MiB Torch unpack exceeded 3600 seconds | Avoided the full locked environment for the real preflight and campaign |
| Original sandbox expired | Task-local files and local Absolut tree disappeared | Confirmed `not_found`, then created one replacement sandbox |
| Create returned HTTP 504 with side effect | CLI reported failure, but a matching new sandbox existed | Used `sandbox list`; did not duplicate the create |
| Checkpoint was temporarily incomplete | Missing final shard names and visible `.partial` files | Waited for atomic shard promotion and validated the index before launch |
| vLLM attempted multimodal setup | Missing image processor error | Added `--language-model-only` |
| C++ ABI mismatch | `CXXABI_1.3.15 not found` when Conda libraries were absent | Exported the Conda library path before starting vLLM |
| FlashInfer annotation incompatibility | Import-time failure in `fd_exchange.py` | Applied an idempotent future-annotations patch |
| Local Absolut layout differed | Per-thread shard files appeared instead of the expected merged file | Used the supported Delta Science AntBO evaluator for final scientific results |
| Relative batch-write paths were ambiguous | Write operation succeeded but scripts were not at the assumed path | Queried the working directory and used explicit absolute `--path` values |
| Background log endpoint was inconsistent | A server execution could not be inspected through `sandbox logs` | Redirected server output to a file in the sandbox working directory |
| Qwen repeated candidates | Five-item completion duplicated one sequence | Switched to independent one-item completions with dynamic exclusions |
| Qwen returned invalid candidates | Charge, hydrophobic, length, or duplicate checks failed | Added rejected sequences and explicit reasons to subsequent prompts |
| Execution wrapper sometimes reported exit code 0 with stderr traceback | Transport completion looked successful despite application failure | Required structured stdout result and validated business content, not exit code alone |
| Sandbox kill was slow | Teardown request remained silent for several minutes | Kept one kill request attached until Delta returned `status: killed` |

## Practical Lessons

### Treat Delta JSON envelopes as the source of control flow

Do not infer success only from shell exit status. Inspect top-level `ok`, the
complete frame, stderr, and the operation-specific business payload. For
scientific evaluation, also verify item counts and finite scores.

### Do not retry operations with unknown side effects

A 504 from sandbox creation did not mean that creation failed. Read-only
discovery identified the new sandbox and prevented a duplicate GPU allocation.
The same principle applies even more strongly to heavy scientific operations:
an unknown evaluation result must not be resubmitted automatically.

### Separate environment validation from scientific validation

The repository environment preflight and the real scientific preflight answer
different questions:

- Environment preflight: can the exact locked project install and execute?
- Scientific preflight: can the chosen model produce admissible candidates and
  can the real evaluator score them?

The locked environment failed within the time window, while the real model and
evaluator path succeeded. Reporting those results separately avoids claiming
that the full repository stack was validated when it was not.

### Preserve evaluator budget during proposal recovery

Proposal retries are cheaper than consuming real evaluations with invalid or
duplicate sequences. The campaign kept the evaluation budget exact by
requiring five admitted candidates before each AntBO call.

### Use persistent model serving for multi-round campaigns

Loading Qwen once and reusing the endpoint across four rounds avoided repeated
checkpoint startup. The server execution ID was treated as campaign state and
cancelled explicitly before sandbox destruction.

### Make model outputs machine-verifiable

The proposal script printed one final JSON object containing round number,
history size, model name, count, and proposals. Structured output made it
possible to distinguish a valid completion from a traceback or transport-only
success frame.

### Keep raw trajectory separate from concise summary

`progress.csv` contains every scientific observation, while `result.json`
contains the concise campaign conclusion and execution evidence. The PNG is a
derived visualization of the CSV, not an independent source of scientific
values.

### Cleanup is part of success

A GPU campaign is incomplete until its background executions are stopped and
the sandbox is destroyed. Cleanup must also run after failures, timeouts, or
partial setup.

## Recommended Improvements for Future Runs

1. Build a campaign image that already contains the exact Python 3.9 and
   pinned Torch environment if repository-native AntBO execution is required.
2. Pre-stage and checksum all Qwen shards before allocating the GPU sandbox.
3. Add a supported readiness operation or persistent background-log lookup to
   reduce manual process diagnostics.
4. Store round state transactionally after every successful evaluation so a
   host interruption can resume without repeating scientific calls.
5. Replace literal output examples in prompts with a JSON schema or guided
   decoding to reduce example copying.
6. Extend the campaign controller with a real surrogate/acquisition model when
   the goal is to reproduce full AntBO Bayesian optimization rather than the
   history-aware Qwen loop used here.
7. Add automated checks for duplicate sequences, finite scores, monotonic
   best-so-far values, CSV row count, PNG signature, and artifact paths before
   cleanup.

## Reproducibility Checklist

- [ ] `delta-cli config show` returns `ok: true`.
- [ ] `delta-cli auth status` reports a configured method.
- [ ] `antbo/ldm-health` supports the requested antigen and confirms Absolut.
- [ ] Exactly one task-owned sandbox is created.
- [ ] The authoritative working directory is recorded.
- [ ] Every model shard referenced by the index exists.
- [ ] No checkpoint `.partial` file remains.
- [ ] vLLM starts with the required compatibility settings.
- [ ] Proposal jobs emit structured JSON.
- [ ] Every candidate passes length, alphabet, developability, and uniqueness checks.
- [ ] Every AntBO request is submitted once.
- [ ] Score count equals design count for every successful batch.
- [ ] The final trajectory contains exactly the requested evaluation budget.
- [ ] Best-so-far values are monotonically non-increasing.
- [ ] `result.json`, `progress.csv`, and `progress.png` are validated.
- [ ] The model server is cancelled.
- [ ] The sandbox is explicitly killed.

## Final Result

The real 20-evaluation campaign completed successfully for `1ADQ_A` using
Qwen3.5-9B proposals and Delta Science AntBO/Absolut scoring. It improved the
best observed binding energy from -82.24 after the first round to -87.72, with
`MKSTLEAVLGM` selected as the best sequence at evaluation 18.
