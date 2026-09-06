# LDM RL — master experiment plan

Single entry point for this line of work. Links to the detailed documents rather than
copying them. Updated every round; **updating this file is not progress — advancing the
work is.**

| | |
|---|---|
| scope | SID `3edc5462-19e3-492e-9aeb-534e7b0e31c5` / Large Discovery Model |
| last updated | 2026-09-05 22:20 UTC (Sydney 2026-09-06 08:20 +1000) |
| detail documents | [`TRAINING_PLAN.md`](TRAINING_PLAN.md) · [`GATE_9B_finite_gradient.md`](GATE_9B_finite_gradient.md) · [`MINIMAL_RL_GAIN_TEST.md`](MINIMAL_RL_GAIN_TEST.md) · [`MAIN_OBJECTIVE_AUDIT.md`](MAIN_OBJECTIVE_AUDIT.md) · [`PRESPEC_lr0_control.md`](PRESPEC_lr0_control.md) |
| machine-readable | [`main_objective_manifest.json`](main_objective_manifest.json) |

## 1. Scientific goal

**Does RL on top of SFT beat SFT-only?**

| | |
|---|---|
| primary metric | **Pareto front hypervolume at search budget 80** |
| evaluation | **G12D** in-distribution, **G12C** transfer (activity model differs, docking receptor 8UN5 shared; G12C is offline-only and never enters the RL loop) |
| baselines | **SFT-only** (no RL) and **base** (no SFT, no RL), both under the identical budget-80 loop |
| arms | R1 base+acq-max · R2 SFT+acq-max (pivot) · R3 SFT+ΔHV · R4 SFT+acq-mean |
| **not the metric** | Spearman(vina, proposal index), `zero_std` rate. Diagnostics from a side investigation; neither substitutes for hypervolume. |

## 2. Step graph

| id | step | depends on | verifiable completion condition | state |
|---|---|---|---|---|
| **S0** | CPU evaluation chain runnable | — | 11/11 packages import; both QSAR models load and predict from raw SMILES; botorch `Hypervolume` computes | **done** |
| **S1** | docking chain importable, Vina binary resolves | S0 | all 6 `tasks.small_molecule` eval modules import, both QSAR models predict, `Hypervolume` computes, `VINA_BIN` runs | **done** — in the `ldm-rl` env, verified 2026-09-05 20:57Z |
| **S1c** | 8UN5 prepared and a ligand docked end to end | S1 | a Meeko receptor PDBQT exists and Vina returns a real score | **done** — aspirin **−6.804**, benzamide **−5.925** kcal/mol, 3.8 s for two ligands |
| **S2** | 9B finite-gradient / real-update gate | S1b | ≥3 consecutive steps with `found_inf_flag == False` **and** `update_successful == True`, plus before/after weight fingerprints showing the parameters moved | **blocked** — attempt 1 failed on a filesystem fault, see §3 |
| **S1b** | Triton kernel cache off the quota-full filesystem | — | the failure reproduces on the quota-full path and does not on the node-local one | **done, verified on CPU with zero GPU** — see §11 |
| **S3** | one 9B RL run that trains | S2 | optimizer applies finite-gradient updates over a meaningful stretch, checkpoint at a pre-declared step | **blocked on S2** |
| **S4** | SFT-only and base reference rows at budget 80 | S1c + inference GPU grant | HV on G12D and G12C for both baselines, no RL involved | **blocked on a GPU grant only** — see §12 |
| **S5** | RL vs frozen-policy comparison | S3, S4 | per `MINIMAL_RL_GAIN_TEST.md`, after §8's gaps are closed | **blocked on S3** |
| **S6** | main table: HV@80 × {G12D, G12C} × {R1..R4, SFT-only, base} | S5 | every cell filled or explicitly marked not-done with its reason | **not started** |

**S4 does not depend on S2 or S3, but it is not CPU-only.** The baselines need no RL and no
9B *training* — they still need **inference GPUs** to generate proposals, plus CPU docking.
"No RL training" must not be read as "no GPU". Its minimal protocol and budget are in §9.

## 3. Current state

| | |
|---|---|
| **running** | nothing |
| **ready** | **S4 pilot — prepared, reviewed, deployed. Blocked on cluster capacity, not on anything of ours.** Coordinator's cluster-wide sample at **2026-09-05 22:11:03 UTC (Sydney 2026-09-06 08:11:03 +1000)**: 4 allocations, 16 nodes, **64/64 GPUs carry a process, 0 free**. That is an occupancy count, not a statement that all of it is useful work. Earlier idle candidates are void; no old node is chased and no spent lease is replayed |
| **not ready** | **S2 attempt 2** — needs S1b verified under a real kernel compile, a current device grant, and budget confirmation. Not ready, not scheduled, not auto-restarted. Any retry suggestion appearing elsewhere is not an authorisation |
| **blocked** | S2 on **inodes for checkpoint writes** · S3 on S2 · S5 on S3 · S6 on S5 |
| **done** | **S0, S1, S1b, S1c** |

### S2 attempt 1 — failed on a filesystem fault, not on the model

Step `6324130.1593` on nid010851, **CANCELLED by own uid**, exit `0:9`, elapsed **1086 s**
(≈ **1.21 of the 2 authorised GPU-hours**). Ended through its own `srun` clients after the
scheduler died; the shared allocation was untouched and is still RUNNING.

Root cause, from the log:

```
OSError: [Errno 122] Disk quota exceeded:
  '/projects/public/u6gb/.triton/cache/LWUQPDGCY3NZF3Y7C6PHL25N5ULLBD4BKBHPMMPCUXB5RI7YK65Q'
```

Triton compiles the GDN kernels at runtime and writes them to `$HOME/.triton`, which is on
Lustre, and the project inode quota is full at 51,200,000/51,200,000. The sglang Scheduler
raised, then the rollout side spun through **482 HTTP 500 retries** and degraded to
`All connection attempts failed`.

**Correction to an earlier claim of mine: forwards did run.** The log holds **9 successful
`POST /generate` 200 responses** and decode batches growing to 1042 tokens before the first
error at 20:40:16Z — those kernels were already cached. The failure hit when the GDN
**extend/prefill** path needed to compile a kernel that was not. The accurate statement is
that **the run never reached the training-update gate**, not that no forward executed.

**Gate evidence obtained: none.** `grad_norm` records 0, `found_inf` 0, skip counts 0,
`update_successful` unobserved, weight fingerprints not taken. A failed launch is not a
completed step — S2 remains blocked.

**Fix applied (S1b), CPU only, not yet exercised:** `run_single_9b.sh:118` now sets
`TRITON_CACHE_DIR` to node-local `/tmp/triton_${USER}_${SLURM_JOB_ID}`, which is outside
the quota (334 G free on nid010851's `/tmp`). Expansion verified to render
`/tmp/triton_kangli.u6gb_6324130` — the first draft wrote `$USER_$SLURM_JOB_ID`, which bash
parses as a variable named `USER_` and would have collapsed the path.

**Not restarted.** Automatic restart is prohibited, and 0.79 GPU-h of the original 2 remain
unspent. Attempt 2 needs an explicit device grant.

### Contention, as sampled at 20:35:39Z (historical — the step has since ended)

At that sample `6324130.1603` and every later `topup-*` / `cell-round1_*` step on nid010851
were FAILED, `.1593` was the only RUNNING compute step there, and NVML listed one compute
process, mine. **That reading is now stale**: `.1593` ended at 20:42Z and the coordinator's
20:48:11Z NVML read shows four cards at 0% and 1,1,1,3 MiB with no compute processes. This
line's current GPU demand is **zero**.

The earlier citation of `.751`/`.752` as COMPLETED could not have settled the question at
the time; the coordinator was right to reject it.

## 4. Provenance

| | |
|---|---|
| code | `_wt_handoff` @ `handoff-pr`; runtime `_wt_runtime` @ `pr/reward-zero-fixes` |
| SFT start point | `LDM-rl/rl/qwen3.5-9B-sft_torch_dist` |
| base start point | `LDM-rl/rl/qwen3.5-9B_torch_dist` |
| activity models | `ldm_rl/results/g12c_qsar_20260901T010923Z`, `ldm_rl/results/g12d_qsar_matched_20260901T011826Z` — load only with that trainer's directory on `sys.path` |
| receptor | 8UN5 is a **PDB id**, not a repo file; fetched and Meeko-prepared at runtime into `<trajectory-dir>/vina_cache` |
| seeds | **training seed and sampling seed are different things.** No historical run records `--seed`, so they cannot be separated retrospectively. Every new run must print both. |
| checkpoints | 100 of 248 `hidden=1536` dirs (1.5B **pilot**); 7 of 90 `hidden=4096` dirs (9B, see §5) |

## 5. What is known about the 9B checkpoints, at the strength the evidence supports

| claim | status |
|---|---|
| 238 of 238 logged `train/grad_norm` values are nan or inf | **measured** |
| ~~therefore no 9B run ever performed a valid update, and there is no model to compare~~ | **withdrawn — overstates the evidence.** `grad_norm` is initialised to `nan` and overwritten only inside the guarded block, so a printed `nan` does not establish the gradients were non-finite |
| `R2-nonanguard` weights changed (648/753 tensors); `R3a` weights did not (0/753) | **measured** |
| whether any step had `found_inf_flag == False` | **unknown** — never logged |
| whether gradients were finite on the steps where weights moved | **unknown.** Weight change alone does not prove it — it is consistent with a finite-gradient update but was never instrumented to exclude other causes |
| skip counts per run | **unknown** — never recorded |

S2 exists to turn those three unknowns into measurements.

## 6. Budget

| item | authorised | actual | note |
|---|---|---|---|
| S2 gate, attempt 1 | **4 GPU × 30 min = 2 GPU-h** | **1.21 GPU-h** (1086 s × 4), failed | covers initialisation, compilation, failures, retries |
| S2 remaining | 0.79 GPU-h of the same authorisation | 0 | attempt 2 not launched; no auto-restart |
| lr=0 control arm (closed) | 27.5 GPU-h | 21.93 GPU-h | **spent on a different question; remainder not transferable** |
| S0, S1 | — | 0 | CPU only |

**Stop rules for S2:** 30 minutes wall, or pass criteria met, or a located failure. The step
then ends through its own `srun` client. **No automatic restart, no scanning for free cards,
no cross-node top-up, no additional runs.** The shared allocation is never cancelled.

**A failed launch is not a completed step.** S2 counts as `done` only against §2's condition.

## 7. Next actions

1. **S2** — let it run. Report `found_inf`, skip counts and weight fingerprints, or "not
   reached". No intervention unless it fails or hits the cap.
2. **S1** — obtain a Vina binary (conda-forge `autodock-vina` ships Boost, or a prebuilt
   binary plus `VINA_BIN`), then prepare 8UN5 into the cache. CPU-only, unblocks S4.
3. **S4** — once S1 lands, the SFT-only and base reference rows can be produced **without
   any 9B training**. Largest piece of the main table not hostage to S2.

**Infrastructure blocker, and it is worse than a documentation nuisance.** Both storage
quotas are full — Lustre project inodes at 51,200,000/51,200,000 and VAST `/home` at
101G/101G. New files cannot be created anywhere but node-local scratch, which is why this
document overwrites tracked content instead of adding a file. **It also killed S2 attempt 1
by blocking Triton's kernel cache**, so the quota is not only stopping writes, it is
stopping compute. `TRITON_CACHE_DIR` routes around it for this launcher; a general fix needs
inodes actually freed, which is a decision for the account owner — `mv` does not release
inodes, only deletion or a different mount does.

## 8. Design gaps in `MINIMAL_RL_GAIN_TEST.md`, to close before S5 runs

Raised by the coordinator, recorded rather than quietly patched:

### 8.1 Checkpoint selection — resolved to one rule

~~"fixed step, decided before any evaluation" **and** "the last step at which the RL arm
logged finite gradients"~~ — these are not the same rule and cannot both hold.

**Frozen rule:** the checkpoint is the one saved at **step 30**, chosen now, before any
evaluation and before any RL run exists. Both arms use step 30. If an arm never reaches
step 30, that arm is **not evaluated** and is reported as not-reached — it is not replaced
by whatever step it did reach, because "the furthest step this run got to" is a function of
the run's own behaviour and would let the arm choose its own checkpoint.

Why 30: the historical 1.5B runs save every 10 steps and commonly reach 29-49, so 30 is
inside the range without being the maximum, and it was picked from the *save cadence*,
never from an evaluation score.

### 8.2 Two sources of variability, never one error bar

| what varies | what it measures | how many |
|---|---|---|
| **evaluation seed** | sampling noise of the budget-80 loop **for one fixed checkpoint** | 5 per arm per eval set |
| **training seed** | how much a differently seeded RL run would differ | **1 — this design has no replicate** |

So the reported interval is a **within-checkpoint** interval. It cannot support any
statement of the form "RL improves HV by X", because a single training run gives no handle
on run-to-run spread. The strongest honest claim available from this design is about **the
specific checkpoints evaluated**, and any generalisation needs training replicates that are
not budgeted here.

## 9. What the remaining 0.79 GPU-h could and could not establish

**Remaining budget is not a reason to retry.** Written out so the decision rests on
expected evidence, not on unspent hours.

S2 attempt 1 spent **1086 s** and reached the first GDN extend compile. Of that, roughly
900 s was initialisation — Ray, sglang engine bring-up, weight load — which any retry pays
again. **0.79 GPU-h = 711 s of 4-GPU wall**, which is *less* than that initialisation cost.

| outcome | reachable in 711 s? |
|---|---|
| S1b verified (a GDN extend kernel compiles under the node-local cache) | **plausibly** — the failure hit at ~1000 s, and this run skips nothing before it |
| ≥3 consecutive steps with `found_inf_flag == False` and `update_successful == True` | **no** — the first training step lies beyond where attempt 1 died |
| weight fingerprints before/after an update | **no** — requires the above |

So the honest framing: the remaining budget buys **S1b's verification**, not S2's gate. If
that is worth 0.79 GPU-h it should be requested as its own bounded check with that as the
stated deliverable, and S2 re-budgeted separately with initialisation counted.

**No guard is deployed.** Nothing currently protects a future step from being pre-empted or
from colliding with other work on a shared node; the earlier framing of a "claim" as
protection was wrong — a claim is a registry entry, not a lock.

## 10. Paths that will still fill the quota

One cache fix is not end-to-end recovery. Every path below writes during a 9B run, and all
but the Triton cache still land on the quota-full filesystem.

| writer | default location | on quota? | needed for the result? | handling |
|---|---|---|---|---|
| Triton kernel cache | `$HOME/.triton` | **was** | no — regenerable | **fixed**: node-local `/tmp`, guarded |
| Ray session / object spill | `/tmp/ray` (already node-local) | no | no | leave |
| sglang engine temp | node-local | no | no | leave |
| `train.log`, `train.progress.log` | `<rundir>` on Lustre | **yes** | **yes — the primary evidence** | must stay on Lustre; append-only to an existing file does not consume an inode |
| `gp_history.jsonl`, `episodes.jsonl` | `<rundir>` on Lustre | **yes** | **yes** | same |
| **checkpoints** `ckpt/iter_*` | `<rundir>/ckpt` on Lustre | **yes** | **only for a re-evaluable model** | see the correction below |
| trajectory / `vina_cache` | `<trajectory-dir>` | was assumed yes | **no** | **S1c proved otherwise**: the whole receptor + docking chain ran under node-local `/tmp`, 2.0 MB total, and produced real scores |

### Correction: I conflated the scientific artifact with one implementation of it

~~S2 needs checkpoint writes, so it cannot run until inodes are freed.~~ **Not true.**

| what S2 actually needs | current plan said | what it really requires |
|---|---|---|
| proof the parameters moved | save two full checkpoints and byte-compare | **a hash per parameter tensor, computed in memory** — a few hundred lines of digest, kilobytes, and it can be appended to an existing file or written node-local and retrieved |
| the run log | must live on Lustre | **node-local is fine** if it is pulled back promptly, exactly as S1c's 2.0 MB of artifacts were |
| a re-evaluable trained model | — | **this one genuinely needs durable storage.** A checkpoint you cannot reload is not a result |

So the storage chain is: **fingerprints and logs → node-local, retrieved immediately;
checkpoints → durable, and only when there is a model worth keeping.** S2's gate produces
the first two. Freeing inodes is required for S3's deliverable, not for S2's measurement.

Global quota is the coordinator's to manage and no other task's data is touched here.


## 11. S1b verified by a standalone reproducer — no GPU, no 9B, no Ray

The failure signature from the 9B trace, last six frames:

```
triton/runtime/jit.py:720 run -> :849 _do_compile
  -> triton/compiler/compiler.py:248 compile
    -> triton/runtime/cache.py:248 get_cache_manager -> :55 __init__
      -> <frozen os>:225 makedirs        <-- OSError errno 122
```

**The fault is in Triton's cache-manager constructor, in `os.makedirs`.** It has nothing to
do with the GDN kernel's contents, dtype or shape, and it happens during compilation —
before any device work. So the reproducer needs only *an uncached Triton kernel and an
unwritable cache directory*, and it runs on CPU.

Executed in the run's own environment (`envs/ldm-rl-train`):

| case | `TRITON_CACHE_DIR` | result |
|---|---|---|
| A, reproduces the 9B failure | `/projects/public/u6gb/.triton/cache_reproducer_probe` (Lustre, quota full) | **`OSError errno=122 Disk quota exceeded` at `<frozen os>:225 in makedirs`** — same frame, same errno as the 9B trace |
| B, the S1b fix | `/tmp/triton_reproducer_probe` | **succeeds**, `realpath` on tmpfs |

**Correction to my previous round.** I wrote that S1b's remaining verification "needs GPU
time" and drafted a minimal GPU request for it. That was wrong: the check takes about two
seconds on CPU. It was also self-contradictory — I proposed spending 711 s of 4-GPU wall on
a run whose initialisation alone costs ~900 s. **No GPU request is needed for S1b, and the
draft is withdrawn.**

Cache-path evidence, measured on nid010851 after `site_env.sh`, in the shell **and** inside
a Ray worker:

| check | result |
|---|---|
| declared value | `/tmp/triton_kangli.u6gb_6324130` |
| `realpath` | identical — no symlink hop |
| mount | `tmpfs on /tmp` — not Lustre |
| real child file | 64 KiB written, sha256 `de2f256064a0af79` |
| Ray worker | same env, same `realpath`, wrote a 1024-byte child |
| invalid path | Lustre `mkdir` fails EDQUOT as expected |

## 12. S4 minimal protocol and budget, specified before requesting devices

The baselines are the largest part of the main table that no 9B training gates. Written now
so a device request can be judged on the design rather than the design waiting on cards.

### Arms and alignment

| | |
|---|---|
| arms | **SFT-only** (`qwen3.5-9B-sft_torch_dist`) and **base** (`qwen3.5-9B_torch_dist`), no RL in either |
| eval sets | **G12D** in-distribution, **G12C** transfer |
| budget | **80 proposals** per (arm, set, seed) |
| held identical across arms | receptor 8UN5 chain A, box centre `(7.098, −12.100, −8.160)` and size `(20.421, 22.659, 22.227)` from the XQ6 co-crystal, `exhaustiveness=4`, `n_poses=3`, `vina seed=42`, the two QSAR models, the acquisition function, and the warm-start prefix |
| seeds | **3 sampling seeds** per (arm, set) = 12 runs. These measure sampling noise for a fixed checkpoint; there is no training replicate and no claim will be made as if there were |
| metric | Pareto hypervolume; **reference point and objective normalisation declared and frozen before the first run**, since a reference point chosen after seeing fronts is a free parameter |

### Cost

| component | estimate | basis |
|---|---|---|
| docking, CPU | **≈ 1.9 s per ligand** | measured in S1c: 3.8 s for 2 ligands at exhaustiveness 4. 80 × 12 = 960 ligands ≈ **30 CPU-min**, parallelisable |
| receptor prep | one-off, seconds | measured in S1c |
| QSAR scoring | negligible | sklearn on CPU |
| GP / acquisition | **unknown** | never profiled at budget 80; earlier work saw a single GP fit at ~67 s, which if it recurs per round would dominate everything else. **This is the first thing to measure** |
| **proposal generation, GPU** | **unknown** | depends on tokens per proposal and on whether the 9B fits one card. **Must be measured before a full request** |

### The measurement that would settle the unknowns

**One arm, one set, one seed, budget 80** — a single pilot row. It yields the per-round GPU
time, the GP cost at this budget, and the wall-clock per row, turning both unknowns into
numbers. Everything else in S4 is that row times twelve.

### Stop conditions and the smallest result that changes a judgement

- stop if the pilot row exceeds **1 GPU-hour**, and re-scope rather than continue;
- stop on any arm that cannot complete budget 80, and report it as not-reached;
- **what the pilot row delivers:** the per-round GPU time and the wall-clock for one
  budget-80 row, turning the remaining unknown into a number.

~~If the two baselines overlap, the baseline pair is indistinguishable and the RL
comparison needs rethinking before any RL run is worth its GPU time.~~ **Removed — it is
not a valid inference.** Overlapping intervals do not show equivalence, and whether SFT
improves on base says nothing about whether RL can improve on SFT: they are different
comparisons with different mechanisms. **S4's purpose is to establish reference levels and
to exercise the evaluation chain.** The RL question stays an independent open proposition,
and a single checkpoint evaluated at 3 sampling seeds can only support conclusions about
that checkpoint.


## 13. S4 pilot: ready, waiting on a conflict-free device

Everything except the device is done. **No further code review, environment work or
namespace testing is required** — the next candidate node needs only its UUIDs, namespace
and occupancy checked, then a fresh lease.

| artefact | location | sha256 |
|---|---|---|
| worker | `nid010292:/tmp/kangli.u6gb/s4pilot/s4_worker_local.sh` | `14d0625d9184b217aa5162e68222ba0afca349ef4ba6b319ec9a3815cf5faaf9` |
| spec (coordinator's resource fields) | node-local | `53322682aa156561ee10241a1836edcc38c543fcceb77c7a252054e82892d2ee` |
| guard helper / checker | `/tmp/kangli.u6gb/shared-guard/30b9b86ab92bb078/` | `30b9b86a…4a91a7` / `cf2ee3c0…d99c35` |

### Attempt on nid010292, 22:03:25Z — guard withheld, correctly

`srun` step `6324119.1892`, **exit 73**. All four hash checks passed, then:

```json
{"guard":"withheld",
 "reason":"device occupied or memory/process evidence invalid: GPU-6bd807d0-...",
 "next_action":"report to coordinator; obtain a new lease; do not delete started markers or select alternatives"}
```

The coordinator's independent 22:08:29Z run of the same checker found **GPU0 90,876 MiB /
97% held by PID 150127** and **GPU1 91,439 MiB / 100% held by PIDs 150127 and 154674**.

**The guard was right and my earlier reading was stale.** At 21:47:28Z I measured 1 MiB and
zero compute processes on those cards; roughly sixteen minutes later they were nearly full.
A device reading is only true at its sample instant, which is why the guard re-checks at
exec time rather than trusting a launcher's earlier look. Nothing was started, no marker
deleted, no alternative card selected.

**Budget:** the step failed 6 s before the worker began — **0.003333 GPU-h on 2 cards**,
deducted from the 1.5 GPU-h authorisation. Remaining **1.496667 GPU-h**. The 75 min / 2.5
GPU-h figure I proposed after the EHVI profile was **not authorised**; the spec's resource
fields stand at **45 min / 1.5 GPU-h**. The reason for wanting more is recorded and
unapproved: EHVI costs 8.87 s per round at pool 32, so 80 rounds are about 12.6 minutes of
CPU acquisition before any generation or docking, which makes 45 minutes tight rather than
impossible.

### What happens when the next candidate arrives

1. check the new node's two UUIDs, guard namespace and occupancy — read-only, done without
   asking;
2. report those to the coordinator;
3. run the guarded command with the fresh lease.

No code review, no environment setup, no namespace test, no retry of an old lease.


## 14. CPU results preserved, 2026-09-05 22:15 UTC

### Artefacts, exact paths and hashes

All on the login node `nid010777`, node-local `/tmp`, retrievable:

| file | path | bytes | sha256 |
|---|---|---:|---|
| `acq_profile.json` | `/tmp/s4_profile_20260905T211627Z/acq_profile.json` | 983 | `9281588b831cf4911fcf8ce58c15c234e85e35a2ba6fa482d4881b987f276c02` |
| `gp_profile.json` | `/tmp/s4_profile_20260905T211627Z/gp_profile.json` | 896 | `8aaebc01b7d3f0bcfa5616e83fdc2b0f89724889cdb75bb58063ff79194c7c66` |
| `pilot_spec.json` (mine) | `/tmp/s4_profile_20260905T211627Z/pilot_spec.json` | 6070 | `a39c09a4a4aeb76b0b2346133a30aedb8dc6c787c0acf435111c91d670bd8e0e` |
| `pilot_spec.json` (coordinator's resource fields, authoritative) | node-local on nid010292 | — | `53322682aa156561ee10241a1836edcc38c543fcceb77c7a252054e82892d2ee` |
| S1c docking package | `/tmp/s1c_8un5_210533Z.tar.gz` | 208849 | `c8053734197dcc2ebd06229f6db72e5b7175e61d510fb14e3d6ff903f1fd0b3d` |

### The measurement's actual device, stated precisely

**`device_declared = cuda`, `device_resolved = cuda`.** The profile ran on GPU, not CPU.
I twice described it as a "CPU fallback" and that was wrong; the spec's `cuda` is correct.

But the two halves differ and the distinction matters for the budget:

| component | device | cost |
|---|---|---|
| GP fit + predict | **cuda** (torch 2.11.0+cu129, `envs/ldm-rl` python 3.12.14) | cold start **7.027 s once per run**; steady state **0.563 s** for both objectives |
| **EHVI** | **CPU** — `acquisition.py` is numpy, no torch | **8.87 s per round at pool 32**, linear at **0.277 s per candidate** |

So EHVI is the dominant per-round term at 63x the GP fit, and **it will not get faster on a
different device**, because it never used one. Pool 32 x 80 rounds is about **12.6 minutes
of CPU acquisition** inside the 45-minute wall, before any generation or docking.

### Steps completed

| step | outcome |
|---|---|
| S0 evaluation chain | 11/11 packages; both QSAR models predict from raw SMILES |
| S1 Vina + imports | aarch64 Vina `f458505-mod` already present in `envs/ldm-rl`; 6/6 eval modules import |
| S1b Triton cache | failure reproduced on CPU at `<frozen os>:225 makedirs` errno 122; node-local `/tmp` succeeds; guard added |
| S1c 8UN5 + docking | receptor prepared from co-crystal XQ6; aspirin **−6.804**, benzamide **−5.925** kcal/mol, 3.8 s for two ligands |
| HV implementation | replayed `real_80_qwen35_9b`: HV@1 **13.7081**, HV@80 **18.4541**, identical to the recorded values |
| budget semantics | read from `runtime.py:647`: budget counts **successful evaluations**, not proposals and not rounds |
| GP / EHVI profile | above |
| worker + guard | deployed, hashes verified by the launcher, guard correctly withheld on a genuinely occupied device |

### Questions that remain, and what each needs

| question | needs |
|---|---|
| inference init wall-clock without Megatron/Ray | the pilot |
| seconds and tokens per proposal for the 9B on 2 cards | the pilot |
| whether the observed candidate pool is the pinned 32 | the pilot |
| does the 45-minute wall actually fit 80 successful evaluations | the pilot; if it does not, the run reports not-reached rather than being extended |
| **S2's finite-gradient gate** | inodes freed for checkpoint writes — unrelated to S4 |
| **the RL question itself** | S3, which needs S2 |

Nothing on this list can be answered by more CPU work.


## 15. Conditional next candidate — nid010624, GPUs 0 and 1 only

Recorded 2026-09-05 22:20 UTC (Sydney 2026-09-06 08:20 +1000). **Nothing is launched and
nothing is pre-empted.**

| | |
|---|---|
| allocation / node | **6324128 / nid010624** |
| devices | **GPU 0 and GPU 1 only** |
| **GPU 2 and 3** | **reserved for M3 — never used, not even transiently.** The run requests 2 cards, not 4 |
| condition | valwire step `6324128.1840` (s7/base) was at 85% with an estimated 6 minutes left as of 22:11:16 UTC. **The estimate is not the release.** Start only after a real exit *and* a fresh NVML read showing the two cards idle |
| refill | root has asked that this node not be refilled once valwire saves its results; that request is respected |

### Why the wait is not conservatism

The previous attempt on nid010292 failed exactly here. At 21:47:28Z I measured 1 MiB and
zero compute processes; sixteen minutes later the coordinator's checker found **90,876 MiB
at 97%** on the same card, and the guard withheld. **A completion estimate is a weaker
signal than the stale reading that already misled me once.** So the trigger is the exit
plus a fresh read, never the estimate.

### What is already in place, and what is not repeated

Carried over unchanged: the worker (`14d0625d…f5faaf9`), the authorised spec
(`53322682…92d2ee`), the guard helper and checker (`30b9b86a…4a91a7`, `cf2ee3c0…d99c35`),
and the scientific scope — SFT-only, G12D, seed 42, 80 successful evaluations, TP=2.

**Not repeated:** code review, environment setup, namespace testing, docking, the HV
replay, the GP/EHVI profile. **Not reused:** the spent lease from attempt 1.

### Budget

| | GPU-h |
|---|---|
| authorised | 1.500000 |
| attempt 1, failed 6 s before the worker started | −0.003333 |
| **remaining** | **1.496667** |

45 minutes wall, unchanged. No extension is requested; if 80 successful evaluations do not
fit, the run reports not-reached.

### Sequence when the device actually frees

1. fresh NVML read of nid010624 GPU 0 and 1 — read-only, done without asking;
2. new node's two UUIDs and guard namespace dev:ino reported;
3. coordinator signs a fresh lease;
4. guarded foreground launch, then job.step, guard PID, worker PID and the first sglang
   model-loading line.
