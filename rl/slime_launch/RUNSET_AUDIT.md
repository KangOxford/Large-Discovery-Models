# Runset audit — CORRECTED 2026-09-06 after owner review LDM_OWNER_REVIEW_0538

The previous version of this file made claims that do not survive contact with the B
reports. They are withdrawn here in full, and the corrected work follows. The earlier
commit (`33513678`) is kept as the record of what was claimed.

## Withdrawn

| withdrawn claim | why it fails |
|---|---|
| "the B4 vs B1/B3 duplicate conflict is **resolved**" | It is not. See the reconciliation below: B4's 0–3 does not reproduce on the 13-run dhvn4/LR0 set under **either** seeding rule. |
| "they are **two stages of the same pipeline**, both correct" | **B1 had already refuted this** (progress.md 380–388): the RL runs' `gp_history` and the harness history "are produced by different code with different duplicate semantics, so they are not the same measurement", and dropping a duplicate before evaluation changes what the GP is fitted on, so the trajectory itself differs. I used the harness artifact to arbitrate a dispute it cannot arbitrate. |
| "budget charge **lives at the pre-dedupe stage**", from `llm_call_count 1931` vs `successful_evaluation_count 80` | 1931/80 is a **generation cost ratio per successful evaluation**. It is not 1931 successful evaluations and says nothing about which stage is billed. |
| "**11 of 16 fields absent from every artifact**" | The script initialised every required field to `None`, never searched for most of them, then reported the untouched `None`s as ABSENT. It also read only `_wt_handoff` examples — no run directories, logs, configs, caches or source. |
| "no per-call succeeded predicate can be recomputed" | **False.** `runs/R3a_*/env_out/vina_cache/cache/*.json` carries `status`, `message`, `cached`, `canonical_smiles`, `score` per compound. I had not looked in the run directories. |
| "8UN5 not found within depth 4" | **False.** `runs/R3a_*/env_out/vina_cache/receptors/8UN5_A_XQ6_clean.pdb` exists. |
| "estimands fixed **before any result is seen**" | Historical HV values had already been seen. Listing three estimands is not choosing one. Restated below as what it actually is. |

## The reconciliation, done on one file with the conventions varied

`plan/dup_reconcile.py`. Scope stated: reads `ldm_rl/runs/*/gp_history.jsonl` and
`env_out/vina_cache/cache/*.json` where present. It **excludes** the `real_80` harness
artifact, for B1's reason above.

**On R3a alone** (`gp_history.jsonl` sha256 `dc27c4e499513018`, 1273 rows, 1210 post-prior,
41 distinct prior SMILES), all three published ranges reproduce, and the discriminator is
the prior-seeding rule, not the file and not the key:

| convention | rows to 80 unique | excess |
|---|---:|---:|
| post-prior, prior **not** seeded as seen | **80** | +0 |
| post-prior, prior seeded as seen | **81** | +1 |
| from file start | **102** | +22 |

Raw and RDKit-canonical keys give identical answers here (80 / 81), so the dedupe key is
**not** the explanation.

**On the wider 13-run set** (9 `X-dhvn4-*`, 4 `LR0*`) the clean story breaks:

| convention | observed range | matches |
|---|---|---|
| from file start | **102–111 (+22..+31)** | **reproduces B3 exactly** |
| post-prior, not seeded | 81–92 (**+1..+12**) | close to B3's stated +1..+11 |
| post-prior, seeded | 85–98 (+5..+18) | matches neither |
| **B4's 0–3** | **not reproduced on any of the 13** (minimum +1; only R3a gives +0) | **unreproduced** |

**So the conflict is narrowed, not closed.** Of B3's three candidate explanations, the live
one is **#2 different run set** — not #1 "different file", which B3 itself marked most
likely. B4 must state its exact run ids and the file it read.

## Success predicate — confirmed for one run, unchecked for the rest

`vina_cache` exists **only** for R3a among the runs examined. There:

```
cache entries 1123, status ok 1123, status not-ok 0
distinct canonical_smiles in cache 1123
ligands prepared 1135  ->  ligands minus cache = 12
```

The 12-ligand gap is a real signal and is **not** explained by this pass: 1135 ligands were
prepared but only 1123 reached a docking cache entry. Whether those 12 are failures,
duplicates collapsed by canonical key, or in-flight work is **NOT CHECKED**.

For the other 13 runs the state is **NOT_CHECKED_no_cache_dir** — not "absent". Whether
their success status is recoverable from execution paths or other caches must be settled
**per run**, and neither of these is acceptable shorthand: a finite field is not success,
and a simplified history that omits status does not prove status is unrecoverable.

## Counting categories that must stay separate

Tracked separately from here on, never collapsed: **proposals** / **LLM generation calls**
/ **upstream dedupe drops** / **evaluator invocations** / **succeeded count** / **HV**.
A low-scoring but succeeded evaluation **counts toward the 80**. An unscored duplicate or a
generation call **does not** automatically count as a successful evaluation.

## Estimands — stated, not frozen

`HV(new)`, `HV(prior ∪ new) − HV(prior)` and `HV(new) − HV(prior)` are three different
targets, and for a fixed constant `c`, `Var(HV − c) = Var(HV)` — the withdrawn
variance-reduction argument stays withdrawn. **This is an enumeration, not a choice**: I
have already seen historical HV values, so nothing written here can claim to be
pre-registration. The historical HV work stays labelled **exploratory diagnostic**.

The prospective fixing — exact primary estimand, reference point, prior, success rule and
scientific thresholds — belongs to the B cross-review and the actual Max adjudication, in
advance, for the future plan. No metric is to be selected after seeing results.

## Evaluator and sampler, read at source this session

**Sampler — the seed is recorded but never passed.** `sglang_rollout.py` assigns
`sampling_params["sampling_seed"]` in exactly two places (lines 110 and 583), and **both are
guarded by `sglang_enable_deterministic_inference`**. In `R3a`, `X-dhvn4-170233` and
`LR0-20260905T034032Z` that flag reads **False** in the resolved argument dump while
`rollout_seed` reads **42**. So generation was unseeded in these runs, confirming A2/X-A2.1
at source with the exact guard. A separate seed does reach the engine —
`sglang_engine.py:547` sets `random_seed = args.seed + rank * num_gpus_per_node` at init —
so "seed" must always be qualified: **engine init seed** and **per-request sampling seed**
are different things, and only the former was active.

**Evaluator — the success predicate is a real returned status.** `docking.py` returns
`status="ok"` (line 3325) and `dock_batch` filters on `result.status == "ok"` (3385, 3400).
The non-ok values it can return are **`prep_failed`** (3216, 3226, 3360) and **`dock_failed`**
(3237, 3296, 3305, 3317). `ok_unique` / `ok_pubchem_formula` / `ambiguous` / `not_found` are
a *different* field (`lookup_status`) and must not be conflated with docking status.

So the success predicate is observable in principle, and observed for R3a
(1123/1123 `ok`). Whether the 13 other runs' statuses are recoverable is **per-run open**.
