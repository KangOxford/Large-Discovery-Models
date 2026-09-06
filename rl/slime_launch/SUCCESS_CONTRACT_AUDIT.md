# Success predicate and budget denominator — CURRENT code path traced; historical binding OPEN

2026-09-06. Read at these exact files, hashes independently matched by the owner:

| file | sha256 (16) |
|---|---|
| `tasks/small_molecule/core/rl_real_shared.py` | `82f7125fe4f3a0d6` |
| `tasks/small_molecule/core/engine_adapters.py` | `ed63185ecc09ed88` |
| `tasks/small_molecule/core/rl_real.py` | `24b117d5f3fbf468` |
| `tasks/small_molecule/core/ldm_tilted_case2/loop.py` | `0b181d5ae57d9894` |

**Everything below establishes the CURRENT code path only.** It does **not** establish which
writer or which code version produced each historical `gp_history.jsonl`. That binding is
an open gap, quantified in its own section. The prospective contract at the end is **not**
pre-registration.

## The chain, end to end

| step | file:line | what it does |
|---|---|---|
| build real components | `rl_real.py:68-85` | `vina_fn, activity_fn = build_real_scorers(...)`; `SmilesCandidateEvaluator(vina_fn, activity_fn)`, wrapped by `SharedEvaluator(evaluator, store)` when `gp_history_file` is set |
| **task success predicate** | `engine_adapters.py:265-289` | `status = "succeeded"` **iff BOTH `vina` and `activity` are non-None**; else `"failed"` with `error="non-finite objective score after retries"` |
| **write gate** | `rl_real_shared.py:98` | appends `{smiles, vina, activity}` to `gp_history.jsonl` **only when `status == "succeeded"`** |
| scoring + retries | `loop.py:467-520` | `_score_smiles_with_diagnostics` runs each scorer through `_score_with_retries_with_diagnostics`, then applies **`finite_or_none(value)`** — a non-finite score becomes `None`, which is exactly what the predicate above tests |
| retry budget | `loop.py:48` | **`MAX_SCORE_ATTEMPTS = 2`** — one batch call, then at most one **individual** retry per non-finite index. Each retry is a **separate scorer call**, recorded in `diagnostics[idx]["attempts"]` with `final_value` and `final_finite` |
| docking component | `docking.py:3325, 3385, 3400` | returns `status="ok"`; non-ok values are `prep_failed` and `dock_failed` |

## Retries make the call denominator strictly larger than the row count

Because a non-finite score triggers **one extra individual scorer call per bad index**
(`MAX_SCORE_ATTEMPTS = 2`), the number of scorer invocations behind N successful rows is
**N + (number of retried indices)**, and a molecule that succeeds only on its retry still
produces exactly one `gp_history` row. **So rows cannot be read as calls in either
direction**, and the retry count is recorded only in `EvaluationResult.metadata`
["diagnostics"], which `gp_history` does not persist.

## Three consequences, and one retraction

**1. `vina_cache status=="ok"` is NECESSARY BUT NOT SUFFICIENT.** The task predicate needs
both legs. A molecule can dock cleanly and still fail on the QSAR/activity leg, returning
`"failed"` and never reaching `gp_history`. My earlier label
`CONFIRMED_FROM_vina_cache` claimed a whole-task success predicate from a component status
and is **retracted**; `plan/dup_reconcile.py` now emits
`DOCKING_COMPONENT_STATUS_ONLY_not_task_success_predicate` and names the real contract.

**2. `gp_history.jsonl` is the task-level success log *for files written by this code
version* — from the write gate, not from field non-nullness. Conditional on the historical
binding above.** B1 and B4 both reported this from field non-nullness; the write gate at
`rl_real_shared.py:98` establishes it directly. B3's recorded limit ("a field being
non-null and finite does not prove the upstream success predicate held") is answered **for
this specific file**: the gate is upstream of the write, so every row is a success by
construction. The limit still stands wherever no such gate is shown.

**3. The failure denominator is NOT recoverable from `gp_history`.** Failures are never
written there. Any per-call denominator has to come from the diagnostics carried in
`EvaluationResult.metadata["diagnostics"]` or from the run logs — neither of which
`gp_history` preserves. **So "80 successful evaluations" is computable while "out of how
many attempts" is not, from this file alone.**

## Artifact counts are not call counts

`runs/R3a_*/env_out/vina_cache` holds 1123 `cache/*.json`, 1135 `ligands/`, 2247 `poses/`,
7 `receptors/`. The 1135 − 1123 = 12 difference is an **object-count difference between two
directories**, not twelve failed calls: duplicates, intermediate products, differing cache
keys and in-flight work all produce it. It is recorded as
`ligands_minus_cache_UNEXPLAINED` with `ligands_minus_cache_is_NOT_failure_count: true`,
and stays **UNKNOWN** until traced. Likewise 2247 poses against 1123 cache entries is
roughly 2:1 and is not 2247 dockings.

## Categories that must stay separate

proposals → LLM generation calls → parse/admit rejections → upstream dedupe and cache hits
→ evaluator invocations → **succeeded** (both legs) → rows in `gp_history` → HV.
A low-scoring but succeeded evaluation **counts** toward 80. A cache hit, a duplicate, a
parse rejection or a generation call **does not**.

## The historical binding gap, stated as a bounded unknown

`gp_history.jsonl` rows carry exactly `["activity", "smiles", "vina"]` — **no code version,
no writer id, no timestamp**. So the file cannot identify which version wrote it, and the
current-code trace above transfers to the historical runs **only conditionally**.

What is known: `store.append(` appears at exactly one call site in the current tree
(`rl_real_shared.py:101`), so there is one append path **today**. What is NOT known, and is
not inferable from these files:

1. which code SHA / snapshot / symlink each historical run executed;
2. whether an alternate append path existed at that time;
3. whether the prior-seeding behaviour of `SharedHistoryStore` was the same;
4. therefore, whether "every row is a success by the predicate above" holds for the
   historical files, or only for files written by this version.

**Resolving it needs the per-run code provenance, not more reading of the current tree.**
Until then, statements about historical `gp_history` files are conditional on that binding.

## Raw-count dispute: closed by the owner, not reopened here

The owner independently read all seven original `gp_history` files, verified full SHA256,
and confirmed 80/80/80/83/80/80/81 excluding the first 63 prior rows without pre-seeding
`seen`. **That narrow replication is closed and is not recomputed.** It establishes a
counting convention only — **not** scientific benefit and **not** budget authorization.

## Prospective contract — to be fixed in advance, not claimed now

The following must be declared **before** any comparison run, by the B cross-review and the
Max adjudication, not retrofitted here: the exact primary estimand; the reference point and
its normalisation; the prior manifest; the success rule (this document's predicate, stated
explicitly rather than inherited); the dedupe key and whether cache hits are billed; the
replication unit; and the scientific thresholds. Historical HV values have already been
seen, so nothing in this repository can be presented as pre-registration.
