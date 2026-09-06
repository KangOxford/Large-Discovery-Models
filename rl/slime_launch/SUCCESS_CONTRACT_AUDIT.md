# Success predicate and budget denominator — traced at source

2026-09-06. Historical evidence only; the prospective contract is stated separately at the
end and is NOT claimed to be pre-registered.

## The chain, end to end

| step | file:line | what it does |
|---|---|---|
| build real components | `rl_real.py:68-85` | `vina_fn, activity_fn = build_real_scorers(...)`; `SmilesCandidateEvaluator(vina_fn, activity_fn)`, wrapped by `SharedEvaluator(evaluator, store)` when `gp_history_file` is set |
| **task success predicate** | `engine_adapters.py:265-289` | `status = "succeeded"` **iff BOTH `vina` and `activity` are non-None**; else `"failed"` with `error="non-finite objective score after retries"` |
| **write gate** | `rl_real_shared.py:98` | appends `{smiles, vina, activity}` to `gp_history.jsonl` **only when `status == "succeeded"`** |
| docking component | `docking.py:3325, 3385, 3400` | returns `status="ok"`; non-ok values are `prep_failed` and `dock_failed` |

## Three consequences, and one retraction

**1. `vina_cache status=="ok"` is NECESSARY BUT NOT SUFFICIENT.** The task predicate needs
both legs. A molecule can dock cleanly and still fail on the QSAR/activity leg, returning
`"failed"` and never reaching `gp_history`. My earlier label
`CONFIRMED_FROM_vina_cache` claimed a whole-task success predicate from a component status
and is **retracted**; `plan/dup_reconcile.py` now emits
`DOCKING_COMPONENT_STATUS_ONLY_not_task_success_predicate` and names the real contract.

**2. `gp_history.jsonl` is exactly the task-level success log — now from code, not
inference.** B1 and B4 both reported this from field non-nullness; the write gate at
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
