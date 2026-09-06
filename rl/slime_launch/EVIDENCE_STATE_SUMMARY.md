# What is proven, what is conditional, what is unknown

2026-09-06. Written so the Max adjudication can decide remaining validation without
re-reading the source. No scientific-benefit upgrade, no GPU budget expansion.

## PROVEN about the CURRENT code (hashes above each claim)

| claim | evidence |
|---|---|
| task success predicate = **both** `vina` and `activity` non-None | `engine_adapters.py` `ed63185ecc09ed88`, lines 265-289; else `"failed"`, `error="non-finite objective score after retries"` |
| only `succeeded` results are written to `gp_history.jsonl` | `rl_real_shared.py` `82f7125fe4f3a0d6`, line 98 gate, line 101 the single `store.append(` call site |
| non-finite scores are normalised to `None` before the predicate | `loop.py` `0b181d5ae57d9894`, `_score_smiles_with_diagnostics` applies `finite_or_none` |
| retry budget is **2 attempts**: one batch call + at most one individual retry per bad index | `loop.py:48` `MAX_SCORE_ATTEMPTS = 2` |
| docking `status=="ok"` is one leg only | `docking.py:3325/3385/3400`; non-ok = `prep_failed`, `dock_failed` |
| per-request sampling seed is guarded off | `sglang_rollout.py:110,583` both behind `sglang_enable_deterministic_inference`, which reads **False** in R3a / X-dhvn4-170233 / LR0 while `rollout_seed`=42 |
| an engine-init seed does exist and is separate | `sglang_engine.py:547`, `random_seed = args.seed + rank * num_gpus_per_node` |

## CONDITIONAL on historical code binding (NOT yet established)

Every statement about the **historical** `gp_history.jsonl` files inherits this condition.
Rows carry only `["activity","smiles","vina"]` — **no version, writer id or timestamp** — so
the file cannot say which code wrote it. Unbound: per-run code SHA/snapshot/symlink;
whether an alternate append path existed then; whether prior-seeding behaved identically.
**Resolving this needs per-run code provenance, not more reading of the current tree.**

Also conditional: "generation was unseeded" holds for the three runs whose argv was read.
Per-request seed passed, engine-init seed, and end-to-end reproducibility are **three
different propositions** and only the first two were checked.

## UNKNOWN — genuinely, and not to be filled by inference

| unknown | why it is not derivable |
|---|---|
| **the failure denominator** ("80 successes out of how many attempts") | failures are never written to `gp_history`; the count lives only in `EvaluationResult.metadata["diagnostics"]`, which is not persisted there |
| **scorer-call count behind N rows** | retries add one call per bad index; a molecule succeeding on retry still yields exactly one row. Rows ≠ calls in either direction |
| **1135 ligands vs 1123 cache entries** (R3a) | an object-count difference between directories. Duplicates, intermediate products, differing cache keys and in-flight work all produce it. **Not** 12 failed calls |
| **2247 poses vs 1123 cache entries** | roughly 2:1; not 2247 dockings |
| **cache-hit billing** | whether a cache hit consumes budget is not recorded in any artifact read |

## CLOSED

The raw-row replication (80/80/80/83/80/80/81, first 63 prior rows excluded, `seen` not
pre-seeded) is **closed by owner verification against full SHA256 of all seven files**. It
fixes a counting convention only — not scientific benefit, not budget authorization. Not
recomputed here.

## OUTSTANDING work, with its real blocker

B3 and B4 were dispatched as native agents (`a6316509970de572b`, `a549dfcce58855d90`) and
**both died on HTTP 429 session limit** (resets 09:40 UTC) before writing their mutual
replies. B2/B5 cross-attacks and the Max adjudication are therefore also outstanding. See
`agent_bank/ID_LINEAGE.md` for request ids and durable transcript paths.
