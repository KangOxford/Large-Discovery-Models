# Common-runset audit — can the existing data form the main comparison?

2026-09-06. Read from the artifacts on disk this session. Evidence table:
`results/runset_audit.json` (producer `plan/runset_audit.py`).

## Verdict first

**No. The existing artifacts cannot form a traceable equal-successful-budget main-task
comparison.** Five evaluation artifacts exist; two are small-molecule with a hypervolume
number. Of the sixteen fields the comparison needs, **eleven are absent from every artifact
on disk**:

`model_size`, `init_from`, `train_seed`, `eval_seed`, `checkpoint`, `scorer`, `code_hash`,
`prior_manifest`, `smiles_canonical`, `failure_types`, `budget_charge_per_call`.

What exists is one scorer-replay summary carrying `final_hypervolume 18.454` and a history
of bare `(scores, smiles)` pairs, plus a second, smaller artifact
(`run_small_molecule_w_delta_infra`, HV 15.310, 30 records) whose history does carry
`vina`, `predicted_activity` and `round_idx`.

**There is no per-call succeeded predicate in either artifact.** `real_80`'s
`successful_evaluation_count: 80` is an aggregate in `summary.json`; the other artifact's
`evaluation` field is an integer counter, not a status. So the successful-evaluation
predicate — the thing B3's withdrawn "80 unique AND ref-dominating" label was arguing
about — cannot be recomputed from either file.

## The B4 versus B1/B3 duplicate conflict is resolved: different pipeline stages

| | B1 / B3 | B4 |
|---|---|---|
| source | `assets/examples/real_80_qwen35_9b` | `ldm_rl/runs/*/gp_history.jsonl` |
| what it counts | rows that **survived** upstream dedupe and entered history | the **raw proposal stream**, before that dedupe |
| observed | 80 rows, 80 unique → **0 duplicates** | R3a **1210 rows / 1123 unique** |

`real_80/summary.json` settles it in its own fields:

```
drop_counts = {'duplicate': 4921, 'evaluated': 2607, 'invalid': 471, 'overlength': 7}
successful_evaluation_count = 80    failed_evaluation_count = 0    llm_call_count = 1931
```

**4921 duplicates were dropped before anything reached history.** B3 counting zero
duplicates inside the surviving 80 is therefore correct, and so is B4 counting duplicates
in the pre-dedupe stream. Neither count is evidence about the other, and neither party was
wrong. Any figure quoted from here on must name its stage.

**Where the budget question actually lives.** `llm_call_count 1931` against
`successful_evaluation_count 80` means the charge and the success count are separated by
roughly 24x at this stage. A low-scoring but successful call still consumed budget, so
budget accounting belongs at the **pre-dedupe** stage, not at the surviving-80 stage.

## What this changes about the next step

The blocking work is **not** more runs. It is that no artifact records the provenance a
comparison needs. Before any new GPU row is requested, the evaluator and sampler
implementations actually used by a run must be read directly, and a run must emit:
checkpoint id, scorer id, code hash, training seed, the seed actually passed to the
sampler (recording a seed is not passing it), canonical SMILES key, the raw succeeded
field with failure type, and the per-call budget charge.

## Estimands — fixed here, before any result is seen

Three different targets, not interchangeable:

| | |
|---|---|
| `HV(new)` | hypervolume of the new points alone |
| `HV(prior ∪ new) − HV(prior)` | increment over the prior union |
| `HV(new) − HV(prior)` | difference of two separately-computed volumes |

For a fixed constant `c`, `Var(HV − c) = Var(HV)`. **The withdrawn variance-reduction
argument must not reappear** in any form. The primary metric is fixed before results are
seen, G12D and G12C stay separate, and the comparisons remain SFT+RL vs SFT-only, base+RL
vs SFT+RL, and acquisition max/mean vs real ΔHV, with the run as the replication unit.
The 1.5B harness does not substitute for the 9B main task.

## Evidence carried forward unchanged

- Seven historical 9B logs: **238 norms = 209 NaN / 0 Inf / 29 finite** (per
  `CORRECTION_grad_norm_count.md`). The "all non-finite" claim is withdrawn; the 29 finite
  norms are **not** evidence of valid optimizer updates or of RL benefit.
- A4's raw-byte/alignment inference and the missing provenance behind HV 18.454 remain
  **open unresolved items**, not settled facts.
- The lr=0 diagnostic (`harvest_lr0_arm.py`, SHA `81ed32ef…4558`) is a **conditional
  descriptive** contrast between differently-selected completer groups: 7/20 trained and
  4/5 control reached 400 records; means +0.0286 / +0.0946; difference −0.0660, 95% CI
  [−0.1475, +0.0155]; TOST lower p 0.1762 / upper p 0.0011, equivalence not shown. It is
  **not** HV@80 and not a common-population RL causal effect. It is not to be mechanically
  extended.
- No current GPU grant. All gtop figures are historical observations. S4 remains stopped;
  its 1.208889 GPU-hour balance is confined to the original SFT-only / G12D / seed 42 /
  80-successful-evaluation scope across all phases, and neither the GATE nor the lr=0
  budget transfers into it.
