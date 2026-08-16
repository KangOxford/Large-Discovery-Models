# AI4Bio Mutation Effect Prediction

This adapter searches bounded supervised prediction heads for the pinned
[MLS-Bench task](https://github.com/Imbernoulli/MLS-Bench/tree/cfd57a7e0139c72753e32e31bca593719b098717/tasks/ai4bio-mutation-effect-prediction).
Candidates map frozen 1280-dimensional ESM-2 mutant and mutant-minus-WT
embeddings to scalar protein-fitness predictions.

## Status

Registration, deterministic mock execution, CPU/CUDA contract checks, and the
official ridge seed are verified through `LDMEngine` and the pinned evaluator.
The experiment contract is `qualified` and exposes the locked
`official_campaign` profile. The seed evidence is versioned in
`resources/qualification_seed.json` and is explicitly outside campaign budget.

## Candidate Domain

The response contract accepts strict JSON architecture specifications rather
than arbitrary Python. A candidate selects mutant, delta, or concatenated
features; zero to three dense hidden layers; ReLU, GELU, or SiLU; bounded
dropout and optional layer normalization; and bounded AdamW learning rate and
weight decay. The adapter materializes a benchmark-compatible
`MutationPredictor` class and rejects malformed, non-finite, duplicate, or
over-budget candidates before evaluation.

The upstream budget is 6,957,956 trainable parameters, calculated as
`floor(1.05 * 6,626,625)` from the largest bundled baseline. The fixed
candidate representation is encoded into 15 normalized features for shared
exact-RBF GP-UCB selection.

## Evaluation Contract

The official evaluator uses ProteinGym random five-fold CV for
`BLAT_ECOLX_Firnberg_2014`, `ESTA_BACSU_Nutschel_2020`, and
`RASH_HUMAN_Bandaru_2017`. It reports mean Spearman for each assay and combines
the three through the pinned MLS-Bench scoring engine. That engine first applies
baseline-anchored `bounded_power` normalization to each assay, then computes its
epsilon-floored geometric mean. `official_score` is therefore not the raw
geometric mean of the three Spearman values. The task does not use ProteinGym's
modulo or contiguous folds, so results are MLS-Bench task results, not
ProteinGym supervised-leaderboard results.

## Mock Run

From the repository root:

```bash
python scripts/validate_tasks.py --task ai4bio_mutation_effect_prediction
uv run --project tasks/ai4bio_mutation_effect_prediction python -m pytest \
  tasks/ai4bio_mutation_effect_prediction/tests
uv run --project tasks/ai4bio_mutation_effect_prediction python \
  scripts/check_task_dependencies.py \
  config/ai4bio_mutation_effect_prediction/mock.yaml --no-optional
uv run --project tasks/ai4bio_mutation_effect_prediction python \
  scripts/run_ldm_tts.py config/ai4bio_mutation_effect_prediction/mock.yaml
```

Mock runs write campaign, contract, task-spec, event, checkpoint, budget,
status, search, selection, and summary artifacts under `runs/`. Accepted
architecture specs are collectable as canonical `ldm-2.0` IR when
`LDM_DATA_COLLECTION_ENABLED=1`.

## Official Campaign

Set `upstream-root`, `data-dir`, and `cv-dir` in a protected runtime copy of
`config/ai4bio_mutation_effect_prediction/real.yaml`. The source must be the
pinned MLS-Bench commit, `data-dir` must contain the three official ESM-2 tensor
payloads, and `cv-dir` must contain the matching ProteinGym random-fold CSVs.
Then run:

```bash
python scripts/validate_tasks.py --task ai4bio_mutation_effect_prediction \
  --require-qualified
uv run --locked --project tasks/ai4bio_mutation_effect_prediction python \
  scripts/check_task_dependencies.py \
  config/ai4bio_mutation_effect_prediction/real.yaml --no-optional
uv run --locked --project tasks/ai4bio_mutation_effect_prediction python \
  scripts/run_ldm_tts.py config/ai4bio_mutation_effect_prediction/real.yaml
```

The locked profile admits four deterministic candidates, GP-UCB selects one,
and exactly one expensive evaluation launches three official benchmark jobs.

For an explicitly extended comparison budget, populate the external paths in
`config/ai4bio_mutation_effect_prediction/real_3_iterations.yaml`. Its locked
`official_campaign_3_iterations` profile runs three sequential GP-UCB rounds,
selects and evaluates one candidate per round, and accounts for nine official
benchmark jobs. Results from this profile must be labeled extended-budget and
must not be compared as if they used the one-evaluation primary budget.

The same qualification applies to
`config/ai4bio_mutation_effect_prediction/real_20_iterations.yaml`. Its
`official_campaign_20_iterations` profile is a 20-round extended-budget run
with 20 selected candidates and 60 official assay jobs. It is intended for
trajectory analysis, not direct primary-budget comparison.
