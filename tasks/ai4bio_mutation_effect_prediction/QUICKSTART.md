# Quickstart

From a clean repository checkout:

```bash
uv sync --project tasks/ai4bio_mutation_effect_prediction --group dev
python scripts/validate_tasks.py --task ai4bio_mutation_effect_prediction
uv run --project tasks/ai4bio_mutation_effect_prediction python -m pytest \
  tasks/ai4bio_mutation_effect_prediction/tests
uv run --project tasks/ai4bio_mutation_effect_prediction python \
  scripts/check_task_dependencies.py \
  config/ai4bio_mutation_effect_prediction/mock.yaml --no-optional
uv run --project tasks/ai4bio_mutation_effect_prediction python \
  scripts/run_ldm_tts.py config/ai4bio_mutation_effect_prediction/mock.yaml --dry-run
LDM_DATA_COLLECTION_ENABLED=1 \
uv run --project tasks/ai4bio_mutation_effect_prediction python \
  scripts/run_ldm_tts.py config/ai4bio_mutation_effect_prediction/mock.yaml
```

The mock is deterministic, CPU-only, service-free, and explicitly
non-benchmark. The official seed is qualified separately. After filling the
three external paths in a protected runtime copy of `real.yaml`, verify and run
the locked campaign with:

```bash
python scripts/validate_tasks.py --task ai4bio_mutation_effect_prediction \
  --require-qualified
uv run --locked --project tasks/ai4bio_mutation_effect_prediction python \
  scripts/check_task_dependencies.py \
  config/ai4bio_mutation_effect_prediction/real.yaml --no-optional
uv run --locked --project tasks/ai4bio_mutation_effect_prediction python \
  scripts/run_ldm_tts.py config/ai4bio_mutation_effect_prediction/real.yaml
```
