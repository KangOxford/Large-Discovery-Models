# scripts/smoke

Smoke tests that exercise the repository after reorganisation, without invoking
the external Absolut! binary.

## Files

| File | Purpose |
|---|---|
| `smoke_config.yaml` | Minimal `bo/main.py` config (`bbox.tool=random`, all LDM flags off, `max_iters=3`, `n_init=5`). |
| `custom_init_smoke_config.yaml` | Same shape, used to validate that `bo.custom_init` does not require `bbox.tool=Absolut`. |
| `run_bo_smoke.py` | Static import check + runs `bo/main.py` with the smoke config for 1 trial. Asserts a `results.csv` is produced. |
| `run_custom_init_smoke.py` | Three-scenario test for `./cache/init_dataset` bootstrap (extracted dir / zip only / missing). |
| `README.md` | This file. |

## Usage

From the repository root (`AntBO/`):

```bash
# 1. Validate the import chain + a one-trial BO run.
python scripts/smoke/run_bo_smoke.py

# 2. Validate ./cache/init_dataset bootstrap behaviour.
python scripts/smoke/run_custom_init_smoke.py
```

Both should print `... OK` / `ALL ... TESTS PASSED` and exit with status 0.

## Notes

- The smoke tests use `bbox.tool=random` (provided by `task/tools.py:RandomBlackBox`)
  which returns `np.random.random()` per query. This means no Absolut! binary,
  no GPU, no antigen data is required — only `numpy`, `pandas`, `torch`, etc.
- `run_bo_smoke.py` writes its outputs under `/tmp/antbo_smoke_outputs/` and
  cleans it up afterwards. Re-runs are idempotent.
- `run_custom_init_smoke.py` mutates `./cache/` temporarily; it restores the
  original state in `finally` blocks. Aborting the script mid-run is safe but
  may leave `./cache/` in a transient state — re-run the script to recover.