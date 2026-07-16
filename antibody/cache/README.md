# cache/

Runtime cache for **pre-computed initial datasets** used by
`bo/custom_init.py` (the `--custom_init` feature). Entire directory is
**git-ignored** — regenerate by downloading the zip (see below).

## Contents

| Path | Origin |
|---|---|
| `init_dataset.zip` | Downloaded manually from the original AntBO repository (see below). |
| `init_dataset/` | Auto-extracted from `init_dataset.zip` on first import of `bo.custom_init`. Contains ~8,730 `init_data.pkl` files spanning 140 antigens × multiple seeds × multiple category combinations. |

## Bootstrap behaviour

`bo/custom_init.py` resolves the dataset in this order on every import:

1. If `./cache/init_dataset/` is a directory → use it directly.
2. Else if `./cache/init_dataset.zip` is a file → extract into `./cache/`.
3. Else → raise `FileNotFoundError` with instructions to download the zip.

The `_ensure_init_dataset()` helper handles this transparently. Verified by
`scripts/smoke/run_custom_init_smoke.py`.

## Downloading the zip

The `init_dataset.zip` (~15 MB) is not tracked by git. To obtain it:

1. Clone the original AntBO repository:
   ```bash
   git clone https://github.com/.../AntBO.git /tmp/antbo-upstream
   ```
2. Copy its `bo/init_dataset.zip` (or wherever it lives upstream) into this
   repo's `cache/`:
   ```bash
   cp /tmp/antbo-upstream/bo/init_dataset.zip cache/init_dataset.zip
   ```
3. Verify the bootstrap:
   ```bash
   python -c "from bo.custom_init import INIT_DATA_PATH; import os; print(INIT_DATA_PATH, os.path.isdir(INIT_DATA_PATH))"
   ```

Or, if you already have the extracted `init_dataset/` tree (e.g. from an older
checkout that had it under `bo/init_dataset/`), just place it directly under
`cache/init_dataset/` and the bootstrap is a no-op.

## Custom-init feature

The `custom_init` feature is **disabled by default** in `bo/config.yaml`. To
enable it, set:

```yaml
custom_init: true
custom_init_seed: 42
custom_init_n_loosers: 6
custom_init_n_mascottes: 6
custom_init_n_heroes: 8
custom_init_top_cut_loosers: 0.5
custom_init_top_cut_mascottes: 0.5
custom_init_top_cut_heroes: 0.5
```

Without `cache/init_dataset/` (or zip), enabling `custom_init` will fail at
`BOExperiments.__init__` with the same `FileNotFoundError`.