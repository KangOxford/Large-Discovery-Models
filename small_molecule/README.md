# PDF2Dock

Extract Markush compound information from research PDFs, generate a seed CSV
with SMILES and literature activity values, expand each seed with ReaSyn analog
generation, and rank analog groups with AutoDock Vina. An optional StrBO loop
can tune ReaSyn inference parameters using Vina feedback.

## Workflow Overview

```text
PDF / Supporting Information
  -> compound scope
  -> formula + activity extraction
  -> SMILES lookup
  -> review CSV
  -> ReaSyn analog generation per seed
  -> AutoDock Vina scoring per analog group
  -> optional StrBO loop over ReaSyn inference parameters
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

AutoDock Vina must be available in `PATH`, or specified with `VINA_BIN`:

```bash
export VINA_BIN=/path/to/vina
```

For docking on macOS, the conda environment is usually more reliable for
RDKit/Meeko/Vina:

```bash
conda env create -f environment_docking.yml
conda activate markush-dock
```

## Configure

Put your OpenAI-compatible API configuration in `.env` or environment variables:

```bash
export OPENROUTER_API_KEY=your_api_key
export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Alternatively, configure model, PDFs, output directory, and cache settings in
`markush_config.json`.

## Run Workflow

### Step 1: Extract SMILES and Activity

```bash
python3 pdf2markush_workflow.py \
  --config markush_config.json \
  --pdf PDFs/paper.pdf \
  --pdf PDFs/supporting_information.pdf \
  --outdir output/formula_workflow_outputs \
  --final-csv final_compounds.csv
```

Key outputs:

- `formula_to_smiles.csv`: full review table.
- `final_compounds.csv` or an automatically named final CSV: reviewed seed table for ReaSyn analog generation.
- `activity_records.json`: extracted activity records.
- `compound_scope.json`: final compound scope.

### Step 2: Human Review

Before analog generation, manually check columns such as `SMILES`,
`Activity_nM`, and `Note` in the CSV. The reviewed CSV is the seed table for
ReaSyn.

### Step 3: Generate ReaSyn Analogs

`docking_to_analog_search_example.py` takes the reviewed PDF-extraction seed
CSV, picks seed molecules by literature activity when available, and writes a
ReaSyn-ready SMILES text file plus a reproducible command script. ReaSyn
consumes seed molecules directly as SMILES and generates synthesizable
analog/pathway candidates in inference mode.

```bash
python3 docking_to_analog_search_example.py \
  --top-n 3 \
  --output-dir output/reasyn_demo_prepared
```

If `--smiles-path` is omitted, the script defaults to
`examples/reasyn_demo/autodock_markush_demo_docking_results.csv`. That file has
a historical name, but its contents are a small extraction-output seed table
with SMILES and literature activity values, so the demo skips the PDF extraction
step only.

Required ReaSyn assets under `/path/to/ReaSyn`:

- ReaSyn AR checkpoint:
  download `nv-reasyn-ar-166m-v2.ckpt` from
  `https://huggingface.co/nvidia/NV-ReaSyn-AR-166M-v2/resolve/main/nv-reasyn-ar-166m-v2.ckpt`
  and place it at `data/trained_model/nv-reasyn-ar-166m-v2.ckpt`.
- ReaSyn Edit Bridge checkpoint:
  download `nv-reasyn-eb-174m-v2.ckpt` from
  `https://huggingface.co/nvidia/NV-ReaSyn-EB-174M-v2/resolve/main/nv-reasyn-eb-174m-v2.ckpt`
  and place it at `data/trained_model/nv-reasyn-eb-174m-v2.ckpt`.
- ReaSyn building-block input:
  place the Enamine US Stock building-block catalog at
  `data/building_blocks/building_blocks.txt`.
- ReaSyn reaction-template input:
  place the reaction template file required by ReaSyn at
  `data/rxn_templates/comprehensive.txt`.

Generate ReaSyn's processed indexes from the ReaSyn checkout:

```bash
cd /path/to/ReaSyn
python scripts/preprocess.py --model-config configs/train_ar.yml
```

That preprocessing step should create:

- `data/processed/comp_2048/fpindex.pkl`
- `data/processed/comp_2048/matrix.pkl`

Run ReaSyn directly from the bridge:

```bash
python3 docking_to_analog_search_example.py \
  --smiles-path output/formula_workflow_outputs/final_compounds.csv \
  --top-n 20 \
  --reasyn-repo /path/to/ReaSyn \
  --model-paths data/trained_model/nv-reasyn-ar-166m-v2.ckpt,data/trained_model/nv-reasyn-eb-174m-v2.ckpt \
  --run-reasyn \
  --output-dir output/reasyn_run
```

### Step 4: Dock and Rank ReaSyn Analogs

After ReaSyn writes `reasyn_analogs.csv`, dock the generated analogs. The dock
command recognizes ReaSyn columns such as `target`, `smiles`, `score`,
`synthesis`, and `num_steps`, groups analogs by their seed, ranks each group by
Vina score, and keeps the top K analogs per seed.

```bash
python3 extract_and_dock.py dock \
  --csv output/reasyn_run/reasyn_analogs.csv \
  --allow-unreviewed \
  --pdb-id 8UN5 \
  --chain-id A \
  --work-dir output/docking_work \
  --output-dir output/docking_work/reasyn_analog_results \
  --exhaustiveness 4 \
  --num-modes 3 \
  --analog-top-k 3
```

Docking results are written to:

- `docking_results.csv`: all analog docking attempts.
- `docking_activity_joint_score.csv`: all analogs with Vina and any carried activity metadata.
- `analog_group_topk.csv`: top K Vina-ranked analogs for each seed molecule.
- `analog_overall_best.csv`: the single best analog among the retained top-K rows.
- `docking_metadata.json`: receptor, parameters, output paths, and `analog_summary.group_best` for future black-box optimization.

The command also prints the current best analog for each seed to stdout. No
separate per-seed best CSV is written; use `analog_group_topk.csv` rank 1 rows
or `docking_metadata.json` when that interface is needed programmatically.

### Step 5: Optimize ReaSyn Parameters with StrBO

`reasyn_strbo_optimization.py` runs a sequential black-box loop. Each StrBO
trial samples ReaSyn inference parameters, generates analogs from the current
active seed pool, docks the generated analogs, and returns the best Vina score
from that trial as the objective. Vina kcal/mol is minimized because more
negative scores are better. StrBO encodes each parameter config as a string,
fits a lightweight exact Gaussian process with a normalized character n-gram
kernel inspired by GAUCHE's bag-of-SMILES examples, and scores generated
candidates with an EI/LCB acquisition by default.

After each trial, the next active seed pool is the previous trial's top K analogs
per initial literature seed. By default `--top-k 3`, and the cumulative archive
for each initial literature seed is capped at `2 * k = 6` candidates to avoid
unbounded analog growth.

```bash
python3 reasyn_strbo_optimization.py \
  --smiles-path output/formula_workflow_outputs/final_compounds.csv \
  --top-n-seeds 20 \
  --n-trials 10 \
  --top-k 3 \
  --reasyn-repo /path/to/ReaSyn \
  --model-paths data/trained_model/nv-reasyn-ar-166m-v2.ckpt,data/trained_model/nv-reasyn-eb-174m-v2.ckpt \
  --pdb-id 8UN5 \
  --chain-id A \
  --work-dir output/docking_work \
  --output-dir output/reasyn_strbo
```

Useful outputs:

- `trial_0000/reasyn_analogs_annotated.csv`: ReaSyn analogs tagged with their initial seed group.
- `trial_0000/analog_group_topk.csv`: trial top K analogs per initial seed.
- `trial_0000/trial_summary.json`: sampled ReaSyn parameters, objective value, and capped candidate archive.
- `strbo_summary.json`: best StrBO trial, best parameters, all trial values, acquisition settings, and final capped archive.

The default search space covers `search_width`, `exhaustiveness`, `num_cycles`,
`num_editflow_samples`, `num_editflow_steps`, `filter_sim`, and
`no_exact_break`. Override ranges with flags such as `--search-width-range 4 24`
or `--filter-sim-range 0.6 0.95`. StrBO-specific controls include
`--strbo-initial-random`, `--strbo-candidate-pool-size`,
`--strbo-acquisition`, and `--strbo-kernel-max-ngram`.

### strbo_v1 Package

`strbo_v1/` is a redesigned Bayesian-optimization package for SMILES-driven
analog search. It runs alongside the legacy `strbo/` StrBO module and is the
recommended interface for new work. It provides:

- `VinaScorer` — a **callable** Vina invocation with disk cache and parallel
  workers, defined in `strbo_v1.objective_vina`. `scorer(smiles_list) -> list[float]`
  returns one Vina score per SMILES (or `float("nan")` on any docking
  failure, which the BO loop silently drops from the GP fit). The
  `vina_bin` path is fixed at `VinaScorer.__init__` via
  `VinaScorerConfig.vina_bin`.
- `NNScorer` — a callable wrapper around a trained regression model
  loaded from a joblib file (default:
  `activity_modeling/best_model.joblib`, an `ensemble_nn_ridge_lgbm`
  averaging Tanimoto-KNN, char-tfidf Ridge, and Morgan+LightGBM,
  trained on ChEMBL KRAS G12C IC50). Defined in
  `strbo_v1.objective_nn`. Same `scorer(smiles_list) -> list[float]`
  contract as `VinaScorer`. Output is predicted pIC50 (higher = more
  potent); invalid SMILES yield `float("nan")`. Inference failures are
  configurable via `NNScorerConfig.on_error` (`"all_nan"` default keeps
  the BO loop running; `"raise"` propagates for debugging). The
  pickle-module shim at import time registers
  `activity_modeling.train_g12c_qsar` under the
  `train_g12c_qsar` alias used by the committed artifact.
- `Scorer` — the canonical `Callable[[Sequence[str]], Sequence[float]]`
  TypeAlias, defined in `strbo_v1.scorer` and re-exported by both
  `objective_vina` and `objective_nn`. Any callable matching this
  signature is accepted as a `scorer` argument.
- `GPSurrogate` — single GP class with two kernel implementations
  (`impl="fingerprint+tanimoto"` for Morgan-fingerprint Tanimoto;
  `impl="smiles-strkernel"` for SMILES subsequence string kernel). Uses a
  Cholesky-jitter ladder and falls back to **prior mode** on full failure so
  the BO loop never sees a `None` surrogate.
- `bayesian_analog_search` — warmup + init + BO loop with three acquisitions
  (EI, PI, UCB) all returning "higher = better" uniformly for the
  single-objective case. For **multi-objective** runs (`scorer` is a
  tuple), the loop dispatches by `n_obj`: `n_obj == 2` uses
  Expected Hypervolume Improvement (EHVI, Monte Carlo);
  `n_obj >= 3` uses Chebyshev ParEGO scalarization with a sampled
  simplex reference direction. The candidate pool is a `FIFOSet`
  (FIFO-ordered queue with O(1) membership); set `max_pool_size` to
  bound the queue with FIFO eviction of the oldest entry. `acq_budget`
  subsamples a large pool before the GP+acquisition step (the string
  kernel is much more expensive than Tanimoto, so this trades a bit
  of selection quality for tractable runtime). Analogue SMILES
  longer than `smiles_max_len` (default 100) are dropped at the
  pool-ingestion step (canonical-length check when
  `canonicalize_pool=True`, raw-text check otherwise).
- `random_analog_search` / `random_analog_search` baselines with lazy
  expansion (refill `pool_min_size` after each scoring batch) and an
  optional bounded `FIFOSet(max_size=pool_max_size)` (FIFO queue). Same
  `smiles_max_len` filter applies. The `expansion="best"` strategy
  uses Chebyshev ParEGO scalarization (single simplex weight sampled
  per refill) for any number of objectives.
- `strbo_v1.acquisition` — single-obj EI/PI/UCB + multi-obj
  `hypervolume` (exact 1D + 2D sweep-line; `NotImplementedError` for
  `n_obj >= 3`), `expected_hypervolume_improvement` (2D MC;
  `NotImplementedError` otherwise), and generic-N
  `chebyshev_scalarize` / `sample_simplex_weights`. The 2D MC EHVI
  uses `strbo_v1.rng.RNG` (unified python/numpy/torch under one
  seed) for reproducible sampling.
- `strbo_v1.scorer.DEFAULT_REF` — per-backend default reference point
  registry: `vina=0.0` (kcal/mol), `nn=5.0` (pIC50 baseline), `mock=0.0`.
  Override at runtime via `strbo_v1.scorer.register_ref(name, default)`;
  per-run override via `--ref-point X,Y`.

The `Scorer` (`Callable[[Sequence[str]], Sequence[float]]`) and
`analog_fn` (`Callable[[Sequence[str]], Sequence[str]]`) callable
contracts are framework-agnostic — the search loops never see a
DataFrame. The canonical analog implementation is
`strbo_v1.analog.generate_analogs`, which returns a `pandas.DataFrame`
with a `target` column (input SMILES) and a `smiles` column (analogue
SMILES). The `run_search.py::_build_reasyn_analog` adapter is a
closure that bakes in the `ReasynConfig` and flattens the DataFrame
to a plain `list[str]`; this is the single point that knows about the
DataFrame representation. The BO loop invokes the analog generator
**once per phase** (warm-up, init, BO round) with the full target
list; the random loop's per-target expansion invokes it with a
single-element list.

```python
from strbo_v1 import (
    BayesianAnalogSearchConfig, VinaScorer, VinaScorerConfig,
    bayesian_analog_search, random_analog_search,
)
from strbo_v1.analog import ReasynConfig, generate_analogs
from strbo_v1.gp import GPConfig

scorer = VinaScorer(VinaScorerConfig(vina_bin="../bin/vina", cache_dir="output/vina_cache/"))
# scorer is already callable: scorer(smis) -> list[float]

def analog_fn(smis):
    # Adapter: the search loops expect a flat ``Iterable[str] -> Iterable[str]``
    # contract. ``generate_analogs`` returns a DataFrame; this closure is
    # the single point that knows about that representation. The loops
    # never see a DataFrame.
    df = generate_analogs(smis, ReasynConfig(...))
    return df["smiles"].tolist() if df is not None and len(df) > 0 else []

history = bayesian_analog_search(
    seed_smiles=["CCO", "CCN"],
    scorer=scorer,
    analog_fn=analog_fn,
    config=BayesianAnalogSearchConfig(
        init_size=10, batch_size=1, n_iterations=20,
        acquisition="ei", acq_budget=2000,
        max_pool_size=500,  # FIFO cap; None = unbounded
        gp_config=GPConfig(impl="fingerprint+tanimoto", device="cuda"),
    ),
)
```

### Search Comparison Experiments

`run_search.sh` runs each (method, seed) trajectory as a separate
`run_search.py` invocation and writes
`<output_dir>/<method>_seed=<seed>.json` files. Each JSON has a `config`
echo (all knobs) and a `history` list of `{index, smiles, score}` entries.

```bash
bash run_search.sh                                       # real Vina + ReaSyn
python plot_search_results.py --input-dir output/bo     # aggregate + plot
```

The supported methods are:

- `random`, `random-best` — uniform / Chebyshev-ParEGO expansion.
- `bo-tanimoto`, `bo-strkernel` — standard Bayesian optimization
  (Tanimoto fingerprint GP / SMILES string-kernel GP).
- `bo-tanimoto-ldm`, `bo-strkernel-ldm` — same GP backends with a
  two-phase LLM advisor (`strbo_v1.bayesian_ldm_search`). The LDM
  variants insert a per-round Phase A (LLM mutates pool, reviews
  pending analogues) and Phase B (LLM reviews / overrides BO
  suggestions) call around the BO step. Requires `LLM_API_KEY` +
  `LLM_BASE_URL` in `.env`; the model is hardcoded to
  `DeepSeek-V4-Flash` (override per-run via `--llm-model`).

For `bo-*-ldm` runs, the main JSON also carries an `"llm_trajectory"`
key with the per-round LLM/BO log (Phase A blocks, BO suggestions,
Phase B decisions, scores). Choose which methods to run by editing
the `METHODS=()` list near the top of `run_search.sh`.

Per-method arguments (e.g. `--pool-min-size`, `--acq-budget`) are
inlined at the python invocation; shared knobs (Vina / ReaSyn / GP
config) live at the top of the script. Add `--objective mock` to each
`python run_search.py` line in the script for fast CPU-only smoke runs
(no Vina, no ReaSyn). Use `--objective nn` to score candidates with
`activity_modeling/best_model.joblib` (no Vina, no ReaSyn). Use
`--objective vina+nn` for multi-objective BO with EHVI (n_obj=2);
`--objective vina+nn+mock` (or any `+`-joined combo) for n_obj>=3
which uses Chebyshev ParEGO. The shared `SMILES_MAX_LEN` knob
(default 50) drives both the search-loop pool filter and the GP
string-kernel featurization cap.

`--ref-point X,Y` overrides the per-backend default reference point
(see `strbo_v1.scorer.DEFAULT_REF` for the registry). It is required
for length validation when `n_obj >= 2`; the default falls back to
the registry's per-backend values. Silently ignored for single-
objective runs. `--ehvi-n-samples` (default 128) controls the
Monte-Carlo accuracy of EHVI; `--che-alpha` (default 1.0) controls
the simplex-weight concentration in Chebyshev ParEGO.

`--seed-smiles` accepts either a comma-separated list (`"CCO,CCN,CCC"`)
or a path to an existing file (one SMILES per line, blank lines
filtered). All SMILES are RDKit-validated and auto-canonicalized; an
invalid entry raises `ValueError` with file + 1-based line number
(file mode) or 1-based position (comma mode). The same canonical
forms are then recorded in the per-trajectory JSON's `config.seed_smiles`
echo, so any individual trajectory can be re-run verbatim by passing
the same value back to `--seed-smiles`.

The BO methods have **separate** `acq_budget` settings
(`BO_ACQ_BUDGET_TAN=2000`, `BO_ACQ_BUDGET_STR=200`) because the SMILES
string-kernel GP is much more expensive than the Tanimoto-fingerprint GP.

`plot_search_results.py` reads all `*_seed=*.json` in the input
directory, groups by method, and writes `<output>.png` + `<output>.csv`.
The plot dispatch is per-method: `n_objectives == 1` produces a
best-so-far curve; `n_objectives == 2` produces a cumulative
hypervolume curve (w.r.t. the JSON's `ref_point` or `--ref-point` on
the CLI); `n_objectives >= 3` falls back to per-objective
best-so-far curves (HV is not implemented for n_obj>=3).

## Notes

- Cache is reused by default; add `--force` to re-run extraction.
- Use `--target-compound-id` and `--exclude-compound-id` to manually control the compound scope.
- Receptor preparation depends on Meeko.
