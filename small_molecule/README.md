# Small-Molecule Task Guide

The small-molecule task proposes SMILES strings, evaluates docking and activity
objectives, and uses direct LLM ordering or BO/EHVI-style acquisition to select
candidates. Mock runs use deterministic local scorers. Real runs can use
AutoDock Vina, the G12D activity model, and ReaSyn analog generation.

## Quick Start

From the repository root:

```bash
uv sync --project small_molecule
uv run --project small_molecule python scripts/run_ldm_tts.py config/small_molecule/mock_m1_stratified_oversample.yaml
```

Mock runs do not need Vina, ReaSyn, the G12D model, or a real LLM endpoint.

## Environment

For real runs, configure the LLM endpoint and task-specific external paths:

```bash
export CUDA_VISIBLE_DEVICES=0
export LLM_BASE_URL=http://127.0.0.1:52308/v1
export LLM_API_KEY=EMPTY
export LLM_MODEL_NAME=Qwen3.5-9B

export VINA_BIN=/path/to/vina
export G12D=small_molecule/activity_modeling/best_g12d_model.joblib
export REASYN_REPO=/path/to/ReaSyn
export REASYN_PYTHON=/path/to/ReaSyn/.venv/bin/python
```

The real config mirrors these values into explicit `args` so dry-runs show the
complete command:

```yaml
env:
  VINA_BIN: /path/to/vina
  G12D: small_molecule/activity_modeling/best_g12d_model.joblib
  REASYN_REPO: /path/to/ReaSyn
  REASYN_PYTHON: /path/to/ReaSyn/.venv/bin/python
args:
  llm-url: ${LLM_BASE_URL}
  api-key: ${LLM_API_KEY}
  llm-model-name: ${LLM_MODEL_NAME}
  vina-bin: ${VINA_BIN}
  nn-model-path: ${G12D}
  reasyn-repo: ${REASYN_REPO}
  reasyn-python: ${REASYN_PYTHON}
```

## Dependencies

| Dependency | What It Is | Required For | Configure With |
| --- | --- | --- | --- |
| Python environment | RDKit, Meeko, Gemmi, Gauche/gpytorch, sklearn/LightGBM, OpenAI client, plotting, and data libraries. | Molecule parsing, receptor/ligand prep, GP models, NN scoring, LLM calls. | `uv sync --project small_molecule`; run with `uv run --project small_molecule ...`. |
| AutoDock Vina | External docking executable. Produces the Vina objective; lower is better. | Real Vina scoring. | `args.vina-bin`, `env.VINA_BIN`, or `vina` on `PATH`. |
| G12D activity model | Model artifact at `small_molecule/activity_modeling/best_g12d_model.joblib`. Produces the activity objective; higher is better. | Real activity scoring. | `args.nn-model-path`, often through `env.G12D`. |
| ReaSyn | External reaction/synthesis-aware analog generator checkout plus AR and Edit Bridge checkpoints. | Seed-analog proposal methods. | `args.reasyn-repo`, `env.REASYN_HOME`, `env.REASYN_REPO`; interpreter through `args.reasyn-python`, `env.REASYN_PYTHON`, or `env.REASYN_BIN`. |
| LLM endpoint | OpenAI-compatible chat endpoint. | Real LLM proposals. | `LLM_BASE_URL` / `LLM_API_KEY`, plus matching `llm-url` / `api-key` args. |

## Dependency Check

Run the preflight before a real experiment:

```bash
python scripts/check_task_dependencies.py config/small_molecule/real_m1_seed_analog.yaml
```

If a direct-only method will not call ReaSyn but your config still contains
ReaSyn path placeholders, skip optional checks:

```bash
python scripts/check_task_dependencies.py config/small_molecule/real_m1_seed_analog.yaml --no-optional
```

The checker validates:

- LLM URL, model name, and API key setup
- requested CUDA visibility
- AutoDock Vina executable and `--help` response
- G12D activity model artifact
- ReaSyn checkout, interpreter, checkpoints, and CUDA devices when configured

The small-molecule real workflow also performs an early runtime preflight for
Vina and the activity model before starting the search.

## Vina

Check Vina manually:

```bash
$VINA_BIN --help
```

Vina caches receptor preparation and docking results under
`<trajectory-dir>/vina_cache` by default. To reuse an old cache, set
`args.vina-cache-dir` to that directory. If a round has candidates but no
selection, inspect `selection_results.failed_evaluations` in `rounds.jsonl`.

Important Vina fields:

| Field | Meaning |
| --- | --- |
| `args.vina-bin` | Vina executable path. |
| `args.vina-pdb-id` | Receptor PDB id; default is `8UN5`. |
| `args.vina-chain-id` | Receptor chain id. |
| `args.vina-cache-dir` | Optional existing docking cache to reuse. |
| `args.vina-exhaustiveness` | Vina exhaustiveness. |
| `args.vina-n-poses` | Number of poses requested. |

## ReaSyn

ReaSyn must expose `reasyn/sampler/parallel.py` under the checkout. The
selected interpreter must be able to import ReaSyn and its chemistry
dependencies:

```bash
$REASYN_PYTHON -c "import sys, os; sys.path.insert(0, os.environ['REASYN_REPO']); from reasyn.chem.mol import Molecule; print(Molecule('CCO').is_valid)"
```

Default ReaSyn checkpoints, relative to the ReaSyn checkout:

```text
data/trained_model/nv-reasyn-ar-166m-v2.ckpt
data/trained_model/nv-reasyn-eb-174m-v2.ckpt
```

Override them with `args.reasyn-model-path` as a comma-separated pair. Use
`args.reasyn-devices` for CUDA device IDs and `args.reasyn-time-limit` to bound
each ReaSyn call.

Interpreter resolution order:

1. `args.reasyn-python`
2. `env.REASYN_PYTHON`
3. `env.REASYN_BIN`
4. `<REASYN_REPO>/.venv/bin/python`
5. the current `small_molecule/.venv` interpreter

If ReaSyn raises an import error such as `No module named 'omegaconf'`, install
or sync that dependency in the interpreter selected by the list above.

## Real Runs

Starter config:

```bash
uv run --project small_molecule python scripts/run_ldm_tts.py config/small_molecule/real_m1_seed_analog.yaml
```

Use `--dry-run` before changing deployment paths:

```bash
python scripts/run_ldm_tts.py config/small_molecule/real_m1_seed_analog.yaml --dry-run
```

## Plotting

Plot a completed run:

```bash
uv run --project small_molecule python small_molecule/plots/plot_pareto_hv.py small_molecule/ldm_runs/case2_mock_m1
```

The plotter writes `pareto_hv.png`, `pareto_hv.pdf`,
`pareto_hv_hypervolume.csv`, and `pareto_hv_summary.csv` under the run's
`plots/` directory.
