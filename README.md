# LDM-TTS

This repository is trimmed around the Large Discovery Model test-time search
algorithm and three task adapters:

| Path | Purpose |
| --- | --- |
| `ldm_tts/` | Shared budgeted-search, trajectory, JSON-log, and score-ranking primitives. |
| `nanogpt/` | Code-edit LDM-TTS over a nanoGPT-style training script. |
| `small_molecule/` | Acquisition-tilted LDM-TTS for SMILES proposal, GP/EHVI selection, Vina scoring, and NN activity scoring. |
| `antibody/` | AntBO-style LDM-TTS for CDRH3 sequence proposal, GP acquisition, and Absolut evaluation. |
| `tests/` | Repository-level tests for the shared LDM-TTS core. |

Task-local README files and legacy experiment notes have intentionally been
removed. This root README is the canonical setup and run guide.

## Housekeeping Policy

Track only the LDM-TTS runtime, minimal task adapters, small configuration
files, focused tests, and required model artifacts. Generated runs, caches,
scratch files, notebooks, plots, PDFs, and historical experiment scripts stay
out of git.

Ignored generated paths include:

- `**/TTS/runs/`, `**/TTS/ablation_runs/`, and `**/TTS/ablation_buffer/`
- `antibody/cache/`, `antibody/outputs/`, and `antibody/temp/`
- `small_molecule/output/`
- Python caches, local virtual environments, and `.env` files

## Shared Core

Use the shared package for mechanics that should not be copied across tasks:

- `ldm_tts.loop.run_budgeted_search`
- `ldm_tts.trajectory.JsonlTrajectoryRecorder`
- `ldm_tts.trajectory.AtomicJsonLog`
- `ldm_tts.scoring`

Keep proposal generation, scoring, prompt construction, and environment
evaluation inside the task adapter.

## nanoGPT Task

Install the nanoGPT task environment:

```bash
cd nanogpt
uv sync
```

Optional full-training setup downloads/prepares data for `train.py`:

```bash
uv run prepare.py
```

Fast local smoke run with the mock objective:

```bash
python TTS/run_expanded_search.py \
  --train-file TTS/mock_train.py \
  --operation-schema TTS/operation_schema_mock_train.json \
  --generator operation_mock \
  --method best_of_n \
  --breadth 2 \
  --depth 1 \
  --iterations 2 \
  --warmup 1 \
  --out-dir TTS/runs/mock_smoke
```

Real code-search runs usually target `TTS/real_train.py` or `train.py`, use
`TTS/operation_schema_real_train.json`, and evaluate with:

```bash
--eval-command "uv run python {train_path}"
```

For OpenAI-compatible LLM endpoints, pass `--llm-url`, `--llm-model-name`, and
optionally `--api-key`.

## Small-Molecule Task

Install the molecule task environment:

```bash
cd small_molecule
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Fast mock run without external services:

```bash
python TTS/run_tilted_case2_tts.py \
  --mock \
  --method m1_stratified_direct_llm_oversample_sir \
  --budget 8 \
  --m1-k-direct-llm 16 \
  --trajectory-dir TTS/runs/case2_mock
```

Real runs need:

- `LLM_BASE_URL` and `LLM_API_KEY`, or `--llm-url` and `--api-key`
- AutoDock Vina in `PATH` or `--vina-bin /path/to/vina`
- ReaSyn checkout via `REASYN_HOME`, `REASYN_REPO`, or `--reasyn-repo`
- ReaSyn AR and EB checkpoints, by default under
  `data/trained_model/nv-reasyn-ar-166m-v2.ckpt` and
  `data/trained_model/nv-reasyn-eb-174m-v2.ckpt`
- The NN activity model artifact at
  `activity_modeling/best_g12d_model.joblib`

Example real run shape:

```bash
python TTS/run_tilted_case2_tts.py \
  --method m1_stratified_direct_llm_oversample_sir \
  --init-strategy llm_cold_start \
  --budget 80 \
  --m1-k-direct-llm 512 \
  --max-candidates-per-round 256 \
  --kernel sk \
  --gp-device cpu \
  --llm-url http://127.0.0.1:52307/v1 \
  --llm-model-name Qwen3-Coder-30B-A3B-Instruct \
  --vina-bin /path/to/vina \
  --trajectory-dir TTS/runs/case2_real
```

## Antibody Task

Install the antibody task environment:

```bash
cd antibody
conda env create -f environment.yaml
conda activate DGM
```

Fast mock run without Absolut or an LLM endpoint:

```bash
python TTS/example_run_antbo_tts.py \
  --mock \
  --antigen SMOKE_ANTIGEN \
  --budget 4 \
  --n-init 3 \
  --parallel-budget 8 \
  --out-dir TTS/runs/antbo_tts_mock
```

Real runs need:

- Absolut installed and configured in `bo/config.yaml` under `bbox.path`
- `LLM_BASE_URL` and `LLM_API_KEY`, or the equivalent CLI flags
- An antigen via `--antigen` or an antigen list via `--antigens-file`

Example real run shape:

```bash
python TTS/example_run_antbo_tts.py \
  --config bo/config.yaml \
  --antigen 1ADQ_A \
  --seed 42 \
  --budget 200 \
  --n-init 20 \
  --parallel-budget 600 \
  --llm-url http://127.0.0.1:52313/v1 \
  --llm-model-name Qwen3-Coder-30B-A3B-Instruct \
  --out-dir TTS/runs/antbo_tts_real
```

`antibody/cache/init_dataset/` and `antibody/cache/init_dataset.zip` are legacy
AntBO custom-init data. They can exist locally if you need that old path, but
they are not tracked by this LDM-TTS-focused repository.

## Quick Checks

Run the shared core test from the repository root:

```bash
python -m pytest tests/test_ldm_tts_core.py
```

Run focused task checks when changing task code:

```bash
cd small_molecule
python -m pytest tests/test_tilted_loop.py tests/test_tilted_methods_m1.py
```

```bash
cd antibody
python -m pytest tests/bo/ldm
```
