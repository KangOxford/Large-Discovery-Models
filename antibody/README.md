# Antibody Task Guide

The antibody task wraps an AntBO-style CDRH3 optimization loop. It proposes
fixed-length amino-acid sequences, evaluates binding energy, and lets the LLM
update DSL trust-region or bias atoms during search.

## Quick Start

From the repository root:

```bash
uv sync --project antibody
uv run --project antibody python scripts/run_ldm_tts.py config/antibody/mock_ei.yaml
```

Mock runs do not need Absolut or a real LLM endpoint.

## Environment

For real runs, configure the LLM endpoint and CUDA visibility:

```bash
export CUDA_VISIBLE_DEVICES=0
export LLM_BASE_URL=http://127.0.0.1:52313/v1
export LLM_API_KEY=EMPTY
export LLM_MODEL_NAME=Qwen3.5-9B
```

The antibody adapter maps CLI/config values into the environment expected by
the underlying AntBO LDM loop, including `LLM_BASE_URL`, `LLM_API_KEY`,
`LLM_MODEL`, `LLM_MAX_TOKENS`, and `LLM_DISABLE_THINKING`.

## Dependencies

| Dependency | What It Is | Required For | Configure With |
| --- | --- | --- | --- |
| Python environment | Pinned AntBO-era BoTorch/gpytorch/torch/sklearn stack plus OpenAI client. | GP acquisition, sequence search, and LLM policy updates. | `uv sync --project antibody`; run with `uv run --project antibody ...`. |
| Absolut | External antibody-antigen binding-energy evaluator. Lower energy is better. | Real objective evaluations. | `antibody/bo/config.yaml` under `bbox.path`, or a config passed with `--config`. |
| Antigen inputs | Antigen name or antigen list. | Real and mock task selection. | `--antigen`, or `--antigens-file` such as `test_5_antigens.txt`. |
| LLM endpoint | OpenAI-compatible chat endpoint for DSL updates. | Real LDM loops. | `LLM_BASE_URL` / `LLM_API_KEY`, or `llm-url` / `api-key` config args. |

## Dependency Check

Run the preflight before a real experiment:

```bash
python scripts/check_task_dependencies.py config/antibody/real_lcb.yaml
```

The checker validates:

- LLM URL, model name, and API key setup
- antigen input or antigens file
- `antibody/bo/config.yaml`
- requested CUDA visibility
- Absolut path for real runs

## Absolut And AntBO Config

The default AntBO config is `antibody/bo/config.yaml`. Update `bbox.path`
before real runs:

```yaml
bbox:
  tool: Absolut
  path: /path/to/Absolut
  process: 2
  startTask: 0
```

The same config also controls:

| Field | Meaning |
| --- | --- |
| `device` | GP/acquisition device; default is `cuda`. |
| `seq_len` | CDRH3 sequence length; default is `11`. |
| `n_init` | Initial evaluated sequence count. |
| `kernel_type` | AntBO GP kernel choice. |
| `llm` | LDM loop prompt, sampling, candidate, and search-budget settings. |
| `bbox` | Absolut evaluator settings. |

If CUDA is unavailable, adjust `device` before launching a real run.

## Antigen Inputs

Use a single antigen:

```bash
uv run --project antibody python scripts/run_ldm_tts.py config/antibody/mock_ei.yaml \
  --set args.antigen=SMOKE_ANTIGEN
```

Or use an antigen list, as in `config/antibody/real_lcb.yaml`:

```yaml
args:
  antigens-file: test_5_antigens.txt
```

Paths are resolved from the antibody task working directory unless they are
repository-root-relative or absolute.

## Real Runs

Starter config:

```bash
uv run --project antibody python scripts/run_ldm_tts.py config/antibody/real_lcb.yaml
```

Use `--dry-run` before changing deployment paths:

```bash
python scripts/run_ldm_tts.py config/antibody/real_lcb.yaml --dry-run
```

## Legacy Environment Notes

`antibody/environment.yaml` is retained as the legacy conda environment for the
original pinned CUDA/PyTorch/DGL AntBO stack. Prefer `uv` for the unified
LDM-TTS runner unless you need to reproduce the exact legacy environment.

`antibody/cache/init_dataset/` and `antibody/cache/init_dataset.zip` are legacy
AntBO custom-init data. They can exist locally if you need that old path, but
they are not tracked by this LDM-TTS-focused repository.
