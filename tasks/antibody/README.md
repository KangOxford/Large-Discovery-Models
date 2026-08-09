# Antibody Task Guide

The antibody task wraps an AntBO-style CDRH3 optimization loop. It proposes
fixed-length amino-acid sequences, evaluates binding energy, and lets the LLM
update DSL trust-region or bias atoms during search.

For a new installation, follow the numbered
[clean-room quick start](QUICKSTART.md) before using this reference guide.

## Quick Start

From the repository root:

```bash
uv sync --locked --project tasks/antibody
CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/antibody \
  python scripts/run_ldm_tts.py config/antibody/mock_ei.yaml
```

Mock runs do not need Absolut or a real LLM endpoint.

## Environment

For the first real CPU-only run, configure the external Absolut installation
and OpenAI-compatible model through environment variables:

```bash
export CUDA_VISIBLE_DEVICES=''
export ABSOLUT_PATH=/path/to/Absolut
export LLM_BASE_URL=https://your-model-host.example/v1
export LLM_API_KEY=your-api-key
export LLM_MODEL_NAME=your-served-model
```

`LLM_BASE_URL` is the OpenAI-compatible API root ending in `/v1`, not the full
`/chat/completions` route. `EMPTY` is suitable only for a local server that does
not validate credentials; authenticated endpoints require their real key.

The antibody adapter maps these variables into the environment expected by the
underlying AntBO LDM loop. Keep credentials in the environment rather than in
YAML or command-line arguments so run plans and process listings do not expose
them.

## Dependencies

| Dependency | What It Is | Required For | Configure With |
| --- | --- | --- | --- |
| Python environment | Pinned AntBO-era BoTorch/gpytorch/torch/sklearn stack plus OpenAI client. | GP acquisition, sequence search, and LLM policy updates. | `uv sync --locked --project tasks/antibody`; run with `uv run --locked --project tasks/antibody ...`. |
| Absolut | External antibody-antigen binding-energy evaluator. Lower energy is better. It is not bundled with this repository. | Real objective evaluations. | `ABSOLUT_PATH`, `--absolut-path`, or `bbox.path` in a private AntBO config. |
| Antigen inputs | Antigen name or antigen list. | Real and mock task selection. | `--antigen`, or `--antigens-file` such as `test_5_antigens.txt`. |
| LLM endpoint | OpenAI-compatible chat endpoint for DSL updates. | Real LDM loops. | `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL_NAME`. |

## Dependency Check

Run the preflight before a real experiment:

```bash
CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/antibody \
  python scripts/check_task_dependencies.py config/antibody/real_cpu_smoke.yaml
```

The checker validates:

- LLM URL, model name, and API key setup
- antigen input or antigens file
- `tasks/antibody/bo/config.yaml`
- requested CUDA visibility
- Absolut path for real runs

## Absolut And AntBO Config

The default AntBO config is `tasks/antibody/bo/config.yaml`. Its `bbox.path` is
intentionally unset because Absolut is an external installation. Prefer an
environment variable for deployment-specific paths:

```bash
export ABSOLUT_PATH=/path/to/Absolut
```

You can instead use a per-run argument or a private task config:

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

Experiment YAML selects the shared posterior acquisition with `args.acq`:
`lcb`, `ucb`, `ei`, or `mean`. Configure confidence-bound exploration with
`args.acq-beta` and the EI improvement margin with `args.acq-xi`.

The procedure accepts `--device cpu` and `--absolut-path /path/to/Absolut`,
which override the task config for one run without changing shared files.

## Antigen Inputs

Use a single antigen:

```bash
CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/antibody \
  python scripts/run_ldm_tts.py config/antibody/mock_ei.yaml \
  --set args.antigen=SMOKE_ANTIGEN
```

Or use an antigen list, as in `config/antibody/real_lcb.yaml`:

```yaml
args:
  antigens-file: test_5_antigens.txt
```

Paths are resolved from the antibody task working directory unless they are
repository-root-relative or absolute.

## Minimal First Real Run

De-risk a new antibody deployment in four stages before starting the full real
configuration below.

1. Verify that the configured model is reachable and accepts Chat Completions.
   The OpenAI client reads the key from the environment, keeping it out of the
   command line:

   ```bash
   uv run --locked --project tasks/antibody python - <<'PY'
   import os
   from openai import OpenAI

   client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
   print([model.id for model in client.models.list().data])
   reply = client.chat.completions.create(
       model=os.environ["LLM_MODEL_NAME"],
       messages=[{"role": "user", "content": "Reply with OK"}],
       max_tokens=8,
   )
   print(reply.choices[0].message.content)
   PY
   ```

2. Check the exact CPU smoke configuration. This validates the LLM settings,
   antigen input, AntBO config, requested device, and Absolut installation:

   ```bash
   CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/antibody \
     python scripts/check_task_dependencies.py config/antibody/real_cpu_smoke.yaml
   ```

3. Run the real adapter's contract dry-run. This enters the task adapter,
   resolves the config, and emits the LDM task specification without calling
   the LLM or Absolut:

   ```bash
   CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/antibody \
     python scripts/run_ldm_tts.py config/antibody/real_cpu_smoke.yaml \
     --set args.dry-run=true \
     --set args.out-dir=ldm_runs/antbo_first_real_contract
   ```

4. Run one real LLM proposal and one Absolut evaluation:

   ```bash
   CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/antibody \
     python scripts/run_ldm_tts.py config/antibody/real_cpu_smoke.yaml \
     --set args.out-dir=ldm_runs/antbo_first_real_tiny
   ```

The tiny run is not a mock: it contacts the configured model and evaluates the
selected CDRH3 sequence with Absolut. Increase `budget`, `n-init`, and
`parallel-budget` only after this path succeeds.

For a repeatable CPU-only version of the same check, use the included smoke
config after exporting `ABSOLUT_PATH` and the three `LLM_*` variables:

```bash
CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/antibody \
  python scripts/run_ldm_tts.py \
  config/antibody/real_cpu_smoke.yaml
```

## Real Runs

Starter config:

```bash
uv run --locked --project tasks/antibody \
  python scripts/run_ldm_tts.py config/antibody/real_lcb.yaml
```

Use `--dry-run` before changing deployment paths:

```bash
python scripts/run_ldm_tts.py config/antibody/real_lcb.yaml --dry-run
```

## Legacy Environment Notes

`tasks/antibody/environment.yaml` is retained as the legacy conda environment for the
original pinned CUDA/PyTorch/DGL AntBO stack. Prefer `uv` for the unified
LDM-TTS runner unless you need to reproduce the exact legacy environment.

`tasks/antibody/cache/init_dataset/` and `tasks/antibody/cache/init_dataset.zip` are legacy
AntBO custom-init data. They can exist locally if you need that old path, but
they are not tracked by this LDM-TTS-focused repository.
