# Antibody Clean-Room Quick Start

This guide reproduces the validated CPU-only antibody path from a new checkout.
It builds the locked Python 3.9 environment, runs the AntBO mock, checks an
OpenAI-compatible model, and performs one real `1ADQ_A` CDRH3 proposal and one
Absolut evaluation.

Run every command from the repository root. Absolut is an external dependency
and is not bundled with this repository.

## Prerequisites

- `uv` is installed.
- An Absolut installation containing an executable `src/bin/Absolut` exists.
- An OpenAI-compatible model URL, model name, and API key are available.
- The repository contains `tasks/antibody/uv.lock`.

## 1. Verify The Checkout

```bash
test -f tasks/antibody/uv.lock
python scripts/validate_tasks.py --task antibody
```

Keep credentials outside the checkout. Do not copy `api_credential.json` into
the repository.

## 2. Build The Locked Environment

```bash
uv sync --locked --project tasks/antibody
CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/antibody \
  python -c 'import sys, torch; print(sys.version); print("cuda_available=", torch.cuda.is_available())'
```

The validated build used CPython 3.9.25. The CUDA check must print `False` for
this guide.

## 3. Run The Mock Workflow

```bash
CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/antibody \
  python scripts/check_task_dependencies.py \
  config/antibody/mock_ei.yaml --no-optional

CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/antibody \
  python scripts/run_ldm_tts.py config/antibody/mock_ei.yaml --dry-run

CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/antibody \
  python scripts/run_ldm_tts.py config/antibody/mock_ei.yaml
```

The mock must finish without a model endpoint or Absolut. Its fourth evaluation
exercises the post-warmup EI acquisition path.

## 4. Configure Absolut And The Model

```bash
export CUDA_VISIBLE_DEVICES=''
export ABSOLUT_PATH=/path/to/Absolut
export LLM_BASE_URL=https://your-model-host.example/v1
export LLM_API_KEY=your-api-key
export LLM_MODEL_NAME=your-served-model

test -x "$ABSOLUT_PATH/src/bin/Absolut"
```

When credentials are supplied as a protected JSON file with `url`, `key`, and
`model` fields, load it without copying it into the checkout:

```bash
export LDM_CREDENTIAL_FILE=/secure/path/api_credential.json
export LLM_BASE_URL="$(jq -r .url "$LDM_CREDENTIAL_FILE")"
export LLM_API_KEY="$(jq -r .key "$LDM_CREDENTIAL_FILE")"
export LLM_MODEL_NAME="$(jq -r .model "$LDM_CREDENTIAL_FILE")"
```

## 5. Probe The Model API

```bash
CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/antibody python - <<'PY'
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["LLM_BASE_URL"],
    api_key=os.environ["LLM_API_KEY"],
    timeout=60,
)
models = client.models.list().data
print("configured_model_listed=", any(
    model.id == os.environ["LLM_MODEL_NAME"] for model in models
))
reply = client.chat.completions.create(
    model=os.environ["LLM_MODEL_NAME"],
    messages=[{"role": "user", "content": "Reply with exactly OK"}],
    max_tokens=8,
)
print("chat_reply=", reply.choices[0].message.content)
PY
```

Do not start Absolut work until both API operations succeed.

## 6. Run The Exact Real Preflight

```bash
CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/antibody \
  python scripts/check_task_dependencies.py \
  config/antibody/real_cpu_smoke.yaml
```

The preflight must report the LLM URL, model, masked API key, CPU device, and
`$ABSOLUT_PATH/src/bin/Absolut` as ready. Resolve every `FAIL` first.

## 7. Run The Task Contract Dry-Run

This enters the antibody adapter and emits its resolved LDM task contract
without calling the model or Absolut:

```bash
CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/antibody \
  python scripts/run_ldm_tts.py config/antibody/real_cpu_smoke.yaml \
  --set args.dry-run=true \
  --set args.out-dir=runs/antbo_first_real_contract
```

Confirm `device` is `cpu`, the antigen is `1ADQ_A`, the budget is one, and no
credential value appears in the output.

## 8. Run One Real Proposal And Evaluation

Run at most one heavy antibody smoke at a time:

```bash
CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/antibody \
  python scripts/run_ldm_tts.py config/antibody/real_cpu_smoke.yaml \
  --set args.out-dir=runs/antbo_first_real_tiny
```

This configuration uses one model proposal, one 11-residue CDRH3 evaluation,
and no random fallback. The validated clean-room trial selected `DGIGVDGAQRY`
for `1ADQ_A`; Absolut returned `-74.7000`. Model output can vary, so use this as
a reference rather than an expected fixed result. Lower energy is better.

## 9. Inspect Results

```bash
rg --files tasks/antibody/runs/antbo_first_real_tiny
```

Inspect the newest run directory beneath that path:

- `results.csv` contains the evaluated sequence, energy, and source.
- `llm_acq_decisions.jsonl` contains the parsed proposal and fallback status.
- `llm_antigen_context.json` confirms context came from Absolut.
- `ldm_parallel_decisions.json` records post-warmup acquisition decisions.
- `config.json` contains the task contract and non-secret run configuration.

For this one-evaluation smoke, `budget=1` and `n_init=1`, so the real run checks
the LLM warmup and Absolut path. The mock in step 3 checks post-warmup EI.

## 10. Run The Test Suite

```bash
env -u LLM_BASE_URL -u LLM_API_KEY -u LLM_MODEL_NAME -u LLM_MODEL \
  -u ABSOLUT_PATH CUDA_VISIBLE_DEVICES='' \
  uv run --locked --project tasks/antibody \
  pytest -q tests tasks/antibody/tests
```

The validated clean-room suite reported `352 passed`.

## 11. Clear Runtime Secrets

```bash
unset LLM_BASE_URL LLM_API_KEY LLM_MODEL_NAME LLM_MODEL
unset LDM_CREDENTIAL_FILE ABSOLUT_PATH CUDA_VISIBLE_DEVICES
```

Verify the credential file was not copied into the checkout:

```bash
test ! -e api_credential.json
```
