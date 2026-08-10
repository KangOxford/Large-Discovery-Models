# Small-Molecule Clean-Room Quick Start

This guide reproduces the validated CPU-only direct-LLM smoke path from a new
checkout. It builds the task environment, runs the mock workflow, provisions
AutoDock Vina, checks an OpenAI-compatible model, and evaluates one real
molecule with Vina and the bundled KRAS G12D activity model.

Run every command from the repository root. The first real run deliberately
does not use ReaSyn, so no GPU or ReaSyn checkout is required.

## Prerequisites

- `uv` is installed.
- Conda is available if Vina is not already installed.
- An OpenAI-compatible model URL, model name, and API key are available.
- The repository contains `tasks/small_molecule/uv.lock` and the G12D model
  artifact.

## 1. Verify The Checkout

```bash
test -f tasks/small_molecule/uv.lock
test -s tasks/small_molecule/resources/models/best_g12d_model.joblib
python scripts/validate_tasks.py --task small_molecule
```

Keep credentials outside the repository. In particular, do not copy an
`api_credential.json` file into this checkout.

## 2. Build The Locked Environment

```bash
uv sync --locked --project tasks/small_molecule
CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/small_molecule \
  python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
```

The CUDA check must print `False` for this CPU-only path.

## 3. Run The Mock Workflow

Check dependencies, inspect the resolved command, and then run the mock:

```bash
CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/small_molecule \
  python scripts/check_task_dependencies.py \
  config/small_molecule/mock_m1_stratified_oversample.yaml \
  --no-optional

CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/small_molecule \
  python scripts/run_ldm_tts.py \
  config/small_molecule/mock_m1_stratified_oversample.yaml \
  --dry-run

CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/small_molecule \
  python scripts/run_ldm_tts.py \
  config/small_molecule/mock_m1_stratified_oversample.yaml
```

The mock must finish without an LLM endpoint, Vina, or ReaSyn.

## 4. Install AutoDock Vina

Vina is an external executable. Create the repository-provided Conda
environment when it is not installed already:

```bash
conda env create -f tasks/small_molecule/environments/docking.yaml
export VINA_BIN="$(conda run -n markush-dock which vina | sed '/^[[:space:]]*$/d' | tail -n 1)"
test -x "$VINA_BIN"
"$VINA_BIN" --help
```

If `markush-dock` already exists, update it instead:

```bash
conda env update -n markush-dock \
  -f tasks/small_molecule/environments/docking.yaml --prune
```

## 5. Configure The Real Run

Use environment variables so credentials do not appear in YAML, dry-run
output, or process arguments:

```bash
export CUDA_VISIBLE_DEVICES=''
export G12D="$PWD/tasks/small_molecule/resources/models/best_g12d_model.joblib"
export LLM_BASE_URL=https://your-model-host.example/v1
export LLM_API_KEY=your-api-key
export LLM_MODEL_NAME=your-served-model
```

When credentials are supplied as a protected JSON file with `url`, `key`, and
`model` fields, load them without copying the file into the checkout:

```bash
export LDM_CREDENTIAL_FILE=/secure/path/api_credential.json
export LLM_BASE_URL="$(jq -r .url "$LDM_CREDENTIAL_FILE")"
export LLM_API_KEY="$(jq -r .key "$LDM_CREDENTIAL_FILE")"
export LLM_MODEL_NAME="$(jq -r .model "$LDM_CREDENTIAL_FILE")"
```

## 6. Probe The Model API

```bash
CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/small_molecule python - <<'PY'
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

Do not continue until model discovery and Chat Completions both succeed.

## 7. Run The Real Dependency Preflight

The first real run uses the direct-LLM method. `--no-optional` omits ReaSyn
checks while retaining the model, CPU, Vina, and G12D checks:

```bash
CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/small_molecule \
  python scripts/check_task_dependencies.py \
  config/small_molecule/real_m1_seed_analog.yaml \
  --set args.vina-bin="$VINA_BIN" \
  --set args.nn-model-path="$G12D" \
  --no-optional
```

Resolve every `FAIL` before running a real evaluation.

## 8. Run The Task Contract Dry-Run

This executes the task adapter's dry-run without contacting the model or
evaluators:

```bash
CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/small_molecule \
  python scripts/run_ldm_tts.py \
  config/small_molecule/real_m1_seed_analog.yaml \
  --set args.dry-run=true \
  --set args.budget=1 \
  --set args.init-size=1 \
  --set args.trajectory-dir=runs/first_real_contract
```

Confirm the resolved plan is CPU-only and contains no API key.

## 9. Evaluate One Real Molecule

```bash
CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/small_molecule \
  python scripts/run_ldm_tts.py \
  config/small_molecule/real_m1_seed_analog.yaml \
  --set args.budget=1 \
  --set args.init-size=1 \
  --set args.batch-size=1 \
  --set args.m1-k-direct-llm=4 \
  --set args.max-candidates-per-round=4 \
  --set args.max-empty-reservoir-rounds=2 \
  --set args.allow-early-stop=true \
  --set args.vina-bin="$VINA_BIN" \
  --set args.nn-model-path="$G12D" \
  --set args.vina-exhaustiveness=1 \
  --set args.vina-n-poses=1 \
  --set args.trajectory-dir=runs/first_real_tiny
```

The validated clean-room trial selected `CC1(C)NC(=O)c2ccccc2N1`, with Vina
`-7.725` and predicted G12D activity `4.642444678591025`. Model output can vary,
so treat those values as a reference, not an assertion for future runs.

## 10. Inspect Results And Run Tests

```bash
rg --files tasks/small_molecule/runs/first_real_tiny
sed -n '1,240p' \
  tasks/small_molecule/runs/first_real_tiny/summary.json

CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/small_molecule \
  pytest -q tests tasks/small_molecule/tests
```

Inspect `rounds.jsonl` for the selected molecule, `failed_evaluations`, Vina
score, and activity score. The validated clean-room suite reported `479 passed,
12 skipped`.

## 11. Clear Runtime Secrets

```bash
unset LLM_BASE_URL LLM_API_KEY LLM_MODEL_NAME LLM_MODEL
unset LDM_CREDENTIAL_FILE G12D VINA_BIN CUDA_VISIBLE_DEVICES
```

ReaSyn is required only when switching to a seed-analog method. Complete the
ReaSyn installation and full dependency check in the task README before using
that path.
