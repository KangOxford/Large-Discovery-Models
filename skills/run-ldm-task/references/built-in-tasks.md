# Built-In Task Run Recipes

## Contents

- [Common Rules](#common-rules)
- [nanoGPT](#nanogpt)
- [Small Molecule](#small-molecule)
- [Antibody](#antibody)

## Common Rules

Treat each task README as authoritative if its commands change. Use the same
`--set` overrides for dependency checks and execution so preflight evaluates the
actual plan. Runner-level `--dry-run` prints command resolution only; the
task-level contract commands below additionally exercise argument parsing and
emit the semantic task specification without objective evaluation.

All task adapters accept OpenAI-compatible URL, model, and key arguments. Never
place authenticated keys in tracked YAML or literal shell commands.

## nanoGPT

Files:

```text
tasks/nanogpt/README.md
config/nanogpt/mock_best_of_n.yaml
config/nanogpt/real_operation_tool_best_of_n.yaml
config/nanogpt/real_operation_tool_fixed_best_of_n.yaml
```

Model variables: `TTS_LLM_URL`, `TTS_LLM_MODEL`, `TTS_LLM_API_KEY`.

Mock run:

```bash
uv run --locked --project tasks/nanogpt python scripts/check_task_dependencies.py \
  config/nanogpt/mock_best_of_n.yaml --no-optional
uv run --locked --project tasks/nanogpt python scripts/run_ldm_tts.py \
  config/nanogpt/mock_best_of_n.yaml --dry-run
uv run --locked --project tasks/nanogpt python scripts/run_ldm_tts.py \
  config/nanogpt/mock_best_of_n.yaml
```

Light real dependency check and zero-iteration contract smoke:

```bash
uv run --locked --project tasks/nanogpt python scripts/check_task_dependencies.py \
  config/nanogpt/real_operation_tool_best_of_n.yaml \
  --set args.iterations=0 \
  --set args.warmup=0 \
  --set args.skip-eval=true \
  --no-optional

uv run --locked --project tasks/nanogpt python scripts/run_ldm_tts.py \
  config/nanogpt/real_operation_tool_best_of_n.yaml \
  --set args.iterations=0 \
  --set args.warmup=0 \
  --set args.skip-eval=true \
  --set args.run-name=nanogpt_real_contract_smoke
```

Tiny real evaluation after preparing data and tokenizer:

```bash
uv run --locked --project tasks/nanogpt python scripts/check_task_dependencies.py \
  config/nanogpt/real_operation_tool_best_of_n.yaml \
  --set args.iterations=1 \
  --set args.warmup=0 \
  --set args.breadth=1 \
  --set args.depth=1

uv run --locked --project tasks/nanogpt python scripts/run_ldm_tts.py \
  config/nanogpt/real_operation_tool_best_of_n.yaml \
  --set args.iterations=1 \
  --set args.warmup=0 \
  --set args.breadth=1 \
  --set args.depth=1 \
  --set args.run-name=nanogpt_real_tiny
```

The light check may omit data only when `skip-eval=true` and `--no-optional`
are both set. Real evaluation must fail when data/tokenizer artifacts are absent.

## Small Molecule

Files:

```text
tasks/small_molecule/README.md
config/small_molecule/mock_m1_stratified_oversample.yaml
config/small_molecule/real_m1_seed_analog.yaml
```

Model variables: `LLM_BASE_URL`, `LLM_MODEL_NAME`, `LLM_API_KEY`.
Evaluator variables: `VINA_BIN`, `G12D`; ReaSyn paths are needed only by analog
methods.

Mock run:

```bash
uv run --locked --project tasks/small_molecule python scripts/check_task_dependencies.py \
  config/small_molecule/mock_m1_stratified_oversample.yaml --no-optional
uv run --locked --project tasks/small_molecule python scripts/run_ldm_tts.py \
  config/small_molecule/mock_m1_stratified_oversample.yaml --dry-run
uv run --locked --project tasks/small_molecule python scripts/run_ldm_tts.py \
  config/small_molecule/mock_m1_stratified_oversample.yaml
```

Light real dependency check and contract smoke:

```bash
uv run --locked --project tasks/small_molecule python scripts/check_task_dependencies.py \
  config/small_molecule/real_m1_seed_analog.yaml \
  --set args.vina-bin="$VINA_BIN" \
  --set args.nn-model-path="$G12D" \
  --no-optional

uv run --locked --project tasks/small_molecule python scripts/run_ldm_tts.py \
  config/small_molecule/real_m1_seed_analog.yaml \
  --set args.dry-run=true \
  --set args.budget=1 \
  --set args.init-size=1 \
  --set args.trajectory-dir=runs/small_molecule/first_real_contract
```

Tiny real direct-LLM evaluation:

```bash
uv run --locked --project tasks/small_molecule python scripts/run_ldm_tts.py \
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
  --set args.trajectory-dir=runs/small_molecule/first_real_tiny
```

`--no-optional` omits ReaSyn only for a direct method. Do not use it after
switching to an analog method.

## Antibody

Files:

```text
tasks/antibody/README.md
tasks/antibody/resources/default_config.yaml
config/antibody/mock_ei.yaml
config/antibody/real_lcb.yaml
```

Model variables: `LLM_BASE_URL`, `LLM_MODEL_NAME`, `LLM_API_KEY`. Real runs also
require `bbox.path` in `tasks/antibody/resources/default_config.yaml` to locate Absolut.

Mock run:

```bash
uv run --locked --project tasks/antibody python scripts/check_task_dependencies.py \
  config/antibody/mock_ei.yaml --no-optional
uv run --locked --project tasks/antibody python scripts/run_ldm_tts.py \
  config/antibody/mock_ei.yaml --dry-run
uv run --locked --project tasks/antibody python scripts/run_ldm_tts.py \
  config/antibody/mock_ei.yaml
```

Real dependency check and contract smoke for one antigen:

```bash
uv run --locked --project tasks/antibody python scripts/check_task_dependencies.py \
  config/antibody/real_lcb.yaml \
  --set args.antigen=YOUR_ANTIGEN

uv run --locked --project tasks/antibody python scripts/run_ldm_tts.py \
  config/antibody/real_lcb.yaml \
  --set args.antigen=YOUR_ANTIGEN \
  --set args.dry-run=true \
  --set args.budget=1 \
  --set args.n-init=1 \
  --set args.parallel-budget=8 \
  --set args.out-dir=runs/antbo_first_real_contract
```

Tiny real evaluation:

```bash
uv run --locked --project tasks/antibody python scripts/run_ldm_tts.py \
  config/antibody/real_lcb.yaml \
  --set args.antigen=YOUR_ANTIGEN \
  --set args.budget=1 \
  --set args.batch-size=1 \
  --set args.n-init=1 \
  --set args.parallel-budget=8 \
  --set args.n-trials=1 \
  --set args.out-dir=runs/antbo_first_real_tiny
```

The tiny real run contacts the configured model and evaluates one selected
sequence with Absolut. A missing Absolut installation is a blocking failure.
