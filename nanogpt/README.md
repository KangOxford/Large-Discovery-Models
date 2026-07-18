# nanoGPT Task Guide

The nanoGPT task searches over code or structured operation edits for a
training script. Mock runs use a tiny deterministic target. Real runs evaluate
candidate training code and optimize the reported validation BPB.

## Quick Start

From the repository root:

```bash
uv sync --project nanogpt
uv run --project nanogpt python scripts/run_ldm_tts.py config/nanogpt/mock_best_of_n.yaml
```

The mock config uses:

- `nanogpt/ldm_task/mock_train.py`
- `nanogpt/ldm_task/operation_schema_mock_train.json`
- output under `nanogpt/ldm_runs/`

## Environment

For real LLM-backed runs, configure an OpenAI-compatible endpoint. The config
files usually set `llm-url` and `llm-model-name` explicitly, but these
environment variables are also supported by the nanoGPT task:

```bash
export CUDA_VISIBLE_DEVICES=0
export TTS_LLM_URL=http://127.0.0.1:52312/v1
export TTS_LLM_MODEL=Qwen3-Coder-30B-A3B-Instruct
export TTS_LLM_API_KEY=EMPTY
```

`OPENAI_API_KEY` is also accepted when `TTS_LLM_API_KEY` is unset. Local
OpenAI-compatible servers often accept `EMPTY`; remote or authenticated
endpoints need a real key.

## Data And Tokenizer

Mock operation-search runs do not need the pretraining dataset. Real training
runs need data and tokenizer artifacts prepared by `prepare.py`.

`prepare.py` downloads text shards from `karpathy/climbmix-400b-shuffle`,
trains a BPE tokenizer with `rustbpe`, and writes metadata used by `train.py`.

Prepare a small local test dataset:

```bash
cd nanogpt
uv run python prepare.py --num-shards 8 --download-workers 8
```

Prepare the default number of shards:

```bash
cd nanogpt
uv run python prepare.py
```

Use `--num-shards -1` only when you intentionally want the full shard set. If
your machine cannot write to `/mnt/data0/hf_data/autoresearch`, point that path
to local storage with a mount or symlink, or edit `CACHE_DIR` in
`nanogpt/prepare.py` for your deployment.

## Dependency Check

Run the preflight from the repository root before a real run:

```bash
python scripts/check_task_dependencies.py config/nanogpt/real_operation_tool_best_of_n.yaml
```

The checker validates the target train file, operation schema, LLM settings,
CUDA visibility, and `prepare.py` data/tokenizer artifacts.

## Real Runs

Real code-search runs usually target `nanogpt/ldm_task/real_train.py` or
`nanogpt/train.py`, use `nanogpt/ldm_task/operation_schema_real_train.json`,
and evaluate candidates with:

```bash
--eval-command "uv run --project {repo_root}/nanogpt python {train_path}"
```

Starter configs:

```bash
uv run --project nanogpt python scripts/run_ldm_tts.py config/nanogpt/real_operation_tool_best_of_n.yaml
```

```bash
uv run --project nanogpt python scripts/run_ldm_tts.py config/nanogpt/real_operation_tool_fixed_best_of_n.yaml
```

Use `real_operation_tool_best_of_n.yaml` for dynamically expanded operation
features. Use `real_operation_tool_fixed_best_of_n.yaml` for the fixed full
operation schema.

## Customization

Operation schemas live under `nanogpt/ldm_task/`. The shared
`ldm_tts.parameter_space` module validates operation payloads, computes feature
dimensions, serializes schemas, and builds active feature subsets. For real
train-code search, keep the operation schema aligned with top-level assignments
in the target `train.py`.

Useful config fields:

| Field | Meaning |
| --- | --- |
| `args.train-file` | Seed training script to edit. |
| `args.operation-schema` | Structured operation space. |
| `args.generator` | `operation_mock`, `operation_tool`, or another generator. |
| `args.initial-operation-features` | Initial active operation-feature subset. |
| `args.max-active-operation-features` | Maximum active features after expansion; `0` means all. |
| `args.eval-command` | Command used to evaluate a generated candidate. |

Use a zero-iteration smoke run to write metadata without training:

```bash
python -m nanogpt.ldm_task.procedure \
  --project-root nanogpt \
  --train-file ldm_task/mock_train.py \
  --operation-schema ldm_task/operation_schema_mock_train.json \
  --generator operation_mock \
  --method best_of_n \
  --iterations 0 \
  --warmup 0 \
  --out-dir /tmp/ldm_tts_nanogpt_smoke \
  --run-name nanogpt_contract_smoke \
  --eval-command "python {train_path}" \
  --skip-eval \
  --no-progress
```
