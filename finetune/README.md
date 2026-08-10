# Full SFT: Distilling the LDM-TTS Rationale Policy

This directory contains the full-parameter supervised fine-tuning recipe for an
LDM-TTS proposal model. The canonical data pipeline remains under [`../data`](../data):
it collects accepted teacher actions as `ldm-2.0` IR, adds evidence-grounded
reasoning, and validates the resulting corpus. This directory turns that augmented
IR into group-disjoint train/evaluation shards and runs LlamaFactory.

## Target Contract

The model target is the repository's existing JSON action contract:

```json
{"type":"propose","reasoning":"<visible rationale>","payload":{},"summary":"..."}
```

Reasoning stays in the JSON `reasoning` field. It is not emitted through Qwen's
hidden `<think>...</think>` channel. This keeps training and inference aligned
with the task prompts, validators, and response parsers that require one JSON
action. The recipe therefore uses the `qwen3_5_nothink` template.

The target is the accepted teacher proposal before BO selection and evaluation.
Selected candidates, objective values, acquisition probabilities, and provenance
remain audit metadata and are not training targets.

## Layout

```text
finetune/
  README.md
  prepare_dataset.py
  config/
    full_sft_rationale.yaml
  LLaMA-Factory/                  # pinned git submodule

data/generated/full_sft/         # generated and ignored by git
  ldm_rationale_train.jsonl
  ldm_rationale_eval.jsonl
  dataset_info.json
  split_summary.json
  checkpoints/
```

## 1. Install LlamaFactory

Initialize the pinned framework from the repository root:

```bash
git submodule update --init --recursive finetune/LLaMA-Factory

cd finetune/LLaMA-Factory
conda create -n llama python=3.11 -y
conda activate llama
```

Install a CUDA-compatible PyTorch build for the training host, then install the
pinned framework and its DeepSpeed requirements:

```bash
pip install -e .
pip install -r requirements/deepspeed.txt
cd ../..
```

The editable package already declares PyTorch as a core dependency. DeepSpeed is
not a package extra at the pinned revision, so it must be installed from the
requirements file. Full-parameter SFT of a 9B model with ZeRO-3 CPU offload also
requires substantial aggregate GPU memory and host RAM.

FlashAttention can improve throughput when a compatible build is available:

```bash
pip install flash-attn --no-build-isolation
```

After installing it, change `flash_attn: sdpa` to `flash_attn: fa2` in the
training config.

## 2. Collect And Augment IR

Follow [`../data/README.md`](../data/README.md) to collect accepted actions and
add expert reasoning. The full-SFT preparation input is augmented IR, not an
already rendered Alpaca file:

```text
data/generated/my_campaign/ldm_ir_augmented.jsonl
```

Preserve the original IR. The preparation step reads it without modification.

## 3. Build Grouped Train And Evaluation Shards

From the repository root, prepare the default full-SFT workspace:

```bash
python finetune/prepare_dataset.py \
  --input data/generated/my_campaign/ldm_ir_augmented.jsonl \
  --output-dir data/generated/full_sft \
  --eval-fraction 0.10 \
  --seed 42
```

Repeat `--input` to combine campaigns. Rows are grouped by a run or trajectory
identifier, including the small-molecule collector's `trajectory_dir`;
`antigen`/`seed` are used when no run identifier exists. For historical IR
without per-row provenance, each input file is treated as one group. Such data
must be supplied as at least two run-level files. `--eval-fraction` is applied
to whole groups rather than individual rows, so the realized row fraction can
differ when group sizes are uneven.

The command:

- validates every input as `ldm-2.0` IR;
- excludes records marked `reasoning_available: false`;
- excludes records without a non-empty action-level rationale;
- assigns whole provenance groups to either train or evaluation;
- renders both shards with the canonical prose renderer;
- writes the matching LlamaFactory `dataset_info.json` and `split_summary.json`.

Existing outputs are preserved by default. Pass `--overwrite` only when replacing
the prepared shards intentionally.

## 4. Run Data Quality Gates

Run the maintained unit tests and audit the augmented source IR:

```bash
python -m pytest \
  tests/test_data_collection.py \
  tests/test_data_augmentation.py \
  tests/test_finetune_preparation.py

python data/build_ldm2.py audit \
  --in-ir data/generated/my_campaign/ldm_ir_augmented.jsonl

python data/verify.py validity \
  --in-ir data/generated/my_campaign/ldm_ir_augmented.jsonl
```

Then verify both rendered shards, their registry, and the configured context
limit:

```bash
python data/verify.py alpaca \
  --sft data/generated/full_sft/ldm_rationale_train.jsonl \
  --dataset-info data/generated/full_sft/dataset_info.json \
  --cutoff-len 16384

python data/verify.py alpaca \
  --sft data/generated/full_sft/ldm_rationale_eval.jsonl \
  --dataset-info data/generated/full_sft/dataset_info.json \
  --cutoff-len 16384
```

Inspect `split_summary.json` before training. Confirm that both shards are
non-empty, skipped-row counts are expected, and evaluation groups are genuinely
held out at the run, antigen, or seed level.

## 5. Train

The provided config fine-tunes `Qwen/Qwen3.5-9B`. For another
LlamaFactory-supported model, update both `model_name_or_path` and `template`.

Run from `finetune/` so the config's relative paths resolve correctly:

```bash
cd finetune
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  llamafactory-cli train config/full_sft_rationale.yaml
```

The default checkpoint path is under the ignored prepared-data workspace:

```text
data/generated/full_sft/checkpoints/qwen3_5_9b_full_sft_rationale/
```

`overwrite_output_dir` is disabled. Use a distinct checkpoint directory for each
experiment, or resume explicitly according to LlamaFactory's checkpoint workflow.
To log to Weights & Biases, set `report_to: wandb` and add a unique `run_name`.

The effective batch size is `1 x 8 x #GPUs`, using `bf16`, gradient
checkpointing, a `16384` token cutoff, cosine scheduling, `1e-5` learning rate,
and two epochs.

## 6. Preserve Inference Parity

Prompt the trained model with the same renderer used for SFT:

```python
from ldm_tts.data import render_prose
```

Serve Qwen with thinking disabled and pass the rendered prompt plus the matching
task system message. The response must remain one JSON action with `type`,
`reasoning`, `payload`, and `summary`. Do not deploy this checkpoint against a
legacy task-specific prompt or strip the parent artifact unless the same change
was made and evaluated during training.
