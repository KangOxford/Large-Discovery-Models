# Fine-Tuning: Distilling the LDM-TTS Policy

This directory reproduces the supervised fine-tuning (SFT) stage of the Large
Discovery Model. Test-time search with the LDM loop (`generate -> select ->
evaluate -> update`) produces trajectories of proposal decisions; we distil that
*acquisition-guided policy* back into the base LLM so a single forward pass
proposes candidates the way the full search loop would. Concretely, each round's
context becomes the prompt and the accepted proposal (optionally with its
chain-of-thought rationale) becomes the target, in Alpaca format, and we run a
full-parameter SFT with DeepSpeed ZeRO-3 (CPU offload).

The example configs here fine-tune `Qwen/Qwen3.5-9B`, but the recipe is
model-agnostic: point `model_name_or_path` at any LLaMA-Factory-supported model
and adjust `template` accordingly. Two targets are provided:

| Config | Target | `enable_thinking` | Dataset |
| --- | --- | --- | --- |
| `config/full_sft_cot.yaml` | chain-of-thought (default) | `true` | `ldm_mixed_cot` |
| `config/full_sft_nocot.yaml` | direct answer | `false` | `ldm_mixed_nocot` |

See [`../DATA_COLLECTION.md`](../DATA_COLLECTION.md) for how the search
trajectories are collected and rendered into these Alpaca datasets.

## Layout

```text
finetune/
  README.md
  config/
    full_sft_cot.yaml      # full SFT, chain-of-thought target
    full_sft_nocot.yaml    # full SFT, direct-answer target
  data/
    dataset_info.json      # LLaMA-Factory dataset registration
    # ldm_mixed_cot_sft.jsonl        # prepared in step 2
    # ldm_mixed_nocot_sft.jsonl      # prepared in step 2
  LLaMA-Factory/           # git submodule (training framework)
```

## 1. Install LLaMA-Factory

The training framework is pinned as a git submodule. From the repository root:

```bash
git submodule update --init --recursive finetune/LLaMA-Factory

cd finetune/LLaMA-Factory
conda create -n llama python=3.11 -y && conda activate llama
pip install -e ".[torch,deepspeed]"
cd ../..
```

Requires a CUDA-capable PyTorch build and DeepSpeed. Full-parameter SFT of a 9B
model with ZeRO-3 offload was validated on 2x-8x GPUs with ample CPU RAM (the
optimiser and parameter states are offloaded to host memory).

Installing FlashAttention can noticeably speed up training. Install a build that
matches your CUDA and PyTorch versions (e.g. `pip install flash-attn
--no-build-isolation`) and set `flash_attn: fa2` in the config. The provided
configs default to `flash_attn: sdpa`, which needs no extra dependency and runs
anywhere.

## 2. Prepare the dataset

Render the collected search trajectories into Alpaca-format SFT files following
[`../DATA_COLLECTION.md`](../DATA_COLLECTION.md), and place them in
`finetune/data/`:

```text
finetune/data/
  dataset_info.json          # already provided
  ldm_mixed_cot_sft.jsonl    # chain-of-thought target
  ldm_mixed_nocot_sft.jsonl  # direct-answer target
```

`dataset_info.json` already registers these files as the `ldm_mixed_cot` and
`ldm_mixed_nocot` datasets. Each row is a standard Alpaca record:

```json
{"instruction": "<round context / task prompt>", "input": "", "output": "<accepted proposal (+ rationale for CoT)>", "system": "<system prompt>"}
```

## 3. Train

Run from the `finetune/` directory so the relative paths in the config resolve:

```bash
cd finetune
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  llamafactory-cli train config/full_sft_cot.yaml
```

Checkpoints are written to `output/ldm_cot/`. Swap in
`config/full_sft_nocot.yaml` for the direct-answer variant. To log to Weights &
Biases, set `report_to: wandb` and add a `run_name` in the config.

### Key hyper-parameters

Full-parameter SFT, DeepSpeed ZeRO-3 with CPU offload, `bf16`, gradient
checkpointing, `cutoff_len: 16384`, effective batch size
`per_device_train_batch_size (1) x gradient_accumulation_steps (8) x #GPUs`,
`learning_rate: 1e-5`, cosine schedule, `warmup_ratio: 0.03`, 2 epochs.
`enable_thinking` selects the chain-of-thought vs direct-answer target.
