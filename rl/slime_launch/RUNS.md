# Small-molecule RL — run matrix

Real-mode GRPO on the small-molecule LDM loop. **Train on KRAS G12D** (ready
evaluator: `best_g12d_model.joblib` + 8UN5), **evaluate on G12C and G12D**.
Reward = acquisition (default), aggregated per-round. Two axes, sharing R2 as
the pivot:

| Run | Model (`MODEL_HF`) | Reward | Episodes file |
|-----|--------------------|--------|---------------|
| **R1** | Qwen3.5-9B base | acquisition-**max** | `rl_episodes_sm_acqmax.jsonl` |
| **R2** | SFT (no-GP) | acquisition-**max** | `rl_episodes_sm_acqmax.jsonl` |
| **R3** | SFT (no-GP) | **improvement** (real outcome) | `rl_episodes_sm_improve.jsonl` |
| **R4** | SFT (no-GP) | acquisition-**mean** | `rl_episodes_sm_acqmean.jsonl` |

- R1 vs R2 isolates the **model** (base vs SFT-init).
- R2 vs R3 vs R4 isolates the **reward** (decision-time max / real outcome / decision-time mean).
- Run each with ≥3–5 seeds (`--seed-offset`) for error bars.

## 0. One-time prep
```bash
cd /mnt/data0/ys/LDM/rl/slime_launch
# convert both models HF -> Megatron torch_dist
MODEL_HF=/mnt/data0/hf_models/models/Qwen3.5-9B                 SAVE=/mnt/data0/ys/LDM/rl/qwen3.5-9B_torch_dist      bash convert_9b.sh
MODEL_HF=/mnt/data0/hf_models/models/LDM-CoT-SFT               SAVE=/mnt/data0/ys/LDM/rl/qwen3.5-9B-sft_torch_dist  bash convert_9b.sh
# generate all episode files (reward baked in)
bash gen_episodes_runs.sh
# warm the shared GP once (rollout-only, docks warmup.num_samples molecules)
bash run_warmup_real_slime.sh    # or the 9B warmup once wired; GP history is model-agnostic
```

## 1. Launch runs (4 GPUs each, TP=2)
```bash
BASE=/mnt/data0/hf_models/models/Qwen3.5-9B ;      BASE_REF=/mnt/data0/ys/LDM/rl/qwen3.5-9B_torch_dist
SFT=/mnt/data0/hf_models/models/LDM-CoT-SFT ;      SFT_REF=/mnt/data0/ys/LDM/rl/qwen3.5-9B-sft_torch_dist
R=/mnt/data0/ys/LDM/rl ; export WANDB_KEY=<your-wandb-key>   # optional

# R1 base, acq-max
MODEL_HF=$BASE MODEL_REF=$BASE_REF EPISODES=$R/../rl_episodes_sm_acqmax.jsonl  SAVE=$R/qwen3.5-9B_rl_R1_base_acqmax  WANDB_RUN=R1_base_acqmax  bash run_train_real_9b.sh
# R2 sft, acq-max
MODEL_HF=$SFT  MODEL_REF=$SFT_REF  EPISODES=$R/../rl_episodes_sm_acqmax.jsonl  SAVE=$R/qwen3.5-9B_rl_R2_sft_acqmax   WANDB_RUN=R2_sft_acqmax   bash run_train_real_9b.sh
# R3 sft, real reward
MODEL_HF=$SFT  MODEL_REF=$SFT_REF  EPISODES=$R/../rl_episodes_sm_improve.jsonl SAVE=$R/qwen3.5-9B_rl_R3_sft_improve  WANDB_RUN=R3_sft_improve  bash run_train_real_9b.sh
# R4 sft, acq-mean
MODEL_HF=$SFT  MODEL_REF=$SFT_REF  EPISODES=$R/../rl_episodes_sm_acqmean.jsonl SAVE=$R/qwen3.5-9B_rl_R4_sft_acqmean  WANDB_RUN=R4_sft_acqmean  bash run_train_real_9b.sh
```

## 2. Reward semantics
- `reward: acquisition`, `acquisition_agg: max|mean` — per-round reward is the
  max (or mean) EHVI acquisition score of the evaluated candidate(s); episode
  reward is the sum over rounds. (See `rl/ldm_rl/env.py::_acquisition_reward`.)
- `reward: improvement` — per-round objective improvement over the incumbent
  (real Vina + activity outcome), summed over rounds.

## 3. Notes / risks
- **Env**: use the `ldm-slime-rl` image (torch 2.11 + matched TE); other stacks
  SIGSEGV on GRPO backward.
- **First 9B run**: smoke with `--mode mock` (edit episodes to mock) to confirm
  the hybrid backward + memory before spending real docking.
- **Memory**: 9B + TP=2 + sglang on 4×80G. If OOM, raise TP or drop
  `max_tokens_per_gpu`; recompute-full is already on.
- **Docking throughput** is the real bottleneck for real reward; enable the
  docking cache and raise `vina_max_workers` in `config_real.json` real_kwargs.
- All 4 runs **train on G12D**; the G12C transfer number comes from the
  offline eval harness, not from training.
