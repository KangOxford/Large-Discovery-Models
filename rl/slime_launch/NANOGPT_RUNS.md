# nanoGPT RL — 运行矩阵

真实 reward 的 GRPO：奖励是**实测 `val_bpb`** 相对固定参考配置的降幅。
装环境和硬约束见 `NANOGPT_HANDOFF.md`（**先读那个**）。

---

## 1. 要回答的问题

按重要性排，第 3 个最容易被忽略但最关键。

1. **RL 能不能提升调参能力？** 训完的策略提出的配置，实测 `val_bpb` 是否显著低于
   未训练策略提出的。
2. **RL 需不需要 SFT 打底？** base+RL vs SFT+RL。
3. **RL 出来的 proposer 比现成的搜索强吗？** —— 这个任务**已经有**一套
   LLM 引导的 GP/BO 搜索（`tasks/nanogpt/scripts/run_model_based_search.py`）。
   所以"RL vs 不做 RL"不是有说服力的对照；有说服力的是
   **在相同真实评估预算下，RL 训出来的 proposer vs 那套现成搜索**。
   如果不做这一格，"RL 有用"的结论是站不住的。

---

## 2. Run 矩阵

| Run | 起点 | 说明 |
|---|---|---|
| **R0a** | base（不训） | 参照行：直接让 base 模型提配置 |
| **R0b** | SFT（不训） | 参照行 |
| **R0c** | — | **`run_model_based_search.py`**，相同真实评估预算 |
| **R1** | base | base + GRPO（真实 reward） |
| **R2** | SFT | SFT + GRPO（真实 reward）——主力 |

SFT 起点用 `Yangtze-ailab/LDM-CoT-SFT-Qwen3.5-9B-MixedScience`（和小分子那条线同一个）。

```bash
export REPO_ROOT=/path/to/LDM
export WORK_DIR=/scratch/nanogpt_rl
export MEGATRON_ROOT=/path/to/megatron-lm
export MODEL_ARGS_SCRIPT=$REPO_ROOT/rl/slime/scripts/models/qwen3.5-9B.sh
export TRAIN_GPUS=0,1,2,3
export NANOGPT_EVAL_GPUS=4,5,6,7
export EPISODES=$WORK_DIR/rl_episodes_nanogpt.jsonl

BASE=/path/to/Qwen3.5-9B ;  BASE_REF=$WORK_DIR/qwen3.5-9B_torch_dist
SFT=/path/to/LDM-CoT-SFT ;  SFT_REF=$WORK_DIR/qwen3.5-9B-sft_torch_dist

# R1
MODEL_HF=$BASE MODEL_REF=$BASE_REF SAVE=$WORK_DIR/nanogpt_R1_base \
  WANDB_RUN=R1_base bash $REPO_ROOT/rl/slime_launch/run_train_real_nanogpt.sh
# R2
MODEL_HF=$SFT  MODEL_REF=$SFT_REF  SAVE=$WORK_DIR/nanogpt_R2_sft \
  WANDB_RUN=R2_sft  bash $REPO_ROOT/rl/slime_launch/run_train_real_nanogpt.sh
```

**每个 run 至少 3 个 seed**：换环境 seed 要**重新生成 episode 数据**
（`gen_rl_episodes.py --seed-offset N`），slime 自己的 `--seed` 只管框架侧随机性，
不换实例轨迹。

**不同 run 用不同的 `WORK_DIR`（或至少不同的 `eval_cache_file`）。**
共享评估缓存会让两个 run 互相喂结果——那既省钱也串味，对照就不干净了。想省钱就
共享，想要干净对照就隔离；这是个明确的取舍，别默认。

---

## 3. 训练时看什么

| 指标 | 怎么读 |
|---|---|
| `train/grad_norm` | **先看这个。** nan = 这步更新被跳过了。9B hybrid 目前就卡在这（HANDOFF §5.1），**rollout 跑完 ≠ 训练跑通** |
| 组内 reward std | GRPO 拿它归一化 advantage。奖励经常恰好是 0（没改进），所以零方差组是真实风险——如果常见，提高 `rollout_temperature` |
| `failure_kind` 分布 | 从 `eval_cache.jsonl` 统计。`rejected_by_trainer` 占比高 = 策略还没学会整除约束；`oom` 占比高 = 在提过大的模型 |
| 缓存命中率 | 上升是正常的收敛信号；接近 1 说明策略已经塌到少数几个配置上了 |
| episode 奖励 | **注意单位**：bpb 的降幅。0.004 就已经相当于 upstream 一整轮 99 次真实搜索的总收益 |

统计缓存的一行命令：

```bash
python3 - <<'PY'
import json, collections
rows = [json.loads(l) for l in open("$WORK_DIR/eval_cache.jsonl")]
print("total", len(rows), "ok", sum(r["ok"] for r in rows))
print(collections.Counter(r["failure_kind"] for r in rows if not r["ok"]))
ok = [r["metrics"]["val_bpb"] for r in rows if r["ok"]]
if ok: print(f"best {min(ok):.6f}  median {sorted(ok)[len(ok)//2]:.6f}")
PY
```

---

## 4. 评测协议（重要：必须做重复测量）

可达增益只有测量噪声的约 3 倍（HANDOFF §5.2），所以 best-of-N 天然有向上的
噪声偏差，**单次测量不能作为结论**。

1. 取每个 run 的 best checkpoint，在**留出的实例**上贪心提配置。
2. 每个最终候选**真实重复测量 ≥4 次**，报告均值和标准差。
   重复跑同一配置要先删掉它在 `eval_cache.jsonl` 里的那行，否则命中缓存。
3. 和 R0a/R0b/R0c 在**相同真实评估次数**下比较。
4. 判定用实测噪声底 **0.0013 bpb** 作单位；小于它的差异不要解读。

**成功判据**：R2 的最终配置在重复测量下显著优于 R0b（未训 SFT），并且
**至少不差于 R0c**（现成的 LLM+GP 搜索）在相同真实评估预算下的结果。
