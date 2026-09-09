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

---

# 5. 2026-09-09 实测结果（Isambard-AI GH200，约 2,300 次真实 300 秒训练）

一晚的实测。**每一个数都来自真实训练，没有任何代理模型参与打分。**

完整分析（6 张图，可点开）：
`rl/slime_launch/campaign_20260909/nanogpt_rl_campaign.html`
数据：`campaign_20260909/campaign_data.json`（图和正文都只读它，所以不可能各说各话）

## 5.1 矩阵的实际完成度

| 格 | 实际产出 | 说明 |
|---|---|---|
| **R0a** 基座未训练 | 45 个 episode | 未跑满（规格是每种子 64 个） |
| **R0b** SFT 未训练 | 27 个 episode | 未跑满 |
| **R0c** LLM+GP 搜索 | 432 次评测 | **只跑了 warmup**，见 §5.3 |
| **R1** base + GRPO | **未跑** | 50 个 rollout 步需 21.3 小时，可用 4.9 小时；且集群零空闲 |
| **R2** SFT + GRPO | **未跑** | 同上 |
| 受控批量扫描 | 292 格 | 完成，4 档 |
| 奖励参考点 | 8 次测量 | 完成，取代了一个被挤占的值 |

**R1/R2 记为缺口而不是半启动。** 一条到不了结论的训练曲线，价值低于一句写清楚的"这件事没做"。

## 5.2 主要结果：一个旋钮解释了全部

受控单因子实验，只变 `TOTAL_BATCH_SIZE`，其余旋钮全部固定，292 次真实训练：

| `TOTAL_BATCH_SIZE` | n | `val_bpb` | MFU | steps | 相对默认 |
|---:|---:|---:|---:|---:|---|
| 262,144 | 106 | **0.993251** | 34.83% | 1660 | **−0.010583**，t = −52 |
| 524,288 | 63 | 1.003835 | 35.02% | 840 | *（默认值，也是奖励的参考点）* |
| 1,048,576 | 63 | 1.047850 | 35.16% | 427 | +0.044015，t = +118 |
| 2,097,152 | 60 | 1.092861 | 35.16% | 219 | +0.089027，t = +167 |

**MFU 四档持平、步数随批量减半而翻倍** —— 所以机制是固定墙钟预算下的优化步数，不是吞吐。

**搜索自己给出的观察性估计是 −0.0263，比因果值大 2.3 倍**，因为选了小批量的那批配置在别的维度上也系统性不同（策略是联合提议这些旋钮的）。

**这直接改变 §1 那三个问题该怎么问。** 如果这个动作空间里可达的增益由**一个单调旋钮**主导，那么 RL 策略的工作主要是"找到那个旋钮"——这比"RL 能提升调参能力"弱得多。**这件事应当在 R1/R2 跑之前定下来**，否则任何 reward 上升都会被读成那个更强的说法。

另外：文档引用的"整轮 99 次搜索总增益 0.0044"不是本机的上界。单旋钮一改就是 0.0106，四档跨度 0.0996。**墙钟预算下可达范围是机器的属性和配置空间的属性各占一半**，所以把别的硬件上的数当上界从来就不成立。

## 5.3 R0c 消耗了预算却没干活

每个复制都停在**恰好 8 次评测**，那正是 warmup 的次数。24 次 GP 引导迭代**每次只花 7–9 秒**（一次真实训练是 384 秒），GP 的 buffer 全程钉在 n=8，每次选择都报
`surrogate_score=1e9, pred=None, std=None, ei=None`；而 `budget.json` 记 `external_evaluations: 32 / 32`、`remaining: 0`、`failed_evaluation_count: 0`。

**所以跑的是无引导的 warmup 采样，不是规格里那套引导搜索。** 这让批量那个结果**更强**而不是更弱：连无引导采样都能撞上它。

在修好之前，任何引用 R0c 的地方都要写明它是 warmup-only。

## 5.4 三个会产出"看着合理的错数"的缺陷

**① 奖励的零点被测在一张有邻居的卡上。** 缓存里的 `val_bpb` 是 1.107706，8 次单租户重测是 1.003512 —— 差值是文档所述 99 次搜索全部增益的 **24 倍**，静默地平移了每一个 episode 的奖励，而那一行**格式完全正确**、通过一切检查。
**候选测错了是离群点、会被平均掉；参考点测错了是整个坐标系平移，平均不掉、复现不掉、事后也看不出来**，因为所有分数仍然自洽。
一般规则：**任何被缓存、被所有下游共用的参照量，测量条件必须比任何单个候选更严，而不是相同。**

**② 97% 的 `oom` 标签说的是"当时卡上有别人"。** 469 个格子记录了认领时刻该卡的占用：成功格中位数 **3 MiB**，OOM 格中位数 **91,120 MiB**，148 次 OOM 里 **144 次**发生在已被占用的卡上。
`oom` 是策略要学习的失败类别之一，所以策略会去避开完全可行的配置——**梯度是真的，方向是错的**。修法是多记一个字段，不是加显存预筛。

**③ 一次完整正确的测量被判成 crash。** GH200 上进程打印完整 metric block 之后才在 teardown 里收到 SIGBUS，并往 cwd 里 dump 了 944 MB 的 core。`rl_eval.py:547` 要求 `returncode == 0` **且** `val_bpb` 存在，于是那次测量被记成 `crashed` 丢掉。
已修：`train.py` 末尾 flush 后 `os._exit(0)`。**这移除了崩溃本身，而不是教分类器忽略崩溃**——训练中途的真崩溃仍然非零退出、仍然被抓到。

## 5.5 测量条件把分数移动了 80 倍噪声底

同一配置，卡被另一进程共享 vs 独占：MFU 15.4% vs 35.2%、379 步 vs 851 步、`val_bpb` 1.1089 vs 1.0026。两条独立测量路径互相印证。
而**四个进程分占一个节点的四张卡**，代价是 **−0.000008**，实质为零。

**规则是"一张卡一个评测"，不是"一个节点一个租户"。**

配套：**显存普查在你能用它之前就过期了**。一个节点读到 `[1,1,1,1]` MiB，**60 秒后**是 92,211 MiB。检查必须做在 step 内部，不是起跑之前。

## 5.6 三条被推翻的说法

| 说法 | 结论 | 证据 |
|---|---|---|
| `NANOGPT_HANDOFF.md` §5.1：9B hybrid 从未记录有限 optimizer step，nan 来自 full-attention backward | **基本正确**，应按细节更正而**不是**撤销 | `ldm_rl/runs/<RUN>/train.log`：grad_norm 75 nan/82 与 32 nan/32，而 `pg_loss`/`loss`/`kl_loss` 全 0 nan。（顶层 `runs/*.log` 是 srun 编排日志，不是逐步指标——在那里 grep 会得到相反且错误的结论） |
| slime 装不上 / import 不了 | **错** | slime 0.3.1 与 megatron_core 0.16.0rc0 在 `ldm-rl-train` 里是 editable 安装；配方是 `PYTHONPATH` 加 `CUDA_HOME` |
| Megatron 权重需要转换 | **错** | `qwen3.5-9B_torch_dist/release/` 与 `qwen3.5-9B-sft_torch_dist/release/` 都完整，各 17,079 MB。base 和 SFT 都现成 |

## 5.7 R0a 与 R0b 都是 0 分，但走的是相反的路

| | 真实训练 | 解析成功率 | 因不可解析而 0 | 因无改进而 0 |
|---|---|---|---|---|
| **R0a** 基座 | 4 | 33% | 0/3 | **3/3** |
| **R0b** SFT | **0** | **0%** | **3/3** | 0 |

R0b 的失败**是包装不是内容**：6 个非空轮次，只删掉 JSON 对象**外面**的字符（1、7、17、23、216、7816 个），6 个全部通过真实解析器；有一个 episode 的完整输出是一个合法提案**加一个孤立的反引号**。

**对 R1/R2 的直接含义**：如果 reward 涨了，涨的可能是「学会了输出干净 JSON」而不是「学会了搜索」。**reward 那一列分不开这两种 0**，`parsed_rounds` / `real_trainings` / `zero_reason` 能分开——它们现在写在每条记录里。
