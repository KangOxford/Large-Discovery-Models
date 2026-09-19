# 小分子 RL 训练计划（目标 · 模型 · 实验组）

> 怎么装/怎么跑见 `HANDOFF.md` + `RUNS.md`。本文只讲**做什么、为什么**。

## 1. 目标

SFT 已经把"acquisition-tilted 搜索策略"的**格式与行为**蒸馏进 Qwen3.5-9B（见论文小分子结果）。
本轮 RL 的目标是:**用真实评估反馈(GRPO),在 SFT 之上进一步强化这套搜索策略**,并回答三个问题:

1. **RL 在 SFT 之上还有没有增益?**(SFT+RL vs SFT-only)
2. **RL 需不需要 SFT 打底?**(SFT+RL vs base+RL)
3. **哪种 reward 信号最有效?**(决策期 acquisition 的 max / mean vs 真实评估结果)

**泛化设定:train KRAS G12D → 评测 G12C(迁移)+ G12D(同分布)。**
> G12C/G12D 只差**活性模型**一个 joblib,**docking 受体 8UN5 两者共用**;所以 G12C 评测很轻,且**只在离线评测用,不进 RL 训练循环**(不阻塞训练)。

## 2. 要训哪些模型（2 个起点 × RL）

| 起点 | 说明 |
|---|---|
| **base Qwen3.5-9B** | 未微调,裸 base 起 RL(下界/对照) |
| **SFT no-GP 模型** | `Yangtze-ailab/LDM-CoT-SFT-Qwen3.5-9B-MixedScience`,主力起点 |

外加**无 RL 基线(不训,只评测)**:base(无RL)、SFT(无RL)——作为表格参照行。

## 3. 做哪几组实验（4 个 RL run，两组消融，共用 R2 为枢轴）

| Run | 起点 | reward | 归属 |
|---|---|---|---|
| **R1** | base | acquisition-max | Group A |
| **R2** | SFT | acquisition-max | Group A + B（枢轴） |
| **R3** | SFT | 真实评估结果(**hypervolume ΔHV**) | Group B |
| **R4** | SFT | acquisition-mean | Group B |

- **Group A — 模型消融**:R1 vs R2 → "SFT 打底对 RL 有没有用"。
- **Group B — reward 消融**:R2 vs R3 vs R4 → "决策期 acquisition(max/mean) vs 真实结果,哪个信号最好"。

全部 **train G12D**;reward 已烤进各自 episodes(`gen_episodes_runs.sh`)。
**每个 run 跑 ≥3–5 个 seed**(`--seed-offset`)出误差棒。

## 4. 评测协议

- 取每个 run 的 best checkpoint,放进小分子 LDM loop,在 **G12C 和 G12D 各跑 5 seed**,评测预算 budget=80。
- 指标:**Pareto 前沿 hypervolume**(越高越好),对齐论文 `tab:smallmol-cot`。
- 参照行:base(无RL)、SFT(无RL)、DeepSeek teacher(若要)。

**最终对照表(每格 = C / D 的 HV):**

| 策略 | G12C(迁移) | G12D(同分布) |
|---|---|---|
| base（无RL) | 参照 | 参照 |
| SFT（无RL) | 参照 | 参照 |
| R1 base+RL(acq-max) | | |
| R2 SFT+RL(acq-max) | | |
| R3 SFT+RL(真reward) | | |
| R4 SFT+RL(acq-mean) | | |

## 5. 预期结论 / 成功判据

- **R2 > SFT(无RL)**:RL 在 SFT 之上有增益(核心卖点)。
- **R2 > R1**:SFT 打底让 RL 更有效(而非从 base 硬学)。
- **R3 vs R2 vs R4**:确定最佳 reward 信号;若 R3(真结果)≥ R2,说明该直接优化结果;若 acquisition 系列更稳,支持"蒸馏 acquisition"叙事。
- **迁移(G12C)上仍有增益**:说明 RL 强化的是可迁移的搜索策略,不是对 G12D 的过拟合。

## 6. 分阶段执行（带验证闸,细节见 HANDOFF §E）

1. **P0**:1.5B 在真环境跑通 GRPO,验 **backward + docking + reward** 全链路。
2. **P1**:9B **real 极小 count**(count=1、iterations=2)冒烟,验 hybrid backward / 显存。**训练全程 real,不用 mock。**
3. **P2**:R1–R4 × seed 真训练(train G12D)。
4. **评测**:离线在 C & D 上评,填对照表。
   - 前置(可并行):用 `train_g12c_qsar.py` + `g12c_docking_benchmark.csv` 训出 `best_g12c_model.joblib`(受体沿用 8UN5)。

## 7. 规模 / 成本备注

- 4 run × 5 seed = 20 次训练;每次 4×80G GPU、real docking 是主瓶颈(开缓存 + 并行 `vina_max_workers`)。
- 若算力紧,先跑 **R2(主力)+ R1(对照)各 3 seed** 拿到核心结论,再补 R3/R4 与更多 seed。
