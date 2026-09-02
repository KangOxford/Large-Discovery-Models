# 小分子 RL 交接指南 —— 实施进度对照

对照 [`HANDOFF.md`](HANDOFF.md) 逐条列出交接方要求的事项与当前状态。
在 **Isambard-AI**（GH200 / aarch64 / 驱动 565 / CUDA 12.7，分区 `workq`）上执行。

**读法**：~~划掉的是已完成的~~；**未完成的加粗并链接到一个 notebook**，
每个未完成项对应一个文件，里面有图、数据和判定标准。

---

## 总览

| HANDOFF 章节 | 事项 | 状态 |
|---|---|---|
| §0 | 硬件前置 | ~~完成~~ |
| §1 | CUDA 红线（cu128/cu129） | ~~完成~~ |
| §2A | 训练栈（torch / TE / Megatron / Slime / sglang） | ~~完成~~ |
| §2B | 评测栈（docking + GP + activity） | ~~完成~~ |
| §2C | 仓库本体 | ~~完成~~ |
| §3 | 资产（模型 / vina / 活性模型 / 受体） | ~~完成~~ |
| §4 | 准备（转换 / 生成 episodes / 暖机 GP） | ~~完成~~ |
| §5 | 怎么跑（4 个 run × ≥3–5 seed） | **未完成** · [03](handoff_notebooks/03_p2_training_matrix.ipynb) · [06](handoff_notebooks/06_seed_offset_missing.ipynb) |
| §6 P0 | 1.5B 跑通 GRPO，验 backward + docking + **reward** | 部分 —— backward/docking ~~完成~~，**reward 未通过** · [01](handoff_notebooks/01_reward_always_zero.ipynb) |
| §6 P1 | 9B real 极小 count 冒烟 | 部分 —— 冒烟 ~~成功过~~，**稳定性未完成** · [02](handoff_notebooks/02_9b_run_stability.ipynb) |
| §6 P2 | R1–R4 × seed 真训练 | **未开始** · [03](handoff_notebooks/03_p2_training_matrix.ipynb) |
| §6 步骤 4 | 离线在 C & D 上评测 | **未开始**（前置 QSAR ~~已完成~~）· [04](handoff_notebooks/04_offline_evaluation.ipynb) |
| §7 | 已知坑 | ~~全部验证过~~，另有三条新的（见文末） |
| — | GP kernel 选择与 EHVI 衰减 | HANDOFF 未涉及，但 P2 前必须定 · [05](handoff_notebooks/05_gp_kernel_and_ehvi_decay.ipynb) |

**一句话结论**：环境和准备工作全部完成，1.5B 全链路跑通，9B 冒烟成功过；
但 **GRPO 拿不到梯度**：环境 reward 是正常的（555 步里 87% 在 1e-6 以上，
中位数 1e-2），坏掉的是**组内方差**——`advantage = (r − mean) / std`，
一组轨迹拿到相同 reward 时分子分母同时为 0。所以 P2 现在跑也学不到东西，
这是当前唯一需要先解决的事。

---

## §0 硬件前置

- ~~`uname -m` → aarch64；驱动 565 / CUDA 12.7；4×GH200 每卡 96 GB~~
- ~~磁盘：模型 + torch_dist + 中间产物均已落地~~
- ~~能联网到 HuggingFace~~

## §1 ⚠️ CUDA 红线

- ~~torch 装的是 `+cu128`，`torch.cuda.is_available()` 为 True~~

这条红线在别处踩到过并确认：cu130 的轮子跨 major 版本必挂，
`is_available()` 直接 False，与 HANDOFF 描述一致。

## §2 装环境

### 2A 训练栈
- ~~torch（aarch64, cu128）~~
- ~~Transformer Engine —— 装成，**且 backward 不 SIGSEGV**（§6 P0 已验）~~
- ~~Megatron-LM checkout 到 `1dcf0daf`~~
- ~~Slime + ray~~
- ~~sglang（aarch64 可用版本）~~
- ~~APEX：未装，脚本走 `--no-gradient-accumulation-fusion`，按 HANDOFF 说明可省~~

TE 这一环是 HANDOFF 点名「ABI 最敏感」的，实测在 aarch64 上装成且
反向传播正常——**HANDOFF 担心的那个风险没有发生**。

### 2B 评测栈
- ~~gpytorch / gauche / lightgbm / scikit-learn / joblib / rdkit / meeko / gemmi …~~
- ~~vina 二进制（aarch64），路径已填进 `config_real.json:vina_bin`~~
- ~~`vina_max_workers` 已按 288 核调~~

docking 实测单次 `env.step` 4.09 秒，与 HANDOFF 预期的量级吻合。

### 2C 仓库本体
- ~~clone + submodule + PYTHONPATH~~

## §3 资产

| 资产 | 状态 |
|---|---|
| base Qwen3.5-9B | ~~已下载~~ |
| SFT no-GP 模型（LDM-CoT-SFT） | ~~已下载~~ |
| 冒烟用 Qwen2.5-1.5B-Instruct | ~~已下载~~ |
| vina 二进制（aarch64） | ~~已装~~ |
| G12D 活性模型 | ~~随 repo 提供~~ |
| 8UN5 受体 | ~~已就位~~ |

## §4 准备（一次）

- ~~`convert_9b.sh` 转 Megatron torch_dist —— base 与 SFT 各一次，两个目录都在~~
- ~~`gen_episodes_runs.sh` 生成 4 个 run 的 episodes~~
- ~~`run_warmup_real_slime.sh` 暖共享 GP —— 得到 63 行、41 个唯一分子~~

在此基础上，各 run 至今累计追加 **1751 次真实评测**，
去重后的分子池已重建到 **1493 个唯一分子**。

## §5 怎么跑

- ~~单节点 4×GH200 的 TP / rollout-gpus / actor-gpus 已按 4 卡调~~
- ~~Slurm 提交路径打通~~（实际改用 attach 到已有分配，见文末新坑第 2 条）
- **每个 run ≥3–5 seed** → 未做，见 [03](handoff_notebooks/03_p2_training_matrix.ipynb)
- **`--seed-offset` 这个参数不存在** → 见 [06](handoff_notebooks/06_seed_offset_missing.ipynb)

## §6 训练计划

### ~~P0：1.5B 跑通 GRPO~~ —— 三项里通过两项

- ~~backward 在 ARM/TE 上不 SIGSEGV~~
- ~~docking 全链路通~~
- **reward 未通过** → [01_reward_always_zero.ipynb](handoff_notebooks/01_reward_always_zero.ipynb)

  全链路能跑 **≠** 训练能学。**先纠正我自己一次读错字段**：
  `rollout/rewards` 是 GRPO 归一化之后的 advantage，不是环境 reward；
  环境 reward 在同一个字典里叫 `raw_reward`。读对之后：
  **34 个 run、555 步，环境 reward 有 87% 在 `1e-6` 以上、中位数 `1e-2`**，
  完全正常。

  坏的是**组内方差**：`advantage = (r − mean) / std`，一组轨迹拿到相同
  reward 时分子分母同时为 0，梯度为零——**reward 再大也没用**。
  实测零方差组累计 **409 次**。一个成因是解析失败率约 81%
  （观测被裸拼进对话破坏了 chat 格式，round 1 之后模型不再输出 JSON），
  整组一起失败就一起拿 0。

  **格式修复已验证，但只解决了一半**：把观测包成独立的 user turn 之后，
  **解析失败率 81.1% → 0.5%**（1249 次失败基本清零）。
  而**零方差组没有下降**：0.69 → 0.84 每步，反而略升。

  我先前在这里报过「降 43%（0.69 → 0.39）」，那是**只跑了 36 步时**的读数，
  步数涨到 147 之后翻转。口径已改成只计步数 ≥15 的 run——
  只跑几步的 run 的 `zero_std/step` 噪声极大（0.00 与 1.00 都出现过）。

  按预登记的判定标准，这个结果指向明确的下一步：**解析失败几乎不是
  零方差的成因**（失败率 0.5% 时零方差仍是 0.84），成因在
  `n_samples_per_prompt`（当前是 2，一组只有两条轨迹）或候选去重。
  `n=8` 的对照正在跑。

### P1：9B real 极小 count 冒烟 —— 冒烟成功，稳定性未完成

- ~~hybrid backward 与显存验证通过~~：双节点分卡布局下 `needs_offload=False`，
  绕开了 `torch_memory_saver` 那个断言；`R2-nonanguard` 单个 run 产出 500 次真实评测
- **29 个 9B run 至今零存活** → [02_9b_run_stability.ipynb](handoff_notebooks/02_9b_run_stability.ipynb)

  45% 死于 GPU 显存被邻居占用，21% 死于**主机内存 OOM**（限制在 job 级 cgroup，
  同一分配下所有 step 共享 449 GB）。后者可以事先查，前者在
  `nodelock` 被禁用后规则内无法预留。

### P2：R1–R4 × seed 真训练 —— **未开始**

→ [03_p2_training_matrix.ipynb](handoff_notebooks/03_p2_training_matrix.ipynb)

七项技术准备已就绪（kernel、分卡布局、跨分配起 run、独立 GP/seed/输出目录、
编排活过会话、节点排除、主机内存预检），**挡着的是 reward 有效性、
9B 存活率、以及同时凑出 16 个空节点**。

### 步骤 4：离线在 C & D 上评测 —— **未开始**

→ [04_offline_evaluation.ipynb](handoff_notebooks/04_offline_evaluation.ipynb)

- ~~前置：`train_g12c_qsar.py` 训出的 QSAR 模型已完成~~
  （`g12c_qsar_20260901T010923Z/best_model.joblib` 与 `g12d_qsar_matched_.../best_model.joblib`）
- 评测本身缺的只是被评对象——P2 还没产出检查点

## §7 已知坑 —— 原有的都验证过

- ~~CUDA 红线：确认 cu130 不可用~~
- ~~TE / backward：aarch64 上装成，1.5B 冒烟通过~~
- ~~9B 是 hybrid：转换/训练用 `qwen3.5-9B.sh` 的 spec，走通~~
- ~~显存：9B + 分卡布局在 96 GB/卡上宽裕，每 rank 37.06 GiB 与预算逐位吻合~~
- ~~docking 吞吐：`vina_max_workers=32` 生效，aarch64 vina 正常~~
- ~~代码架构无关：`ldm_rl` 的 61 个测试在这里全过~~

### 新增三条

1. **`--rollout-num-gpus 4` 会起 4 个独立 sglang 引擎**，每个都把完整的 9B
   读进主机内存，这是主机内存 OOM 的直接来源。降到 2 个能减半，代价是一半推理吞吐。
2. **`setsid nohup` 保护不了 Slurm step**：它只让编排脚本脱离会话，
   而 srun 是它的子进程，step 的生命周期绑在 srun 客户端上。会话切换时
   所有这样起的 run 会被 `srun: forcing job termination` 收割。要放 tmux。
3. **`ehvi_all.py` 有一个 bare `except Exception: return _fallback(...)`**，
   GP 出任何问题都会静默返回全零 EHVI。这次证据表明它没有触发
   （`fallback_reason` 全是 None），但它是一个会把建模问题伪装成
   「reward 就是 0」的静默失败点。

---

## 每个未完成项的详细说明

| # | notebook | 讲什么 |
|---|---|---|
| 01 | [GRPO 组内零方差](handoff_notebooks/01_reward_always_zero.ipynb) | 环境 reward 正常（87% >1e-6），零方差组 409 次；成因与判定标准 |
| 02 | [9B 训练不稳定](handoff_notebooks/02_9b_run_stability.ipynb) | 两类 OOM 的区分与各自修法；job 级 cgroup 的 449 GB 限制 |
| 03 | [P2 训练矩阵](handoff_notebooks/03_p2_training_matrix.ipynb) | 十项前置的就绪度；矩阵规模与代价 |
| 04 | [离线评测](handoff_notebooks/04_offline_evaluation.ipynb) | 前置已完成；不依赖 P2 的两件可先做的事 |
| 05 | [GP kernel 与 EHVI 衰减](handoff_notebooks/05_gp_kernel_and_ehvi_decay.ipynb) | sk 阶 2.68 生产不可用；EHVI 随历史趋零是 P2 的隐患 |
| 06 | [`--seed-offset` 不存在](handoff_notebooks/06_seed_offset_missing.ipynb) | 三个 seed 是三件不同的事；HANDOFF 该怎么改 |

完整的调查记录（含每条结论的证据与被推翻的中间结论）在
`/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_rl/results/FINDINGS.md`。

四个已修的代码缺陷已开 PR：
<https://github.com/YihangChen9/Large-Discovery-Models/pull/1>
