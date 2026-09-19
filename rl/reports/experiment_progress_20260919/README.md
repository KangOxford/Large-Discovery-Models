# 训练实验与阶段性结果

优先阅读 **[已执行 Notebook](experiment_progress.ipynb)** 或 **[四页 PDF](experiment_progress.pdf)**。
[独立 HTML](experiment_progress.html) 内嵌全部图片，下载后可离线打开；每张图也有 PNG 与 SVG。

四个小节只保留标题，每张图内含读图说明、定义、样本量、比较限制与来源：

1. 小模型链路验证：一条 1.5B 运行的 30 轮采样、60 次有限梯度记录与三个模型存档。
2. 9B 数值诊断：选定运行的梯度轨迹与有限／非有限计数。
3. 奖励诊断：两条运行各 100 个任务组，49 与 9 个全零组，以及逐轮分布。
4. 结论所需证据：从运行产物到可靠训练，再到 G12D/G12C 的同预算效果评测。

## 数据与复现

实验记录截至 2026-09-10，本次核验为 2026-09-19。这是选定诊断运行的报告，不是完整实验矩阵，五条运行也不是五个独立随机种子。

`data/evidence.json` 冻结五条原始日志的指标、实际参数、episodes 配置中的奖励类型、checkpoint 目录名，以及源文件 SHA256。`data/metrics.csv` 提供逐条指标与原始日志行号。奖励均按 episodes 配置读取为 `acquisition`，不沿用 PR 中其他运行或其他版本的奖励标签。

原始日志位于 `/projects/public/u6gb/kangli/large-discovery-model/ldm_rl/runs/`。CSV 的 index 是日志原始的零起始索引；图中采样／优化序号加一。来自 `perf` 和 `rollout` 的记录按显式轮次 ID 连接，不按文本出现位置配对。未记录值保留为空；非有限梯度在 JSON 中以 `null` 加 `nonfinite: true` 表示。

图 2.3 使用始终记录的 `count_no_gradient` 确定分母，并对正计数逐轮核对 `count_lt_eps` 和 `count_0.0`。这里只能说这两条运行中计数对应全零组；不能推广为所有历史日志的计数等价。

图 2.4 的“未找到”限于列明的检查范围，不是全服务器普查。2026-09-05 的旧审计仅用于评测协议与当时状态，不沿用其“9B 从无有限梯度”的过时断言。搜索预算 80 沿用原计划，本报告不额外断言历史执行中缓存命中、重试或失败是否计入预算。

补充来源：[训练协议](../../slime_launch/TRAINING_PLAN.md)、[主线 PR #2](https://github.com/KangOxford/Large-Discovery-Models/pull/2)、[SkyRL PR #5](https://github.com/KangOxford/Large-Discovery-Models/pull/5)、[奖励 PR #6](https://github.com/KangOxford/Large-Discovery-Models/pull/6)。

在本目录执行：

```bash
python build_report.py
jupyter nbconvert --to notebook --execute --inplace experiment_progress.ipynb
```

依赖：Python、NumPy、Matplotlib、nbformat；执行 Notebook 还需 Jupyter/ipykernel。字体使用 Noto Sans CJK SC，若本机没有，脚本从 notofonts/noto-cjk 下载并缓存，不包含 GPU 任务。

重新从原始目录提取（仅在需要刷新证据时）：

```bash
python build_report.py --extract /path/to/ldm_rl
```

有限梯度、非零奖励、保存存档分别只证明各自的观测事实。报告不以任何单项指标替代 RL 对 SFT 的正式效果结论。
