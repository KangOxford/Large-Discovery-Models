# nanoGPT RL 训练指南（从零到开跑）

一条线性的 runbook。奖励是**实测的 `val_bpb`**：每一个 RL 环境 step 都真的训练一个
nanoGPT 跑满它的 wall-clock 预算，读它自己打印的 `val_bpb`。**没有代理模型参与打分。**

- 装环境细节、硬约束的完整解释、已知问题 → `NANOGPT_HANDOFF.md`
- 跑哪些实验、怎么评测 → `NANOGPT_RUNS.md`
- 本文只回答一件事：**照着敲，怎么跑起来。**

---

## 0. 30 秒版本

```bash
export REPO_ROOT=/path/to/Large-Discovery-Models
export WORK_DIR=/scratch/nanogpt_rl
export PYTHONPATH=$REPO_ROOT/rl:$REPO_ROOT

# ① 无 GPU 冒烟（~10 秒 + ~5 秒）
cd $REPO_ROOT
python -m pytest tasks/nanogpt/tests rl/ldm_rl/tests -q
python -m tasks.nanogpt.scripts.rl_smoke

# ② 一次性准备：数据 → FA3 → 奖励参考点 → episode 数据
export NANOGPT_EVAL_GPUS=4,5,6,7
bash rl/slime_launch/prepare_nanogpt.sh

# ③ 训练
export MEGATRON_ROOT=/path/to/megatron-lm
export MODEL_HF=/path/to/Qwen3.5-9B
export MODEL_REF=$WORK_DIR/qwen3.5-9B_torch_dist
export MODEL_ARGS_SCRIPT=$REPO_ROOT/rl/slime/scripts/models/qwen3.5-9B.sh
export EPISODES=$WORK_DIR/rl_episodes_nanogpt.jsonl
export TRAIN_GPUS=0,1,2,3
bash rl/slime_launch/run_train_real_nanogpt.sh
```

**在往下走之前，先记住这一条**（其余的都能边跑边查）：

> **训练器和评估器必须占不同的 GPU。** Megatron actor + sglang 会把
> `TRAIN_GPUS` 全程占住，所以评估不能借它们的卡。启动脚本会检查两者不相交，
> 重叠就直接报错退出。

---

## 1. 前置

| 项 | 要求 |
|---|---|
| GPU | ≥2 张（1 训练 + 1 评估）能跑通；**推荐 ≥8 张**（4+4），评估池线性扩展 |
| 单卡显存 | 80 GB（H100）。评估跑的是真 nanoGPT，配置过大会 OOM（会被记成 `oom`，不致命） |
| CUDA | 驱动决定轮子后缀。**torch 只能 `+cu128 / +cu129`；装成 `+cu130` 会让 `cuda.is_available()` 直接 False** |
| 磁盘 | ~150 GB（模型 + Megatron torch_dist + 数据 1.5 GB + run 目录） |
| 网络 | 首次准备需要能连 HuggingFace（数据 + FA3 kernel） |
| 其他 | `uv`（评估器用它调真 trainer 的依赖组） |

装 slime / Megatron / sglang 那一套和小分子任务完全一样，照 `../SLIME_TRAINING.md`
走（含那 6 个 patch）。**nanoGPT 的 RL 环境本身很轻**，只要 `numpy` + `ldm_tts`，
重的 torch 栈在评估器 spawn 的子进程里。

---

## 2. 第一步：无 GPU 冒烟（别跳过）

这一步不需要 GPU、不需要数据、不需要模型，几秒钟就能告诉你代码是不是好的。

```bash
cd $REPO_ROOT
export PYTHONPATH=$REPO_ROOT/rl:$REPO_ROOT
python -m pytest tasks/nanogpt/tests rl/ldm_rl/tests -q
```

**应该看到 `162 passed`。** 这里面覆盖了动作空间、约束、对**真实 `train.py`** 的
patch、指标解析、失败分类、缓存、GPU 锁、奖励的路径无关性、GP 选择方向、并发重叠。

然后跑一个完整的 mock episode：

```bash
python -m tasks.nanogpt.scripts.rl_smoke
```

应该看到 5 轮，依次演示：正常测量 → 正常测量 → 一个非法配置被
`rejected_by_trainer` → 一个重复配置被 `already_evaluated` → 夹在推理文字里的 JSON
被严格 parser 拒掉。

> **mock 的 `val_bpb` 是哈希，没有任何信息量，这是刻意的。** 这个任务曾经带过一个
> "看起来很合理"但从未拟合的解析式 val_bpb，有人拿它训练了，最后实测发现它的排序
> 是**反的**。所以 mock 被做成学不到东西的样子，免得再被误当成结果。

---

## 3. 第二步：一次性准备

```bash
export REPO_ROOT=/path/to/Large-Discovery-Models
export WORK_DIR=/scratch/nanogpt_rl        # run 目录 / 缓存 / episode 数据都在这
export NANOGPT_EVAL_GPUS=4,5,6,7           # 物理卡号，必须和 TRAIN_GPUS 不相交
bash $REPO_ROOT/rl/slime_launch/prepare_nanogpt.sh
```

四个阶段，每个阶段应该看到什么：

| 阶段 | 做什么 | 判断成功 | 耗时 |
|---|---|---|---|
| `[1/4]` | 下 16 个 parquet shard + 训 8192 词表 BPE | `$AUTORESEARCH_CACHE_DIR` 下约 1.5 GB | 几分钟~几十分钟（看网速） |
| `[2/4]` | 预取 FlashAttention-3（Hopper） | 打印 `FA3 cached` | 1-2 分钟 |
| `[3/4]` | **真训练默认配置，测出奖励参考点** | 打印 `val_bpb=0.99xxxx` | 每个 budget 约 `budget + 40s` |
| `[4/4]` | 生成 episode prompt 数据 | 打印实例分布和成本估算 | 秒级 |

`[2/4]` 必须在**有外网的节点**上做——没有 egress 的计算节点拉不到那个 kernel。

`[3/4]` 是最容易被误解的一步，值得解释：

> 奖励是"比一个**固定**参考配置低多少 bpb"。这个参考点必须先测出来。
> 如果不给，`improvement` 会拿 **episode 自己第一次测量**当基准，于是每轮增益
> telescope 成 `(第一次 − 最好)` —— **策略会因为"开局故意提一个烂配置、然后把它修好"
> 而拿到更多奖励**。给了固定参考点之后它是 incumbent 的每轮下界，episode 总奖励变成
> `max(0, 参考 − 最好)`，与提出顺序无关。
>
> 因为 pinned knob 保持默认值，**所有实例的参考配置是同一个**，所以这只花
> **每个 budget 一次**真训练，而且结果进缓存，重跑免费。
>
> `gen_rl_episodes.py` 在没有参考点时**默认拒绝生成 episode**。这是故意的。

跑完你会有：

```
$WORK_DIR/nanogpt_reference.json          参考点
$WORK_DIR/rl_episodes_nanogpt.jsonl       episode prompt 数据
$WORK_DIR/eval_cache.jsonl                共享评估缓存（已含参考点那次）
$WORK_DIR/runs/<hash>/                    每次评估真正执行的 train.py + 日志
$WORK_DIR/gpu_locks/                      GPU 占用锁
```

---

## 4. 第三步：转模型权重

slime 训练要 Megatron 的 `torch_dist` 格式：

```bash
cd $REPO_ROOT/rl/slime_launch
MODEL_HF=/path/to/Qwen3.5-9B SAVE=$WORK_DIR/qwen3.5-9B_torch_dist bash convert_9b.sh
```

从 SFT 起点训就再转一次（`Yangtze-ailab/LDM-CoT-SFT-Qwen3.5-9B-MixedScience`）。

---

## 5. 第四步：开始训练

```bash
export MEGATRON_ROOT=/path/to/megatron-lm
export MODEL_HF=/path/to/Qwen3.5-9B
export MODEL_REF=$WORK_DIR/qwen3.5-9B_torch_dist
export MODEL_ARGS_SCRIPT=$REPO_ROOT/rl/slime/scripts/models/qwen3.5-9B.sh
export EPISODES=$WORK_DIR/rl_episodes_nanogpt.jsonl
export TRAIN_GPUS=0,1,2,3
export NANOGPT_EVAL_GPUS=4,5,6,7
export SAVE=$WORK_DIR/nanogpt_R1_base            # 可选
export WANDB_KEY=...                              # 可选

bash $REPO_ROOT/rl/slime_launch/run_train_real_nanogpt.sh
```

**它会在真正开跑前先把这次运行的成本打出来**，看一眼再决定：

```
 episodes per step     : 8
 real runs per step    : <= 32
 evaluation GPUs       : 4
 wall clock per step   : ~45 min (before cache hits)
 total for  50 steps  : ~38 h
```

Slurm 提交大致是：

```bash
#!/bin/bash
#SBATCH -N 1 --gres=gpu:8 -t 48:00:00 -J nanogpt-rl
export REPO_ROOT=... WORK_DIR=... MEGATRON_ROOT=...
export TRAIN_GPUS=0,1,2,3 NANOGPT_EVAL_GPUS=4,5,6,7
export EPISODES=$WORK_DIR/rl_episodes_nanogpt.jsonl
srun bash $REPO_ROOT/rl/slime_launch/run_train_real_nanogpt.sh
```

---

## 6. 怎么判断"真的在训练"

**`rollout` 跑完不等于训练跑通。** 按这个顺序看：

1. **`train/grad_norm`** ← 先看这个。**nan 意味着这步更新被跳过了。**
   目前 Qwen3.5-9B（hybrid：Gated DeltaNet + full-attention + MTP）
   **从来没有记录到一个有限的 optimizer step**，nan 来自某个 full-attention 层的
   backward。dense 1.5B 能干净训练。这与本任务的 RL 组件无关（小分子那条线也卡在这），
   但会同样卡住你 —— 见 `NANOGPT_HANDOFF.md` §5.1，先用 dense 模型验证整条链路。
2. **组内 reward std**：GRPO 拿它归一化 advantage。这里奖励经常恰好是 0（没改进），
   所以零方差组是真实风险。如果常见，提高 `rollout_temperature`。
3. **`eval_cache.jsonl` 的失败分布**：`rejected_by_trainer` 占比高 = 策略还没学会
   整除约束；`oom` 高 = 在提过大的模型。
4. **缓存命中率**：上升是正常的收敛信号；接近 1 说明策略已经塌到少数几个配置上了。

一行命令看评估侧的全貌：

```bash
python3 - <<'PY'
import json, collections
rows = [json.loads(l) for l in open("$WORK_DIR/eval_cache.jsonl")]
ok = [r["metrics"]["val_bpb"] for r in rows if r["ok"]]
print(f"total {len(rows)}  ok {len(ok)}  hit-rate n/a (cache is by config)")
print(collections.Counter(r["failure_kind"] for r in rows if not r["ok"]))
if ok:
    s = sorted(ok)
    print(f"best {s[0]:.6f}  median {s[len(s)//2]:.6f}  worst {s[-1]:.6f}")
PY
```

**读数字时记住一件事**：同配置重复测量的实测标准差是 **0.0013 bpb**，而 upstream
一整轮 99 次真实搜索的总增益只有 **0.0044**。所以**小于 0.0013 的差异不要解读**，
最终候选必须重复测量（重复跑同一配置要先删掉它在 `eval_cache.jsonl` 里的那行，
否则会命中缓存）。

---

## 7. 排错

| 现象 | 原因 / 怎么办 |
|---|---|
| `ERROR: TRAIN_GPUS and NANOGPT_EVAL_GPUS overlap on [...]` | 按提示改成不相交。共享一张卡 = 300s 训练任务和 actor/sglang 抢显存，会 OOM |
| `ERROR: EPISODES points at a missing path` | 先跑 `prepare_nanogpt.sh` |
| `no measured reference for budget(s) 300` | 先跑 `[3/4]`（或 `rl_reference.py`）。**不要**用 `--allow-missing-reference` 绕过，那会让奖励可被 sandbag |
| 参考点测量失败 `could not launch 'uv'` | 装 `uv`，然后**直接重跑**。启动失败被分类为 `launch_failed` 且**不进缓存**，所以修好环境就能恢复 |
| 全部评估都 `rejected_by_trainer` | 大概率 `DEVICE_BATCH_SIZE=96` —— 它除不尽任何一个可用的 `TOTAL_BATCH_SIZE`，永远非法。策略需要时间学会 |
| 大量 `oom` | 策略在提过大的模型。**这是预期行为，不是 bug**：代码故意不做显存前置过滤（之前那个解析式估计低估实测峰值 5–31 倍，作为闸门会永久扭曲动作空间），让现实说话，OOM 只花 ~20 秒 |
| 大量 `timed_out` | 调大 `evaluation.timeout_slack`；超时**不进缓存**（可能是节点偶发问题） |
| episode 很早就 `empty_reservoir_limit` 结束 | 实例的动作空间被搜完了。调大 `--iterations`/`--reservoir-size` 时生成器的最小空间阈值会自动跟着变；`batching` 组单独只有 9 个合法配置，生成器会自动把它和别的组合并 |
| 每个 step 慢到不合理，且只有一张评估卡在忙 | 检查 `bridge.py` 里 `env.step` 还是不是 `await asyncio.to_thread(...)`。同步调用会卡死 event loop，让所有 episode 串行化 |
| 训练启动就 `cuda.is_available()=False` | torch 装成 `+cu130` 了，换 `+cu128/+cu129` |
| `import inspect` 之类莫名报错 | 不要在 `/tmp` 下 cwd 运行（那里可能有文件遮蔽标准库） |

---

## 8. 调规模和成本

一个 rollout step 的真实训练次数 =
`rollout_batch_size × n_samples_per_prompt × iterations × evaluations_per_round`
（默认 `2 × 4 × 4 × 1 = 32`）。

| 评估 GPU | 每 step | 50 step |
|---|---|---|
| 4 | ~45 min | ~38 h |
| 8 | ~23 min | ~19 h |
| 16 | ~11 min | ~9 h |

上限值；相同配置由 `eval_cache.jsonl` 服务，策略越收敛实际越便宜。

想改，按影响从大到小改 `rl/slime_launch/config_nanogpt.json`：
`episodes.iterations` → `training.n_samples_per_prompt` →
`training.rollout_batch_size` → `evaluation.budgets`。

> **`iterations` 不只是成本旋钮**：它就是"这个 episode 允许搜几步"，调小会直接削弱
> 任务本身。要省钱优先动 `n_samples_per_prompt`。

改完 `episodes.*` 或 `evaluation.*` 需要**重新生成 episode 数据**
（重跑 `prepare_nanogpt.sh` 即可，前两步会自动跳过）。

多 seed：`gen_rl_episodes.py --seed-offset N` 重新生成 episode 数据。slime 自己的
`--seed` 只管框架侧随机性，**不会换实例轨迹**。

---

## 9. 文件地图

```text
tasks/nanogpt/core/
  rl_knobs.py    动作空间：schema、镜像 train.py 的几何量、整除约束、规范化标识
  rl_eval.py     真实评估器：物化 → 运行 → 解析；缓存 / GPU 池 / 失败分类
  rl_encoder.py  17 维 GP 特征（归一化 knob + 派生几何量）
  rl_task.py     LDMTaskSpec、严格 parser、分级候选域
  rl_real.py     real 模式装配（含把 GP 选择方向修正过来的 selector）
  rl_adapter.py  mock / real 分发
tasks/nanogpt/scripts/
  rl_smoke.py         无 GPU 冒烟（一个完整 mock episode）
  rl_reference.py     测量奖励参考点
  gen_rl_episodes.py  生成 episode prompt 数据
tasks/nanogpt/tests/  100 个测试，全部无需 GPU
rl/slime_launch/
  config_nanogpt.json         所有旋钮 + 每个旋钮的代价说明
  prepare_nanogpt.sh          一次性准备
  run_train_real_nanogpt.sh   训练启动器
  NANOGPT_RL_TRAINING.md      本文
  NANOGPT_HANDOFF.md          装环境 / 硬约束 / 已知问题（完整版）
  NANOGPT_RUNS.md             运行矩阵 / 评测协议
```

改 `rl_encoder.py` 的特征布局或归一化时**必须同时 bump `FEATURE_VERSION`**：
在一个版本的向量上拟合的 GP 绝不能喂另一个版本的向量（`LDMEnv` 在 task spec 和
encoder 声明不一致时会直接拒绝构建）。
