# nanoGPT RL 交接指南（真实 reward）

把 **nanoGPT 超参搜索的 GRPO 训练**从零跑起来。奖励是**真实测量的 `val_bpb`**：
每一个环境 step 都真的训练一个 nanoGPT 跑满它的 wall-clock 预算。

> 先读 §1 的三条硬约束，再读别的。它们不是建议，是这个任务和其他 LDM 任务在
> 结构上的区别。

---

## 0. 先明确：什么已验证，什么没有

诚实的边界，免得你按错误的预期排查问题。

| 环节 | 状态 |
|---|---|
| 动作空间 / 约束 / 规范化标识 | ✅ 161 个测试，含对**真实 `train.py`** 的 patch 验证 |
| 真实评估器（物化→运行→解析→分类→缓存→GPU 池） | ✅ 用假 trainer 完整验证；**真 trainer 从未在本机跑过** |
| GP 特征编码 + 选择方向 | ✅ 单测覆盖，含方向 bug 的回归测试 |
| 奖励语义（路径无关性） | ✅ 单测断言不变量 + 反向回归守卫 |
| mock 模式端到端 episode | ✅ |
| 并发 episode 重叠评估 | ✅（0.35s vs 串行 1.22s） |
| **真实 300s 训练进 RL 循环** | ❌ **从未跑过**。需要 GPU + 数据 + slime 环境 |
| **9B hybrid 的 GRPO backward** | ❌ 见 §5，**这是已知未解决的拦路虎** |

---

## 1. 三条硬约束

### 1.1 训练器和评估器必须占**不同**的 GPU

奖励要真训练，而 Megatron actor + sglang 会把分给它们的卡**全程**占住。所以评估
不能"借"训练的卡——共享一张卡意味着一个 300s 训练任务去和 actor 权重、sglang 的
KV cache 抢显存，结果是其中一个（或两个）OOM。

```bash
TRAIN_GPUS=0,1,2,3  NANOGPT_EVAL_GPUS=4,5,6,7    # 8 卡节点
```

`run_train_real_nanogpt.sh` 会在启动前检查两者不相交，重叠就直接报错退出。

评估池是**线性扩展**的，卡多就直接加进 `NANOGPT_EVAL_GPUS`（见 §4 的账）。

### 1.2 奖励的参考点必须先测出来

奖励是 `improvement`：**比一个固定参考配置低多少 bpb**。这个参考点必须在训练前
测好并写进 episode 数据。

如果不给参考点，`improvement` 会拿 **episode 自己第一次测量**当基准，于是每轮的
增益 telescope 成 `(第一次 − 最好)` —— **策略会因为"开局故意提一个烂配置然后修好"
而拿到更多奖励**。给了固定参考点之后，参考点是 incumbent 的下界，episode 总奖励
变成 `max(0, 参考 − 最好)`，与提出顺序无关，sandbagging 收益归零。

`prepare_nanogpt.sh` 的第 3 步负责测它。`gen_rl_episodes.py` 在没有参考点时
**默认拒绝生成 episode**（要覆盖得显式加 `--allow-missing-reference`）。

> 这和 `rl_episodes_sm_R3` 那个 ΔHV 的 moving-nadir 是同一类缺陷、同一个解法。

### 1.3 不存在 val_bpb 的解析式代理

这个任务里**只有实测的分数**。`rl_knobs.py` 有一个测试专门断言它不导出任何名字里
带 `bpb` 的东西。

原因是有过一次真实教训：这个任务此前带过一个"看起来很合理"但从未拟合的解析式
val_bpb 公式，有人拿它训练了，最后测出来它的**排序是反的**（Spearman −0.60；
7 个策略提案里 6 个是 17–50σ 的实质回退，而公式认为完全 inert 的那个 knob
实际代价 0.023 bpb）。所以 `mock` 模式的分数是**故意做成信息无关的**（canonical
key 的哈希），它只能验管道，学不到任何东西——这样没人会把 mock 结果误当成结果。

---

## 2. 装环境

### 2.1 训练栈（slime / Megatron / sglang）

这部分和小分子任务完全一样，照 `SLIME_TRAINING.md` 走，包括那 6 个 patch 和
**CUDA 红线**（torch 只能 `+cu128/+cu129`；`+cu130` 会让 `cuda.is_available()`
直接 False）。

### 2.2 环境侧（很轻）

nanoGPT 的 RL 环境只需要 `numpy` + `ldm_tts`，**不需要** torch/gpytorch/rdkit。
重的 torch 栈在评估器 spawn 的 trainer 子进程里，不在环境进程里。

所以 episode 数据里的 `task_python` 应该就填**跑 slime 的那个解释器**
（`prepare_nanogpt.sh` 用 `sys.executable` 自动填好了）。

> 必须显式填。`bridge.generate` 对 real 模式一律建 `RemoteLDMEnv`，而它的
> `task_python` 默认值是一个小分子任务的 venv 路径，在这里不存在。

### 2.3 真实 trainer 的依赖

`train.py` 是 opt-in 依赖组（torch 2.9.1+cu128），评估器通过 `uv` 调它：

```bash
uv run --group train --project $REPO_ROOT/tasks/nanogpt python -u train.py
```

所以需要 `uv`，并且 `uv sync --group train` 能装成功。要换调用方式就设
`real_kwargs.run_command`。

`train.py` 还需要 **FlashAttention-3**（Hopper 路径，`kernels.get_kernel(
"varunneal/flash-attention-3")`），首次使用时从 HF hub 拉。**在有外网的节点上预取**
（`prepare_nanogpt.sh` 第 2 步做了），否则没有 egress 的计算节点会失败。

---

## 3. 怎么跑

```bash
export REPO_ROOT=/path/to/LDM
export WORK_DIR=/scratch/nanogpt_rl          # runs / 缓存 / episode 数据都在这
export NANOGPT_EVAL_GPUS=4,5,6,7

# 一次性：数据(16 shards + 8192 BPE, ~1.5G) -> FA3 预取 -> 参考点 -> episode 数据
bash $REPO_ROOT/rl/slime_launch/prepare_nanogpt.sh

# 训练
export MEGATRON_ROOT=/path/to/megatron-lm
export MODEL_HF=/path/to/Qwen3.5-9B
export MODEL_REF=/path/to/qwen3.5-9B_torch_dist        # convert_9b.sh 转出来的
export MODEL_ARGS_SCRIPT=$REPO_ROOT/rl/slime/scripts/models/qwen3.5-9B.sh
export EPISODES=$WORK_DIR/rl_episodes_nanogpt.jsonl
export TRAIN_GPUS=0,1,2,3
export WANDB_KEY=...                                    # 可选

bash $REPO_ROOT/rl/slime_launch/run_train_real_nanogpt.sh
```

启动脚本会在真正开跑前把这次运行的**成本**打出来，先看一眼再决定。

跑之前建议先做无 GPU 的冒烟：

```bash
cd $REPO_ROOT && PYTHONPATH=$PWD/rl:$PWD python -m pytest tasks/nanogpt/tests rl/ldm_rl/tests -q
```

---

## 4. 成本

一个 rollout step 的真实训练次数 =
`rollout_batch_size × n_samples_per_prompt × iterations × evaluations_per_round`。

默认配置是 `2 × 4 × 4 × 1 = 32` 次，每次约 `budget + 40s`：

| 评估 GPU 数 | 每 step | 50 step 合计 |
|---|---|---|
| 4 | ~45 min | ~38 h |
| 8 | ~23 min | ~19 h |
| 16 | ~11 min | ~9 h |

这是**缓存命中前**的上限。相同配置（跨 episode、跨 worker）由共享的
`eval_cache.jsonl` 服务，所以策略越收敛，实际开销越低。

想调成本，按影响从大到小：`episodes.iterations` → `training.n_samples_per_prompt`
→ `training.rollout_batch_size` → `evaluation.budgets`。

**注意 `iterations` 不只是成本旋钮**：它就是"这个 episode 允许搜几步"，调小会直接
削弱任务本身。

---

## 5. 已知问题

### 5.1 ⚠️ 9B hybrid 的 backward 会出 nan（未解决）

**Qwen3.5-9B（Gated DeltaNet + full-attention + MTP）从来没有记录到一个有限的
optimizer step**，nan 来自某个 full-attention 层的 backward。dense 1.5B 能干净训练。
这个问题**与本任务的 RL 组件无关**（小分子那条线也卡在这），但它会同样卡住你。

- 先用 dense 模型验证整条链路（`convert_dense7b.sh` +
  `run_train_real_dense7b.sh` 是为隔离这个问题准备的对照）。
- 只盯 `train/grad_norm`：nan 意味着这步更新被跳过了，rollout 跑完 ≠ 训练跑通。
- `SLIME_TRAINING.md` 里还记了一个 TE/torch ABI 不匹配导致 backward SIGSEGV 的
  问题，和上面这个不是一回事，但同样发生在 backward。

### 5.2 信噪比：可达增益只有噪声底的约 3 倍

这是这个任务**科学上**最重要的限制，不是 bug：

- upstream 记录的一整轮真实搜索（99 次真训练）总共只把 val_bpb 推进了 **0.0044**
- 同配置重复测量的实测标准差是 **0.0013**（n=4 的 baseline 复现，另有 3 个重复
  配置的合并 within-config std）

也就是说**单次 `val_bpb` 里有相当一部分是测量噪声**，而 episode 奖励是由单次测量
算出来的。后果：

- 策略会部分学到"利用测量噪声"，因为 best-of-N 天然有向上的噪声偏差；
- 别把小于 ~0.0013 的 `val_bpb` 差异读成真实改进；
- 报告结论时，最终候选**必须重复测量**（重复跑同一个配置需要先删掉它在
  `eval_cache.jsonl` 里的那行，否则会命中缓存）。

能缓解的方向：加大 `evaluation.budgets`（相对噪声下降）、对最终候选做重复测量、
或者接受它并且只在 σ 单位下汇报。**注意给奖励乘一个全局系数是无用的**——GRPO 用
组内 std 归一化 advantage，纯乘性缩放是恒等变换。

### 5.3 动作空间会被耗尽

`batching` 组单独放开只有 **9 个合法配置**（`DEVICE_BATCH_SIZE=96` 除不尽任何一个
可用的 `TOTAL_BATCH_SIZE`）。这比一个 episode 要提的提案数还少，reservoir 会因为
跨轮去重而变空，episode 提前以 `empty_reservoir_limit` 结束。

`gen_rl_episodes.py` 会自动把这类过小的组**合并**（而不是丢掉，那样策略就永远学不
会设这两个 knob）。改 `--iterations`/`--reservoir-size` 时这个阈值会自动跟着变。

### 5.4 显存没有前置过滤

`check_hard_constraints` **只**执行 `train.py` 真正 assert 的规则（批大小整除）。
**故意没有** VRAM 预筛：此前那个解析式显存估计低估实测 `peak_vram_mb` 达 5–31 倍
（活化项斜率差约 500 倍），作为闸门它一边声称限定了可行域、一边放进了会 OOM 的
配置。错误的闸门会**永久扭曲动作空间**，而真的 OOM 很便宜（~20s）且会被记成
`failure_kind="oom"`。所以让现实说话，OOM 率从结果里读，别信模型。

### 5.5 其他

- **response 长度**：实测 reset prompt 940–1780 token，4 轮 transcript 再加约 1300，
  所以 `rollout_max_response_len=4096` 大约是 2 倍余量。`bridge.py` 关掉了 thinking
  （`enable_thinking=False`）；**如果重新打开，这个值要大幅提高**。
- **超时**：默认允许 `budget + 300s`。实测启动+编译+末尾 val 是 28–51s，300s 很宽松，
  但挂住一个 GPU 很贵。
- **缓存会记住失败**：OOM / trainer 拒绝 / 发散都会被缓存（它们必然重现）。
  只有 `timed_out` 不缓存，因为可能是节点的偶发问题。
- `/tmp` 里任何 `inspect.py` 之类的文件会遮蔽标准库；不要在 `/tmp` 下 cwd 运行。

---

## 6. 代码在哪

```text
tasks/nanogpt/core/
  rl_knobs.py    动作空间：schema、镜像 train.py 的几何量、整除约束、规范化标识
  rl_eval.py     真实评估器：物化 -> 运行 -> 解析；缓存 / GPU 池 / 失败分类
  rl_encoder.py  17 维 GP 特征（归一化 knob + 派生几何量）
  rl_task.py     LDMTaskSpec、严格 parser、分级的候选域
  rl_real.py     real 模式装配（含把 GP 方向修正过来的 selector）
  rl_adapter.py  mock / real 分发
tasks/nanogpt/scripts/
  rl_reference.py     测量奖励参考点
  gen_rl_episodes.py  生成 episode prompt 数据（实例采样）
tasks/nanogpt/tests/  89 个测试，全部无需 GPU
rl/slime_launch/
  config_nanogpt.json           所有旋钮 + 每个旋钮的代价说明
  prepare_nanogpt.sh            一次性准备
  run_train_real_nanogpt.sh     训练启动器
```

改了 `rl_encoder.py` 的特征布局或归一化，**必须**同时 bump `FEATURE_VERSION`：
在一个版本的向量上拟合出来的 GP 绝不能喂另一个版本的向量（`LDMEnv` 会在
task spec 和 encoder 的声明不一致时直接拒绝构建）。
