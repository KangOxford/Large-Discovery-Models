# 小分子 RL 交接指南（装环境 · 跑 · 训练计划）

从零把 **小分子 acquisition-RL（GRPO on Slime）** 跑起来。运行矩阵见同目录 `RUNS.md`。
**方向：train KRAS G12D → eval G12C & G12D**（G12D 评测器现成；G12C 只在离线评测时用）。

---

## A. 装环境（最关键，别跳）

> ⚠️ 必须是**配平的 torch 2.11 + 匹配 TE** 栈，否则 GRPO backward 会 SIGSEGV。**强烈建议直接用我们的 Docker 镜像**，不要自己从头 pip 装。

### A1. 代码
```bash
git clone --recurse-submodules -b rl https://github.com/YihangChen9/Large-Discovery-Models.git LDM
cd LDM
git submodule update --init rl/slime rl/megatron-lm
cd rl/megatron-lm && git checkout 1dcf0dafa884ad52ffb243625717a3471643e087 && cd ../..
```

### A2. 环境镜像（三选一，优先第 1）
> `image.yangtzeailab.com` 是**私有 Harbor**(不能匿名 pull),需先登录。
1. **拿到 Harbor 凭据(推荐:pull-only robot token)** → 登录后拉：
   ```bash
   docker login image.yangtzeailab.com -u <robot名> -p <robot-token>
   docker pull image.yangtzeailab.com/opensandbox/ldm-slime-rl:latest
   ```
2. **没凭据但拿到镜像 tar** → 导入(有权限方 `docker save ... | gzip` 导出,约十几~几十 G)：
   `gunzip -c ldm-slime-rl.tar.gz | docker load`
3. **都不行** → 用 `rl/slime/build_conda.sh` 重建（torch 2.11 + TE 2.16.1 + 匹配 megatron/sglang commit），再按 `rl/SLIME_TRAINING.md` 打 6 个 patch。**最容易踩坑，尽量避免。**

镜像里已备：conda env `/root/micromamba/envs/slime`、Megatron `/root/megatron-lm`、torch 2.11+cu129、匹配 TE、rdkit/meeko（docking 用）。

### A3. 起容器（挂代码 + 资产盘 + GPU）
```bash
docker run --gpus all -it --shm-size=32g \
  -v /path/to/LDM:/mnt/data0/ys/LDM \
  -v /path/to/assets:/mnt/data0 \
  image.yangtzeailab.com/opensandbox/ldm-slime-rl:latest bash
```
> 脚本里的路径都写死在 `/mnt/data0/...`，把资产放到对应位置最省事（见 B）。

---

## B. 资产（放到这些路径）

| 资产 | 路径 | 来源 |
|---|---|---|
| base 模型 Qwen3.5-9B | `/mnt/data0/hf_models/models/Qwen3.5-9B` | HF `Qwen/Qwen3.5-9B` |
| SFT no-GP 模型 | `/mnt/data0/hf_models/models/LDM-CoT-SFT` | HF `Yangtze-ailab/LDM-CoT-SFT-Qwen3.5-9B-MixedScience` |
| 冒烟 Qwen2.5-1.5B | `/mnt/data0/hf_models/models/Qwen2.5-1.5B-Instruct` | HF |
| vina 二进制 | `/mnt/data0/dock-project/bin/vina` | 随 assets 包 |
| G12D 活性模型 | `tasks/small_molecule/resources/models/best_g12d_model.joblib` | **已在 repo** |
| 8UN5 受体（制备好） | `/mnt/data0/ys/LDM/docking_work/receptors/…` | 随 assets 包 / 或 meeko 从 RCSB 8UN5 重制备 |

下模型：
```bash
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen3.5-9B --local-dir /mnt/data0/hf_models/models/Qwen3.5-9B
hf download Yangtze-ailab/LDM-CoT-SFT-Qwen3.5-9B-MixedScience --local-dir /mnt/data0/hf_models/models/LDM-CoT-SFT
```

---

## C. 准备（一次）
```bash
cd /mnt/data0/ys/LDM/rl/slime_launch
# 1) 转 Megatron torch_dist（base + SFT 各一次）
MODEL_HF=/mnt/data0/hf_models/models/Qwen3.5-9B  SAVE=/mnt/data0/ys/LDM/rl/qwen3.5-9B_torch_dist      bash convert_9b.sh
MODEL_HF=/mnt/data0/hf_models/models/LDM-CoT-SFT SAVE=/mnt/data0/ys/LDM/rl/qwen3.5-9B-sft_torch_dist  bash convert_9b.sh
# 2) 生成 4 个 run 的 episodes（reward 已烤进数据）
bash gen_episodes_runs.sh
# 3) 暖共享 GP（rollout-only，dock warmup.num_samples 个分子）
bash run_warmup_real_slime.sh
```

---

## D. 怎么跑（4 个 run，各 4 卡 / TP=2）
命令全在 **`RUNS.md`**（R1 base·acq-max / R2 SFT·acq-max / R3 SFT·真reward / R4 SFT·acq-mean）。每个 run 建议 **≥3–5 seed** 出误差棒。开 wandb 加 `export WANDB_KEY=<key>`。

---

## E. 训练计划（分阶段，带验证闸）

**P0 — 环境 & backward 验证（1.5B，最便宜，先做这个）**
用现成 `qwen2.5-1.5B_torch_dist_te` 跑 `run_train_real_shared.sh`（3 卡）。
✅ 通过标准：GRPO **backward 不 SIGSEGV**、reward/loss 有正常值、走过 ≥5 rollout step、能 `--save`。这一步同时验证 docking + acquisition reward + bridge 全链路。

**P1 — 9B 冒烟（mock，验 hybrid）**
转好 9B torch_dist 后，把 episodes 改 `--mode mock` 跑 `run_train_real_9b.sh`。
✅ 通过标准：9B hybrid **backward 能过**、不 OOM、reward 会动。专治"Qwen3.5 hybrid 未验证"。

**P2 — 真训练（4 个 run × seed，real）**
按 `RUNS.md` 起 R1–R4，全部 **train G12D**。
- R1 vs R2：base-RL vs SFT-RL（SFT 打底有没有用）
- R2 vs R3 vs R4：reward 消融（acq-max / 真结果 / acq-mean）

**评测（train D → 测 C & D）**
取各 run 的 best checkpoint，用离线 eval harness 在 **G12C 和 G12D** 上跑 LDM loop（budget 80，5 seed），比 hypervolume。参照行：base（无RL）、SFT（无RL）。
> G12C 评测器 = 原 G12C campaign 的 G12C 受体 + G12C 活性模型（论文 G12C 那列跑的那套）。

---

## F. 已知坑
- **环境 ABI**：非配平 torch/TE → backward SIGSEGV。务必用镜像。
- **9B 是 hybrid**（线性注意力+MTP）：转换/训练用 `qwen3.5-9B.sh` 的 spec（脚本已 source）；先 mock 冒烟。
- **显存**：9B+TP=2+sglang 于 4×80G；OOM 就升 TP 或降 `max_tokens_per_gpu`（recompute-full 已开）。
- **docking 吞吐**：同步、`vina_max_workers=1` 会堵 rollout；`config_real.json` 里调大 workers + 开缓存。
