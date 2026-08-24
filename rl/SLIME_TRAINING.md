# Slime 完整 rollout / 训练流水线 · Slime End-to-End Rollout/Training Guide

在服务器 `<server-host>:<ssh-port>`、路径 `<repo-root>`（下文以 `/mnt/data0/ys/LDM`
为例）上，用 **Qwen2.5-1.5B-Instruct**（标准 transformer 架构）跑 Slime + Megatron-LM
的 rollout 训练流水线。所有路径/IP/用户名都是示例，请按你的实际环境替换。

> This guide reproduces the Slime + Megatron-LM rollout/training pipeline on
> server `<server-host>:<ssh-port>`, path `<repo-root>` (illustrated below as
> `/mnt/data0/ys/LDM`), using **Qwen2.5-1.5B-Instruct** (standard transformer
> architecture). All paths/IPs/usernames are examples — replace them with your
> real environment.

> Qwen3.5 系列是 hybrid（Gated DeltaNet 线性注意力）+ MTP 架构，虽然
> `rl/megatron-lm` 和 slime 都内置了支持，但需要额外的 hybrid layer spec /
> pattern 定制。**本指南先用标准架构的 Qwen2.5 验证流水线**，Qwen3.5 见文末"已知问题"。
>
> Qwen3.5 models are hybrid (Gated DeltaNet linear attention) + MTP. Although
> both `rl/megatron-lm` and slime have built-in support, they need extra hybrid
> layer spec / pattern customization. **This guide validates the pipeline with
> standard Qwen2.5 first**; see "Known issues" for Qwen3.5.

---

## 0. 前置环境 · Prerequisites

| 项 / Item | 值 / Value |
|---|---|
| Python 环境 / env | `/home/zsgpu/miniconda3/envs/agentic`（Python 3.12, torch 2.13+cu130, sglang 0.5.18, slime 0.3.1 editable → `rl/slime`） |
| Megatron-LM | `/mnt/data0/ys/LDM/rl/megatron-lm`（必须 checkout 到 slime 期望的 commit / must checkout the commit slime expects） |
| 模型 / model | `/mnt/data0/hf_models/models/Qwen2.5-1.5B-Instruct` |
| GPU | 任意空闲 GPU（转换 1 卡；rollout/训练按需）/ any free GPUs |

关键 `PYTHONPATH`（转换、训练、rollout 都要 / needed for convert/train/rollout）：

```bash
export PYTHONPATH=/mnt/data0/ys/LDM/rl/megatron-lm:/mnt/data0/ys/LDM/rl:/mnt/data0/ys/LDM:$PYTHONPATH
export PATH=/home/zsgpu/miniconda3/envs/agentic/bin:$PATH
```

---

## 1. 环境适配 · Environment Adaptations（6 个 patch）

slime 官方构建脚本 `rl/slime/build_conda.sh` pin 了 `MEGATRON_COMMIT=1dcf0dafa…`、
`SGLANG_COMMIT=0b3bb0cbe…`、`transformer_engine==2.16.1`。agentic 环境是另一组版本
（sglang 0.5.18、numpy 2.x），因此需要以下适配。全部改动都在服务器 `rl/slime/` 与
`rl/megatron-lm/`，备份在 `/tmp/*.bak`。

> slime's official build script `rl/slime/build_conda.sh` pins
> `MEGATRON_COMMIT=1dcf0dafa…`, `SGLANG_COMMIT=0b3bb0cbe…`, `transformer_engine==2.16.1`.
> The agentic env uses different versions (sglang 0.5.18, numpy 2.x), hence the
> adaptations below. All changes live under `rl/slime/` and `rl/megatron-lm/`,
> backed up in `/tmp/*.bak`.

### 1.1 Megatron-LM checkout 到 slime 期望的 commit · checkout to slime's expected commit

```bash
cd /mnt/data0/ys/LDM/rl/megatron-lm
git checkout 1dcf0dafa884ad52ffb243625717a3471643e087
```

否则 slime 的 `model_provider` 访问的 `moe_use_legacy_grouped_gemm` 等参数在新版
Megatron 中已改名/删除。

> Otherwise slime's `model_provider` accesses args like
> `moe_use_legacy_grouped_gemm` that newer Megatron renamed/removed.

### 1.2 绕过 numpy 1.x 保守断言 · bypass the numpy 1.x guard

`rl/slime/slime/backends/megatron_utils/initialize.py` 中原有 / originally had:

```python
assert np.__version__.startswith("1."), "Megatron does not support numpy 2.x"
```

该 fork 的 `pyproject.toml` 对 `numpy` 无版本 pin（接受 2.x），注释掉此断言即可。
降级 numpy 会破坏 sglang（`ml-dtypes`/`scipy` 要求 `numpy>=2.0`），不能靠降级。

> This fork's `pyproject.toml` does not pin `numpy` (accepts 2.x), so comment out
> the assert. Downgrading numpy breaks sglang (`ml-dtypes`/`scipy` require
> `numpy>=2.0`), so downgrade is not an option.

### 1.3 无 TE 时回退 local 实现 · fall back to local impl when TE is absent

`rl/slime/slime/backends/megatron_utils/arguments.py` 的 `_set_default_megatron_args`
开头加入 / prepend:

```python
try:
    import transformer_engine  # noqa: F401
except ImportError:
    if getattr(args, "transformer_impl", None) in (None, "transformer_engine"):
        logger.info("transformer_engine not installed; forcing transformer_impl='local'")
        args.transformer_impl = "local"
```

Megatron 默认 `transformer_impl="transformer_engine"`，无 TE 时会崩。

> Megatron defaults to `transformer_impl="transformer_engine"`, which crashes
> without TE.

### 1.4 local 实现的 Qwen 权重映射（双向）· local-impl Qwen weight mapping (both directions)

TE 实现把 input layernorm 融合进 qkv、post layernorm 融合进 fc1；local 实现是两个
独立成层。两个转换方向都要补 / TE fuses input layernorm into qkv and post layernorm
into fc1; local keeps them as separate layers. Both directions need patching:

- HF → Megatron：`rl/slime/slime/backends/megatron_utils/hf_to_megatron/qwen.py`
  的 `qwen_hf_tensor` mapping 补 / add:

  ```python
  "input_layernorm.weight": "input_layernorm.weight",
  ```

- Megatron → HF：`rl/slime/slime/backends/megatron_utils/megatron_to_hf/qwen2.py`
  的 `convert_qwen2_to_hf` 补 / add:

  ```python
  elif rest == "input_layernorm.weight":
      return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
  elif rest == "pre_mlp_layernorm.weight":
      return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
  ```

后者是 `update_weights`（训练后把 Megatron 权重推回 sglang）需要的。

> The latter is required by `update_weights` (pushing Megatron weights back to
> sglang after training).

### 1.5 sglang 0.5.18 的只读 ServerArgs · read-only ServerArgs in sglang 0.5.18

`rl/slime/slime/backends/sglang_utils/sglang_engine.py` 的 `launch_server_process` 中：

```python
try:
    server_args.host = server_args.host.strip("[]")
except AttributeError:
    pass  # sglang>=0.5.18 的 resolved ServerArgs 只读；IPv4 host 本就没有方括号
```

### 1.6 安装 Transformer Engine（真实训练必需）· install TE (required for real training)

rollout-only 可以用 local 实现（无 TE），但**真实 GRPO 训练**会强制
`variable_seq_lengths=True`（varlen），local 的 `DotProductAttention` 不支持 packed
sequence，必须装 TE。

> rollout-only works with local impl (no TE), but **real GRPO training** forces
> `variable_seq_lengths=True` (varlen); local `DotProductAttention` does not
> support packed sequence, so TE is required.

TE 2.18.0 有匹配 torch 2.13+cu130 的预编译 wheel，但 `transformer_engine_torch` /
`transformer_engine_cu13` 从 GitHub release 下载（服务器可能直连失败）。若直连失败，
用 pip 装的 `cuda-toolkit` + `nvidia-cudnn-cu13` 做源码编译，需把分散的 `nvidia/*`
头文件/库路径暴露给编译器：

> TE 2.18.0 has prebuilt wheels for torch 2.13+cu130, but
> `transformer_engine_torch`/`transformer_engine_cu13` download from GitHub
> release (may fail to connect). If direct download fails, build from source with
> pip's `cuda-toolkit` + `nvidia-cudnn-cu13`, exposing the scattered `nvidia/*`
> header/lib paths to the compiler:

```bash
NV=/home/zsgpu/miniconda3/envs/agentic/lib/python3.12/site-packages/nvidia
export CUDA_HOME=$NV/cu13
export PATH=$NV/cu13/bin:$PATH
export CPLUS_INCLUDE_PATH=$(find $NV -maxdepth 2 -name include -type d | tr "\n" ":")$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=$(find $NV -maxdepth 2 -name lib -type d | tr "\n" ":")$LIBRARY_PATH
pip install "transformer-engine[pytorch,core_cu13]==2.18.0"
```

> ⚠️ 见"已知问题"：TE 2.18.0 官方预编译 wheel 是为 **NVIDIA NGC torch 26.04** 编译的，
> 与 PyPI torch 2.13.0 ABI 不兼容；源码编译的 TE 能 import，但训练 backward 会 SIGSEGV。
> 真实训练的 backward 尚未跑通，根因是 TE/torch 版本 ABI 不匹配（非 RL 组件问题）。
>
> ⚠️ See "Known issues": TE 2.18.0 prebuilt wheels target **NVIDIA NGC torch 26.04**,
> ABI-incompatible with PyPI torch 2.13.0; source-built TE imports but SIGSEGVs on
> training backward. Real-training backward is not yet working; the root cause is
> TE/torch ABI mismatch (not an RL-component issue).

---

## 2. 权重转换（HF → Megatron torch_dist）· weight conversion

```bash
cd /mnt/data0/ys/LDM/rl/slime
export PYTHONPATH=/mnt/data0/ys/LDM/rl/megatron-lm:/mnt/data0/ys/LDM/rl:$PYTHONPATH
export PATH=/home/zsgpu/miniconda3/envs/agentic/bin:$PATH
CUDA_VISIBLE_DEVICES=2 python tools/convert_hf_to_torch_dist.py \
    --swiglu --num-layers 28 --hidden-size 1536 --ffn-hidden-size 8960 \
    --num-attention-heads 12 --use-rotary-position-embeddings --disable-bias-linear \
    --add-qkv-bias --normalization "RMSNorm" --norm-epsilon 1e-6 --rotary-base 1000000 \
    --group-query-attention --num-query-groups 2 --vocab-size 151936 \
    --no-rope-fusion --no-persist-layer-norm --no-gradient-accumulation-fusion \
    --hf-checkpoint /mnt/data0/hf_models/models/Qwen2.5-1.5B-Instruct \
    --save /mnt/data0/ys/LDM/rl/qwen2.5-1.5B_torch_dist
```

要点 / Notes:

- 模型配置来自 `rl/slime/scripts/models/qwen2.5-1.5B.sh`，但该脚本的
  `--rotary-base 10000` 是 typo，**必须按 config.json 的 `rope_theta=1000000`**。
- `--no-*fusion` 三个参数是 local（无 TE/APEX）实现所需的。若已装 TE，去掉
  `--no-rope-fusion --no-persist-layer-norm`，只保留 `--no-gradient-accumulation-fusion`
  （那是 APEX 依赖，非 TE）。
- 产物是 `/mnt/data0/ys/LDM/rl/qwen2.5-1.5B_torch_dist/release/`（约 3GB）。

> - Model config comes from `rl/slime/scripts/models/qwen2.5-1.5B.sh`, but its
>   `--rotary-base 10000` is a typo — **must use `rope_theta=1000000`** from config.json.
> - The three `--no-*fusion` flags are for local (no TE/APEX) impl. With TE installed,
>   drop `--no-rope-fusion --no-persist-layer-norm`, keep only
>   `--no-gradient-accumulation-fusion` (an APEX dependency, not TE).
> - Output is `/mnt/data0/ys/LDM/rl/qwen2.5-1.5B_torch_dist/release/` (~3GB).

---

## 3. prompt data

```bash
cd /mnt/data0/ys/LDM
PYTHONPATH=/mnt/data0/ys/LDM/rl:/mnt/data0/ys/LDM \
  /home/zsgpu/miniconda3/envs/agentic/bin/python \
  rl/ldm_rl/episodes.py --output rl_episodes_sm.jsonl \
  --task small_molecule --mode mock --count 8 --iterations 4 --reservoir-size 2
```

---

## 4. 启动 rollout-only（验证流水线）· launch rollout-only (validate pipeline)

见 `rl/slime_launch/run_rollout.sh`。核心是 `--debug-rollout-only`（只跑 rollout，
不训练，验证 ray + rollout manager + sglang engine + Megatron actor 加载权重 +
`bridge.generate` 全链路）。

> See `rl/slime_launch/run_rollout.sh`. The key is `--debug-rollout-only` (rollout
> only, no training), which validates the whole chain: ray + rollout manager +
> sglang engine + Megatron actor loading weights + `bridge.generate`.

## 5. 启动真实训练（GRPO）· launch real training

见 `rl/slime_launch/run_train.sh`。相对 rollout 脚本，去掉 `--debug-rollout-only`，
加上 GRPO / optimizer 参数。

> See `rl/slime_launch/run_train.sh`. Relative to the rollout script, drop
> `--debug-rollout-only` and add GRPO / optimizer args.

---

## 6. 验证状态 · Verification status

| 环节 / Stage | 状态 / Status |
|---|---|
| 权重转换 HF→torch_dist / weight conversion | ✅ 通过 / passed |
| rollout-only 完整流水线 / full rollout pipeline | ✅ 通过 / passed（`reward=16.6` 等） |
| bridge multi-turn `rollout_log_probs` 契约 / contract | ✅ 通过（修复了 env-token 0.0 丢失的 bug）/ fixed |
| ref_log_probs（ref forward）/ ref forward | ✅ 通过 / passed |
| actor forward + GRPO loss 前向 / actor forward + loss | ✅ 通过 / passed |
| update_weights（megatron → sglang）/ weight push | ✅ 通过 / passed |
| loss backward（梯度 + NCCL reduce_scatter） | ❌ SIGSEGV（TE/torch ABI 不匹配） |

> RL 主线（rollout → reward → loss 前向）已真实验证；只差 backward 一步。backward
> 的 SIGSEGV 是 TE 源码编译与 PyPI torch 2.13.0 的 toolchain ABI 不匹配（梯度内存损坏 →
> NCCL 同步崩溃），与 RL 组件正确性无关。

> The RL main path (rollout → reward → loss forward) is verified for real; only
> backward is missing. The backward SIGSEGV is a TE-source-build vs PyPI torch
> 2.13.0 toolchain ABI mismatch (corrupted gradient memory → NCCL sync crash),
> unrelated to RL-component correctness.

---

## 已知问题 / 限制 · Known issues / limitations

- **Qwen3.5 hybrid 未验证**：Qwen3.5 是 hybrid linear attention + MTP。要跑它需要
  `--spec slime_plugins.models.qwen3_5:get_qwen3_5_spec` + `--hybrid-layer-pattern`
  等额外定制（参考 `scripts/models/qwen3.5-4B.sh`），本指南未覆盖。
- **TE/torch ABI 不匹配（backward SIGSEGV）**：TE 2.18.0 官方预编译 wheel 只针对
  NVIDIA NGC torch 26.04（`materialize_cow_storage` 等 symbol 与 PyPI torch 2.13.0
  不同）；源码编译的 TE 能 import、forward 正常，但 backward 梯度同步时 SIGSEGV。
  要跑通真实训练，需换 NVIDIA NGC torch 26.04，或用 `build_conda.sh` 重建匹配环境
  （torch 2.11 + TE 2.16.1）。
- **模型多轮退化**：Qwen2.5-1.5B 在 4 轮 episode 的第 1 轮后常输出空/非 JSON，属
  模型能力问题，env 已正确反馈 parse_error，不影响流水线验证。

> - **Qwen3.5 hybrid unverified**: Qwen3.5 is hybrid linear attention + MTP. Running
>   it needs `--spec slime_plugins.models.qwen3_5:get_qwen3_5_spec` +
>   `--hybrid-layer-pattern` etc. (see `scripts/models/qwen3.5-4B.sh`), not covered here.
> - **TE/torch ABI mismatch (backward SIGSEGV)**: TE 2.18.0 prebuilt wheels only target
>   NVIDIA NGC torch 26.04 (symbols like `materialize_cow_storage` differ from PyPI
>   torch 2.13.0); source-built TE imports and runs forward, but SIGSEGVs on backward
>   gradient sync. To run real training, switch to NVIDIA NGC torch 26.04, or rebuild a
>   matched env via `build_conda.sh` (torch 2.11 + TE 2.16.1).
> - **Model multi-turn degradation**: Qwen2.5-1.5B often outputs empty/non-JSON after
>   round 1 of a 4-round episode — a model-capability issue; env correctly reports
>   parse_error, which does not affect pipeline validation.
