# Experiment Entrypoints and Reproduction README

这个仓库里有两层：

```text
1. 入口 / runner
2. 核心实现 / algorithm
```

常见分工是：

```text
scripts/*.py
  负责参数解析、循环 5 个 antigen × 5 个 seed、创建输出目录、调用核心模块

bo/**.py
  负责真正的算法逻辑、prompt 构造、LLM 解析、acquisition、oracle 调用
```

这份文档只管“从哪里启动”和“结果怎么复现”。

---

服务器根目录固定为：

```text
/mnt/data0/shared/AntBO/HEBO/AntBO
```

正式实验只看这一组设置：

```text
antigens = 1ADQ_A, 1FBI_X, 1H0D_C, 1NSN_S, 1OB1_C
seeds    = 42, 43, 44, 45, 46
n_evals  = 200
batch    = 1
```

正式输出统一放到：

```text
/mnt/data0/shared/AntBO/HEBO/AntBO/outputs/experiments/formal_5ag5seed200
```

---

## 1. 先看哪个 README

每个基线具体怎么做，看：

```text
environment_and_interfaces_readme.md
```

---

## 2. 环境准备

```bash
cd /mnt/data0/shared/AntBO/HEBO/AntBO
conda activate DGM
```

如果需要检查基础依赖：

```bash
python - <<'PY'
import bo
import task
print("AntBO import OK")
PY
```

Absolut 检查：

```bash
test -x /mnt/data0/shared/AntBO/Absolut/src/bin/Absolut && echo "Absolut OK"
```

LLM 环境变量检查：

```bash
python - <<'PY'
import os
print("LLM_API_KEY:", bool(os.getenv("LLM_API_KEY")))
print("LLM_BASE_URL:", os.getenv("LLM_BASE_URL"))
print("LLM_MODEL:", os.getenv("LLM_MODEL"))
PY
```

---

## 3. 统一输入文件

```text
/mnt/data0/shared/AntBO/HEBO/AntBO/experiments/formal_5ag5seed200/antigens.txt
/mnt/data0/shared/AntBO/HEBO/AntBO/experiments/formal_5ag5seed200/seeds.txt
```

`antigens.txt` 内容：

```text
1ADQ_A
1FBI_X
1H0D_C
1NSN_S
1OB1_C
```

`seeds.txt` 内容：

```text
42
43
44
45
46
```

---


## 4. 方法入口和复现命令

### 4.1 LDM_fn_seq_argmax

入口：

```text
/mnt/data0/shared/AntBO/HEBO/AntBO/bo/main.py
```

复现命令：

```bash
cd /mnt/data0/shared/AntBO/HEBO/AntBO

for seed in 42 43 44 45 46; do
  python bo/main.py \
    --config bo/config.yaml \
    --antigens_file experiments/formal_5ag5seed200/antigens.txt \
    --seed "$seed" \
    --save-path outputs/experiments/formal_5ag5seed200/ldm_fn_seq_argmax
done
```

注意：

```text
bo/main.py 是原始 AntBO 入口。
它不是 scripts 下面的 runner。
```

### 4.2 LDM_fn_one_argmax

入口：

```text
/mnt/data0/shared/AntBO/HEBO/AntBO/bo/ldm_light/ldm_acq.py
```

复现命令：

```bash
cd /mnt/data0/shared/AntBO/HEBO/AntBO

for seed in 42 43 44 45 46; do
  python bo/ldm_light/ldm_acq.py \
    --config bo/config.yaml \
    --antigens_file experiments/formal_5ag5seed200/antigens.txt \
    --seed "$seed" \
    --n_trials 1 \
    --n_evals 200 \
    --batch_size 1 \
    --n_init 20 \
    --parallel_budget 600 \
    --out_root outputs/experiments/formal_5ag5seed200/ldm_fn_one_argmax_
done
```

### 4.3 LDM_fn_par_softmax

入口：

```text
/mnt/data0/shared/AntBO/HEBO/AntBO/scripts/run_ldm_reservoir_absolut.py --selection softmax
```

复现命令：

```bash
cd /mnt/data0/shared/AntBO/HEBO/AntBO

for seed in 42 43 44 45 46; do
  python scripts/run_ldm_reservoir_absolut.py \
    --config bo/config.yaml \
    --antigens_file experiments/formal_5ag5seed200/antigens.txt \
    --seed "$seed" \
    --n_trials 1 \
    --n_evals 200 \
    --batch_size 1 \
    --n_init 20 \
    --init_mode random \
    --n_strategies 5 \
    --parallel_budget 600 \
    --selection softmax \
    --planner_mode n_choices \
    --softmax_eta 1.0 \
    --out_root outputs/experiments/formal_5ag5seed200/ldm_fn_par_softmax
done
```

### 4.4 LDM_fn_par_argmax

入口：

```text
/mnt/data0/shared/AntBO/HEBO/AntBO/scripts/run_ldm_reservoir_absolut.py --selection argmax
```

复现命令：

```bash
cd /mnt/data0/shared/AntBO/HEBO/AntBO

for seed in 42 43 44 45 46; do
  python scripts/run_ldm_reservoir_absolut.py \
    --config bo/config.yaml \
    --antigens_file experiments/formal_5ag5seed200/antigens.txt \
    --seed "$seed" \
    --n_trials 1 \
    --n_evals 200 \
    --batch_size 1 \
    --n_init 20 \
    --init_mode random \
    --n_strategies 5 \
    --parallel_budget 600 \
    --selection argmax \
    --planner_mode n_choices \
    --out_root outputs/experiments/formal_5ag5seed200/ldm_fn_par_argmax
done
```

### 4.5 LLM_rerank

入口：

```text
/mnt/data0/shared/AntBO/HEBO/AntBO/bo/ldm/llm/LLM_baseline.py
```

复现命令：

```bash
cd /mnt/data0/shared/AntBO/HEBO/AntBO

for seed in 42 43 44 45 46; do
  python bo/ldm/llm/LLM_baseline.py \
    --config bo/config.yaml \
    --antigens_file experiments/formal_5ag5seed200/antigens.txt \
    --seed "$seed" \
    --n_trials 1 \
    --n_evals 200 \
    --batch_size 1 \
    --out_root outputs/experiments/formal_5ag5seed200/llm_rerank__pool
done
```

### 4.6 LLM_gen

入口：

```text
/mnt/data0/shared/AntBO/HEBO/AntBO/scripts/run_llm_direct_absolut.py --method LLM_gen
```

复现命令：

```bash
cd /mnt/data0/shared/AntBO/HEBO/AntBO

for seed in 42 43 44 45 46; do
  python scripts/run_llm_direct_absolut.py \
    --method LLM_gen \
    --config bo/config.yaml \
    --antigens_file experiments/formal_5ag5seed200/antigens.txt \
    --seed "$seed" \
    --n_trials 1 \
    --n_evals 200 \
    --batch_size 1 \
    --out_root outputs/experiments/formal_5ag5seed200/llm_gen
done
```

### 4.7 LDM_gen_softmax

入口：

```text
/mnt/data0/shared/AntBO/HEBO/AntBO/scripts/run_llm_direct_absolut.py --method LDM_gen_softmax
```

复现命令：

```bash
cd /mnt/data0/shared/AntBO/HEBO/AntBO

for seed in 42 43 44 45 46; do
  python scripts/run_llm_direct_absolut.py \
    --method LDM_gen_softmax \
    --config bo/config.yaml \
    --antigens_file experiments/formal_5ag5seed200/antigens.txt \
    --seed "$seed" \
    --n_trials 1 \
    --n_evals 200 \
    --batch_size 1 \
    --n_init 20 \
    --gen_m 5 \
    --softmax_eta 1.0 \
    --gp_train_steps 300 \
    --acq_device cpu \
    --out_root outputs/experiments/formal_5ag5seed200/ldm_gen_softmax
done
```

### 4.8 LDM_gen_argmax

入口：

```text
/mnt/data0/shared/AntBO/HEBO/AntBO/scripts/run_llm_direct_absolut.py --method LDM_gen_argmax
```

复现命令：

```bash
cd /mnt/data0/shared/AntBO/HEBO/AntBO

for seed in 42 43 44 45 46; do
  python scripts/run_llm_direct_absolut.py \
    --method LDM_gen_argmax \
    --config bo/config.yaml \
    --antigens_file experiments/formal_5ag5seed200/antigens.txt \
    --seed "$seed" \
    --n_trials 1 \
    --n_evals 200 \
    --batch_size 1 \
    --n_init 20 \
    --gen_m 5 \
    --gp_train_steps 300 \
    --acq_device cpu \
    --out_root outputs/experiments/formal_5ag5seed200/ldm_gen_argmax
done
```

---

## 5. 复现检查

检查每个方法是不是 25 个 run 都在，而且每个 run 都有 200 行：

```bash
cd /mnt/data0/shared/AntBO/HEBO/AntBO
python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("outputs/experiments/formal_5ag5seed200")
for method in [
    "ldm_fn_seq_argmax",
    "ldm_fn_one_argmax_",
    "ldm_fn_par_softmax",
    "ldm_fn_par_argmax",
    "llm_rerank__pool",
    "llm_gen",
    "ldm_gen_softmax",
    "ldm_gen_argmax",
]:
    hits = list((root / method).rglob("results.csv"))
    ok = sum(len(pd.read_csv(p)) == 200 for p in hits)
    print(method, ok, "/", len(hits))
PY
```
