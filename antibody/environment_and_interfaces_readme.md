# Environment Interfaces and Baseline Implementation README

本文档只写两件事：

1. 环境接口怎么接
2. 每个 baseline 怎么实现

服务器根目录固定为：

```text
/mnt/data0/shared/AntBO/HEBO/AntBO
```

---

## 1. 环境接口

### Absolut oracle

Absolut 在：

```text
/mnt/data0/shared/AntBO/Absolut/src/bin/Absolut
```

输入是 CDRH3 sequence，输出是 energy。能量越低越好。正式实验都要走这个 oracle。

### LLM 接口

LLM 通过环境变量接入：

```text
LLM_API_KEY
LLM_BASE_URL
LLM_MODEL
```

LLM 在这些方法里的作用只有两类：

1. 生成候选搜索指令，或者直接生成 sequence
2. 在 candidate pool 里选择 sequence

LLM 不直接算 acquisition，也不直接调用 Absolut。

### Acquisition 接口

核心函数是：

```text
/mnt/data0/shared/AntBO/HEBO/AntBO/bo/ldm_light/ldm_acq.py::fit_gp_and_make_acquisition
```

它接收历史真实评估点，返回 GP 和 acquisition callable。这个 callable 用来给候选 sequence 打分。

### Candidate pool

candidate pool 就是一批已经生成好的合法 CDRH3 sequence。LLM 只在 pool 里选，不自己发明新的评分规则。

---

## 2. Baseline implementation

每个 case 都按同样的顺序看：

```text
Path -> Initialization -> LLM output -> code behavior -> final sequence
```

### 2.1 LDM_fn_seq_argmax

#### Path

```text
run from:
  /mnt/data0/shared/AntBO/HEBO/AntBO/bo/main.py

core implementation:
  /mnt/data0/shared/AntBO/HEBO/AntBO/bo/localbo_cat.py
  /mnt/data0/shared/AntBO/HEBO/AntBO/bo/ldm/
```

#### Initialization

```text
n_init = 20
llm_init_enabled = false
```

前 20 个点是原始 AntBO BO loop 产生的初始真实评估点。

#### LLM output

```text
candidate generation instruction
bias function
```

candidate generation instruction 是函数式搜索指令，不是随便写的代码。

#### code behavior

1. 用已有真实评估点训练 GP
2. 读取 LLM 输出的 candidate generation instruction
3. 生成 candidates
4. 对每个 candidate 算 acquisition score
5. 对每个 candidate 算 bias score
6. 合成 combined score
7. 把 top candidates 交给 LLM review
8. LLM 选一个
9. 送 Absolut

#### final sequence

LLM 决定候选怎么生成，也决定最后从 top candidates 里选哪个。

---

### 2.2 LDM_fn_one_argmax

#### Path

```text
run from:
  /mnt/data0/shared/AntBO/HEBO/AntBO/bo/ldm_light/ldm_acq.py

supporting acquisition code:
  /mnt/data0/shared/AntBO/HEBO/AntBO/bo/ldm/acquisition/parallel_search.py
```

#### Initialization

```text
n_init = 20
```

前 20 步是 warmup。warmup 阶段每一步先生成 candidate pool，再让 LLM 从 pool 里选 sequence。

#### LLM output

warmup 阶段：从 candidate pool 中选 sequence。  
acquisition 阶段：输出 1 条 candidate generation instruction。

#### code behavior

warmup：
1. 代码生成 candidate pool
2. LLM 从 pool 里选 sequence
3. 代码检查合法性和去重
4. 送 Absolut

acquisition：
1. 用 history 训练 GP
2. LLM 输出 1 条 candidate generation instruction
3. 代码执行 instruction，生成最多 600 个 candidates
4. 对每个 candidate 计算 EI
5. 选 EI 最大的 candidate
6. 送 Absolut

#### final sequence

warmup 阶段是 LLM 直接选。  
acquisition 阶段是 LLM 决定怎么找候选，代码用 EI argmax 决定最终 sequence。

---

### 2.3 LDM_fn_par_softmax

#### Path

```text
run from:
  /mnt/data0/shared/AntBO/HEBO/AntBO/scripts/run_ldm_reservoir_absolut.py --selection softmax

core implementation:
  /mnt/data0/shared/AntBO/HEBO/AntBO/bo/ldm_reservoir/
  /mnt/data0/shared/AntBO/HEBO/AntBO/bo/ldm/acquisition/parallel_search.py
```

#### Initialization

```text
n_init = 20
init_mode = random
```

#### LLM output

```text
5 条 search strategies
```

#### code behavior

1. 用真实评估点训练 GP
2. LLM 一次输出 5 条 search strategies
3. 代码把 parallel_budget=600 分给这些 strategies
4. 每条 strategy 生成自己的 candidate pool
5. 每个 candidate 算 EI
6. 每个 pool 选一个 representative
7. 对 5 个 representatives 做 softmax
8. 采样出 1 条
9. 送 Absolut

#### final sequence

LLM 决定搜索策略，代码生成 candidates 并算分，最终 sequence 由 softmax 采样决定。

---

### 2.4 LDM_fn_par_argmax

#### Path

```text
run from:
  /mnt/data0/shared/AntBO/HEBO/AntBO/scripts/run_ldm_reservoir_absolut.py --selection argmax

core implementation:
  /mnt/data0/shared/AntBO/HEBO/AntBO/bo/ldm_reservoir/
  /mnt/data0/shared/AntBO/HEBO/AntBO/bo/ldm/acquisition/parallel_search.py
```

#### Initialization

```text
n_init = 20
init_mode = random
```

#### LLM output

```text
5 条 search strategies
```

#### code behavior

1. 每条 strategy 生成 candidate pool
2. 代码计算 EI
3. 每个 pool 选一个 representative
4. 在 representatives 里直接选 EI 最大的 sequence
5. 送 Absolut

#### final sequence

LLM 决定搜索策略，代码用 acquisition argmax 决定最终 sequence。

---

### 2.5 LLM_rerank

#### Path

```text
run from:
  /mnt/data0/shared/AntBO/HEBO/AntBO/bo/ldm/llm/LLM_baseline.py

core implementation:
  /mnt/data0/shared/AntBO/HEBO/AntBO/bo/ldm/llm/LLM_baseline.py
```

#### Initialization

没有 GP，也没有 acquisition。只有 candidate pool。

#### LLM output

从 candidate pool 里选 sequence。

#### code behavior

1. 代码生成 candidate pool
2. LLM 在 pool 里挑选
3. 代码检查合法性和去重
4. 送 Absolut

#### final sequence

LLM 直接决定最终 sequence。

---

### 2.6 LLM_gen

#### Path

```text
run from:
  /mnt/data0/shared/AntBO/HEBO/AntBO/scripts/run_llm_direct_absolut.py --method LLM_gen

core implementation:
  /mnt/data0/shared/AntBO/HEBO/AntBO/bo/llm_direct/core.py
```

#### Initialization

没有 acquisition initialization。直接用历史上下文驱动 LLM 生成。

#### LLM output

sequence 本身。

#### code behavior

1. 代码把 prompt 送给 LLM
2. LLM 直接生成 sequence
3. 代码做合法性检查和去重
4. 送 Absolut

#### final sequence

LLM 直接决定最终 sequence。

---

### 2.7 LDM_gen_softmax

#### Path

```text
run from:
  /mnt/data0/shared/AntBO/HEBO/AntBO/scripts/run_llm_direct_absolut.py --method LDM_gen_softmax

core implementation:
  /mnt/data0/shared/AntBO/HEBO/AntBO/bo/llm_direct/core.py
  /mnt/data0/shared/AntBO/HEBO/AntBO/bo/ldm_light/ldm_acq.py
```

#### Initialization

先跑前 20 个 direct-generation 历史点，再开始 acquisition。

#### LLM output

`gen_m` 条 sequence。

#### code behavior

1. LLM 直接生成 `gen_m` 条 sequence
2. 代码训练 GP
3. 给每条 candidate 算 EI
4. 用 softmax 选一个
5. 送 Absolut

#### final sequence

LLM 提供候选，代码用 acquisition + softmax 决定最终 sequence。

---

### 2.8 LDM_gen_argmax

#### Path

```text
run from:
  /mnt/data0/shared/AntBO/HEBO/AntBO/scripts/run_llm_direct_absolut.py --method LDM_gen_argmax

core implementation:
  /mnt/data0/shared/AntBO/HEBO/AntBO/bo/llm_direct/core.py
  /mnt/data0/shared/AntBO/HEBO/AntBO/bo/ldm_light/ldm_acq.py
```

#### Initialization

先跑前 20 个 direct-generation 历史点，再开始 acquisition。

#### LLM output

`gen_m` 条 sequence。

#### code behavior

1. LLM 直接生成 `gen_m` 条 sequence
2. 代码训练 GP
3. 给每条 candidate 算 EI
4. 直接选 EI 最大的 sequence
5. 送 Absolut

#### final sequence

LLM 提供候选，代码用 acquisition argmax 决定最终 sequence。

