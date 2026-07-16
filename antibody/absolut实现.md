# Task 3: Absolut Oracle and Acquisition Interface README

本文档只使用服务器路径：

```text
/mnt/data0/shared/AntBO/HEBO/AntBO
```

目标：说明 AntBO 如何调用 Absolut! 构建 oracle 给蛋白质打分，以及算法中的 acquisition function 接口如何构建。

---

## 1. 数据表示

AntBO 优化 CDRH3 sequence。

两种表示：

```text
string:
  CARDRSTYWYF

index-encoded numpy array:
  shape = [num_seq, seq_len]
  每个位置是 amino acid index
```

默认：

```text
seq_len = 11
alphabet = ACDEFGHIKLMNPQRSTVWY
```

目标：

```text
minimize Absolut binding energy
lower is better
```

---

## 2. Absolut oracle 配置

配置文件：

```text
/mnt/data0/shared/AntBO/HEBO/AntBO/bo/config.yaml
```

关键字段：

```yaml
bbox:
  antigen: PLACEHOLDER
  tool: Absolut
  path: /mnt/data0/shared/AntBO/Absolut
  process: 2
  startTask: 0
```

Absolut executable：

```text
/mnt/data0/shared/AntBO/Absolut/src/bin/Absolut
```

检查：

```bash
test -x /mnt/data0/shared/AntBO/Absolut/src/bin/Absolut && echo "Absolut OK"
```

---

## 3. Oracle 调用链

原始 AntBO：

```text
/mnt/data0/shared/AntBO/HEBO/AntBO/task/task.py::Task.energy
/mnt/data0/shared/AntBO/HEBO/AntBO/task/tools.py::Absolut.energy
```

调用链：

```text
algorithm proposes x_next
  -> Task.energy(x_next)
  -> Absolut.energy(x_next)
  -> write TempCDR3 input file
  -> call ./src/bin/Absolut repertoire
  -> read Energy
  -> return min_energy, sequences
```

LLM/LDM baseline：

```text
/mnt/data0/shared/AntBO/HEBO/AntBO/bo/ldm_light/ldm_acq.py::make_evaluator
/mnt/data0/shared/AntBO/HEBO/AntBO/bo/ldm_light/ldm_acq.py::AbsolutEvaluator.energy
```

`make_evaluator` 逻辑：

```text
if bbox.tool == random:
  use RandomEvaluator
else:
  use AbsolutEvaluator
```

真实实验必须用：

```text
bbox.tool = Absolut
```

---

## 4. Absolut repertoire 执行方式

`Absolut.energy` 做：

```text
1. 把 index-encoded sequence 转成 CDRH3 string。
2. 写 input file:
   TempCDR3_{antigen}.txt

3. 调用:
   taskset -c {startTask}-{startTask + process} \
     ./src/bin/Absolut repertoire \
     {antigen} \
     {input_file} \
     {process}

4. 读取:
   {antigen}FinalBindings_Process_1_Of_1.txt

5. 对每条 sequence 取最小 Energy。
6. 返回 values, sequences。
```

输入文件格式：

```text
1<TAB>SEQUENCE_1
2<TAB>SEQUENCE_2
```

为什么取最小 Energy：

```text
Absolut 会为同一条 sequence 产生多个 binding pose。
AntBO 使用最小 Energy 作为该 sequence 的 oracle score。
```

---

## 5. 并发安全

原始 `task/tools.py::Absolut` 使用 prefix：

```text
run{pid}_{epoch_ms}_
```

作用：

```text
多个实验共享同一个 Absolut installation 时，避免临时文件互相覆盖。
```

LLM/LDM 新 baseline 的 `AbsolutEvaluator` 还使用 lock：

```text
.antbo_llm_acq_{antigen}.lock
```

---

## 6. Antigen context

部分 LLM 方法会把 Absolut antigen 信息放进 prompt。

代码：

```text
/mnt/data0/shared/AntBO/HEBO/AntBO/bo/ldm_light/ldm_acq.py::collect_antigen_context
```

它调用：

```text
./src/bin/Absolut info_antigen {antigen}
./src/bin/Absolut info_filenames {antigen}
```

保存：

```text
llm_antigen_context.json
```

注意：

```text
antigen context 只给 LLM prompt 用，不参与 energy 计算。
```

---

## 7. Acquisition 构建接口

核心函数：

```text
/mnt/data0/shared/AntBO/HEBO/AntBO/bo/ldm_light/ldm_acq.py::fit_gp_and_make_acquisition
```

接口：

```python
gp, f_acq = fit_gp_and_make_acquisition(
    rows,
    gp_train_steps=300,
    device="cpu",
)
```

`rows` 至少包含：

```text
LastProtein
LastValue
```

输出：

```text
gp:
  fitted GP surrogate

f_acq:
  callable acquisition function
```

`f_acq(x)`：

```text
input:
  candidate index-encoded sequences

output:
  acquisition score for each candidate
```

---

## 8. GP 和 EI

`fit_gp_and_make_acquisition` 做：

```text
1. train_seqs = rows["LastProtein"]
2. train_x = seqs_to_indices(train_seqs)
3. y_raw = rows["LastValue"]
4. train_y = normalize(y_raw)
5. train GP with transformed_overlap kernel
6. build minimization Expected Improvement
```

当前 acquisition 是 minimization EI：

```text
z = (best - mu) / sigma
EI = (best - mu) * Phi(z) + sigma * phi(z)
```

含义：

```text
EI 越大，candidate 越值得送 Absolut 评估。
```

---

## 9. Acquisition 在方法中的使用

```text
LDM_fn_one_argmax:
  LLM 输出 1 条 candidate generation instruction。
  execute_atoms 生成 candidates。
  代码算 EI。
  代码 argmax 选最终 sequence。

LDM_fn_par_softmax:
  LLM 输出 K 条 strategies。
  每条 strategy 生成一个 pool。
  每个 pool 选 representative。
  representatives 上 softmax 选最终 sequence。

LDM_fn_par_argmax:
  与 softmax 相同，但最后 argmax。

LDM_gen_softmax:
  LLM 直接生成 gen_m 条 sequences。
  代码算 EI。
  softmax 选最终 sequence。

LDM_gen_argmax:
  LLM 直接生成 gen_m 条 sequences。
  代码算 EI。
  argmax 选最终 sequence。
```

---

## 10. Candidate generation instruction 接口

LDM function 方法里，LLM 输出候选生成指令：

```text
LocalSearch(...)
NeighborSampling(...)
LatinHyperCubeSampling(...)
Or(...)
```

执行入口：

```text
/mnt/data0/shared/AntBO/HEBO/AntBO/bo/ldm/acquisition/parallel_search.py::execute_atoms
```

输出 record 常见字段：

```text
seq
ei
mu
sigma
bias
bias+ei
source
```

---

## 11. Minimal oracle test code

直接在服务器运行下面代码：

```bash
cd /mnt/data0/shared/AntBO/HEBO/AntBO
python - <<'PY'
from pathlib import Path
import yaml

from bo.ldm_light.ldm_acq import make_evaluator, seqs_to_indices

ROOT = Path("/mnt/data0/shared/AntBO/HEBO/AntBO")
config = yaml.safe_load((ROOT / "bo/config.yaml").read_text())

antigen = "1ADQ_A"
evaluator, bbox = make_evaluator(config, antigen, "oracle_test_1ADQ_A")

seqs = ["CARDRSTYWYF"]
values, evaluated_seqs = evaluator.energy(seqs_to_indices(seqs))

print(evaluated_seqs)
print(values)
PY
```

---

## 12. Minimal acquisition test code

直接在服务器运行下面代码：

```bash
cd /mnt/data0/shared/AntBO/HEBO/AntBO
python - <<'PY'
import torch
from bo.ldm_light.ldm_acq import fit_gp_and_make_acquisition, seqs_to_indices

rows = [
    {"LastProtein": "CARDRSTYWYF", "LastValue": -80.0},
    {"LastProtein": "CAKDRSTYWYF", "LastValue": -82.0},
    {"LastProtein": "CARGGSTYWYF", "LastValue": -78.0},
    {"LastProtein": "CARDRSTAWYF", "LastValue": -85.0},
    {"LastProtein": "CARDRSTYAAA", "LastValue": -75.0},
]

gp, f_acq = fit_gp_and_make_acquisition(rows, gp_train_steps=30, device="cpu")

candidates = ["CARDRSTYWYA", "CARGGSTYWYA"]
x = torch.tensor(seqs_to_indices(candidates), dtype=torch.float32)
scores = f_acq(x).detach().cpu().numpy()

for seq, score in zip(candidates, scores):
    print(seq, score)
PY
```
