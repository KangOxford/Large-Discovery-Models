# LLM Advisor for strbo_v1 BO Loop — Design Proposal

> 让 LLM 在每轮 BO 拥有 **三阶段干预**:**Stage A1** 改池 → (若产出 analogs)**Stage A2** 同步审 analogs → **BO 在 mutated pool 上跑** → **Stage B** review BO 候选 → 评分入历史。保持 `bayesian_analog_search.py` 不被破坏,通过外置 orchestrator + advisor 协议接入。每轮 **LLM 最多调三次** (Stage A1 必调,Stage A2 条件调,Stage B 必调;Stage A1 还会在池子不足时循环重试)。

---

## 0. Background: strbo_v1 算法逻辑速览

```
seed_smiles
     │
     ▼
pool: FIFOSet(可选 max_size)   ←── 候选 SMILES 池(待评分)
     │
     ▼
┌──────────────── warm-up (一次性,可关) ────────────────┐
│  if len(pool) < init_size:                            │
│    analog_fn(rng.sample(pool, n_needed))              │  一次批量调用
│    → _add_analogues_to_pool                           │
└───────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────── initialization ──────────────────────┐
│  rng.sample(pool, k=init_size) → init_chosen          │
│  _safe_score_n(scorers, init_chosen) → history        │
│  analog_fn(init_chosen) → pool 扩增                   │  一次批量调用
└───────────────────────────────────────────────────────┘
     │
     ▼
┌──── for it in range(n_iterations) ────┐
│ 1. _select_candidates:                │  ← 传统 BO 步骤(GP + 采集)
│      GP.fit(history) → μ, σ           │
│      acquisition(mu, sigma, ...)      │
│      top-k by acq → candidates        │
│ 2. _safe_score_n(candidates)          │
│ 3. analog_fn(candidates) → pool       │ ← 一次批量调用
└───────────────────────────────────────┘
     │
     ▼
history: List[(smi, score(s))]
```

### 关键文件 / 函数

| 文件 | 角色 | 可控点 |
|---|---|---|
| `strbo_v1/bayesian_analog_search.py:364` `bayesian_analog_search()` | 整体闭环 | warm-up / init / BO 三阶段调度 |
| `strbo_v1/bayesian_analog_search.py:618` `select_candidates()` | **advisor step** | GP → acquisition → top-k(**纯函数,可外置**) |
| `strbo_v1/bayesian_analog_search.py:549-608` | BO 主循环体 | "评分 + 扩增" 紧耦合 |
| `strbo_v1/acquisition.py:80/116/150` | EI / PI / UCB | 单目标采集 |
| `strbo_v1/acquisition.py:493` EHVI / `acquisition.py:589` Chebyshev | 多目标 | EHVI 仅 2D;≥3D 退化为 ParEGO |
| `strbo_v1/gp.py:410` GPSurrogate | Tanimoto-FP 或 smiles-strkernel | 协方差核选择 |
| `strbo_v1/analog.py:585` `generate_analogs()` + `ReasynConfig` | **唯一扩增来源** | search_width / num_cycles / time_limit 全静态 |
| `strbo_v1/utils.py:9` FIFOSet | 候选池 | 只能 `add / discard / popleft`,无策略 |

### 当前架构的两个硬编码瓶颈

1. **`analog_fn` 是单一固定生成器**:warm-up / init / 每 BO round 都用同一个 `ReasynConfig`,扩增策略(SBS / Michael addition / fragment link)一刀切,无法按收敛状态切换。
2. **扩增是被动的**:每 BO round 无差别把 top-k 候选全部丢给 `analog_fn`,没有"这一轮到底该不该扩 / 扩哪些 / 用哪个生成器"的决策层;且 GP 选谁就评谁,无 override 通道。

---

## 1. 总体架构

```
                         ┌───────────────────────────────────────┐
                         │        LLMAdvisorClient               │
                         │  (OpenAI-compatible chat/completions) │
                         └──────────────────┬────────────────────┘
                                            │ json in / json out
                                            ▼
                         ┌───────────────────────────────────────┐
                         │       LLMAdvisor (单类三方法)          │
                         │   - decide_actions(state_A1)           │
                         │   - decide_review_analogs(state_A2)    │
                         │   - decide_review_suggestions(state_B) │
                         │     (各自带有限重试 + 错误反馈)        │
                         └──────────────────┬────────────────────┘
                                            │
        ┌───────────────────────────────────┼──────────────────────────────────┐
        ▼                                   ▼                                  ▼
 ┌──────────────────┐           ┌──────────────────────┐           ┌──────────────────────┐
 │ ReaSyn Pool      │           │ BO Orchestrator       │           │ RoundState (3 份)    │
 │ (多 ReasynConfig) │           │ (新文件)              │           │ PreAction /          │
 └──────────────────┘           └──────────────────────┘           │ PreReviewAnalogs /   │
                                            │                      │ PostSuggestion       │
                                            ▼                      └──────────────────────┘
                              ┌────────────────────────────┐
                              │ strbo_v1.bayesian_search   │  ← 现状,零改动
                              │  - select_candidates (GP)  │
                              │  - safe_score_n            │
                              └────────────────────────────┘
```

**Pool-size loop**:Stage A1 内部有一个最多 5 次的循环:每次跑 `decide_actions`,如果 `len(pool) < pool_min_size` 就把"必须 propose/analog 补池"的错误反馈给 LLM 重试,直到池子达标或达上限。

---

## 2. 数据契约:`PreActionState` / `PreReviewAnalogsState` / `PostSuggestionState`

`RoundState` 拆成三份,**三阶段 LLM 看到的是不同时刻的池**。Stage A1 看到改之前的 pool;Stage A2 看到新生的 analogs(在 mutated pool 上);Stage B 看到改完后的 pool + 在该 pool 上算出的 BO 推荐。

```python
@dataclass(frozen=True)
class PreActionState:
    """Stage A1 用。pool 是 round 起点快照;bo_suggestions 必空
    (BO 还没跑)。LLM 看到的是改之前的 pool;它输出的 propose / reject /
    analog / noop 会被 orchestrator 应用。"""
    round_idx: int
    n_total_rounds: int
    pdf_context: str
    objective_legend: list[dict]

    pool: tuple[str, ...]                   # pool_t(round 起点)
    pool_size_cap: int | None
    history: tuple[tuple[str, tuple[float, ...]], ...]   # 含 init 阶段
    gp_summary: GPSummary                    # 上轮末的 GP 状态(仅参考)

    pareto_front: list[tuple] | None
    best_score_per_obj: list[float | None]
    stagnation_counter: int
    diversity_metric: float | None

    # Stage A1 池容量约束(orchestrator 控制)
    pool_min_size: int = 1                  # 当 len(pool) < pool_min_size 时
                                            # noop 必被拒,LLM 需 propose/analog 补池

    # 本 phase 内部
    previous_errors: tuple[str, ...]
    attempt: int                            # Stage A1 内部计数


@dataclass(frozen=True)
class PreReviewAnalogsState:
    """Stage A2 用。仅当 Stage A1 输出的 analog 块产生非空 new_analogs
    时才构造;LLM 决定 keep / reject / rescore。pool 已是 Stage A1
    改完的状态。"""
    round_idx: int
    n_total_rounds: int
    pdf_context: str
    objective_legend: list[dict]

    pool: tuple[str, ...]                   # pool_t'(Stage A1 后)
    pool_size_cap: int | None
    history: tuple[tuple[str, tuple[float, ...]], ...]
    gp_summary: GPSummary                    # 同 Stage A1 看到的(本轮 BO 未跑)

    new_analogs: tuple[AnalogueRecord, ...] # Stage A1 产出的模拟物(本 phase 必审)

    pareto_front: list[tuple] | None
    best_score_per_obj: list[float | None]
    stagnation_counter: int
    diversity_metric: float | None

    # 本 phase 内部
    previous_errors: tuple[str, ...]
    attempt: int


@dataclass(frozen=True)
class PostSuggestionState:
    """Stage B 用。pool 已是 Stage A1(+A2) 改完的状态;bo_suggestions
    是从 mutated pool 上 GP + 采集算出来的。LLM 输出一个 review_bo 块。"""
    round_idx: int
    n_total_rounds: int
    pdf_context: str
    objective_legend: list[dict]

    pool: tuple[str, ...]                   # pool_t'(Stage A1+A2 后)
    pool_size_cap: int | None
    history: tuple[tuple[str, tuple[float, ...]], ...]
    gp_summary: GPSummary                    # 本轮 BO 步骤的 GP 状态

    bo_suggestions: tuple[PickRecord, ...]  # ★ 在 pool_t' 上算的
    acq_function: str

    pareto_front: list[tuple] | None
    best_score_per_obj: list[float | None]
    stagnation_counter: int
    diversity_metric: float | None

    # 本 phase 内部
    previous_errors: tuple[str, ...]        # Stage B 内部计数
    attempt: int
```

> 设计要点:`pending_analogs` **不是 state 字段**。Stage A1 的 `analog` 块产出 `new_analogs`(orchestrator 内的局部变量),仅当非空时才进入 Stage A2 review。Stage A2 review 完即丢弃,不在 stage 之间 carry。

```python
@dataclass(frozen=True)
class GPSummary:
    """GP 整体状态压缩。"""
    n_train: int
    train_score_mean: float
    train_score_std: float
    pool_pred_mean: float
    pool_pred_std: float
    pool_max_uncertainty: float
    in_prior_mode: bool                     # GP 退化到 prior

@dataclass(frozen=True)
class PickRecord:
    """BO 候选 = GP top-k 之一。LLM 在 Stage B review 时看到的就是这个。"""
    smiles: str
    acq_value: float                        # 采集函数值(EI / UCB / EHVI / Chebyshev 等)
    mu: float                               # GP 后验均值
    sigma: float                            # GP 后验标准差
    nearest_history: list[tuple[str, float]] # 最近 3 个历史邻居(SMILES, score)

@dataclass(frozen=True)
class AnalogueRecord:
    """ReaSyn 产物,待 LLM 在 Stage A2 review。"""
    seed_smiles: str
    analogue_smiles: str
    reasyn_score: float | None              # ReaSyn 自带的合成可行性打分
    synthesis: str | None
    num_steps: int | None
    scf_sim: float | None
    pharm2d_sim: float | None
```

---

## 3. 决策契约:三阶段

> LLM Advisor 每轮分三阶段调用。**Stage A1** 改池,**Stage A2** 同步审 analogs(条件触发),**Stage B** 在 BO 跑完后 review BO 候选。**三阶段允许集互不相交**,跨阶段块直接 `SemanticError`。

### 3.1 Stage A1:池管理(每 round 第一步,LLM call #1)

**允许的块类型**(`PHASE_A_ACTIONS_ALLOWED`,每种至多一个):

| # | type | 是否必出 | 作用 |
|---|---|---|---|
| 1 | `propose` | 可选 | 注入新 SMILES 到 pool |
| 2 | `reject` | 可选 | 从 pool 移除 SMILES |
| 3 | `analog` | 可选 | 选 ReaSyn 种子,产物加入 `new_analogs`(本 round 同步进入 Stage A2 审) |
| 4 | `noop` | 可选 | 明确声明本阶段无池动作;**当 `len(pool) < pool_min_size` 时被拒** |

**禁止的块**:`review_bo`、`review_analogs`。`validate_blocks_phase` 抛 `SemanticError`,advisor 重试并把错误反馈到 prompt。

### 3.2 Stage A2:审 analogs(条件触发,LLM call #2)

**触发条件**:Stage A1 输出的 `analog` 块产生了非空 `new_analogs`(经 `analog_fn` 实际生成)。若 `new_analogs` 为空 → 跳过该阶段。

**允许的块类型**(`PHASE_A_REVIEW_ANALOGS_ALLOWED`):

| # | type | 是否必出 | 作用 |
|---|---|---|---|
| 1 | `review_analogs` | **必出** | 对每个 `new_analogs[i]` 给出 `keep` / `reject` / `rescore_with_different_params` |

**禁止的块**:所有 Stage A1 块 + `review_bo`。`validate_blocks_phase` 严格拒绝。

### 3.3 Stage B:review BO(每 round 第三步,LLM call #3)

**允许的块类型**(`PHASE_B_SUGGESTIONS_ALLOWED`):

| # | type | 是否必出 | 作用 |
|---|---|---|---|
| 1 | `review_bo` | **必出** | 逐个 review 在 mutated pool 上算出的 BO top-k |

**禁止的块**:所有 Stage A1 / A2 块。若出现 → `SemanticError("X not allowed in Stage B")`,丢弃并 warn。

### 3.4 块定义(6 个 Block,跨阶段共享 schema)

#### 3.4.1 `review_bo` — BO 候选 review(Stage B 必出)

LLM 看每个 BO suggestion 的 GP μ/σ/采集值,逐个决定:

- `ok`:保留 BO 的 pick,该 SMILES 送入目标函数评分
- `override:NEW_SMILES`:**替换** BO 的 pick,`NEW_SMILES` 送入目标函数评分(它不需要在 pool 中;**`NEW_SMILES` 本身不进入 pool**,它只走评分)
- `skip`:不评该 SMILES(空 slot;若 batch 未填满则该轮评分数 < batch_size,**不**回退到 BO 次优)

```python
@dataclass
class ReviewBOBlock:
    type: Literal["review_bo"] = "review_bo"
    rationale: str                          # ≤ 600 chars
    decisions: dict[str, str]               # 必含所有 bo_suggestions 的 SMILES
    # 值: "ok" | "override:<SMILES>" | "skip"
    # 缺失 → 默认 "ok"(兜底,warn 一次)
```

**例**:
```json
{
  "type": "review_bo",
  "rationale": "GP picked 3 SMILES; CCC looks like a bad lipinski candidate, swap for an aspirin-like",
  "decisions": {
    "CCO": "ok",
    "CCN": "ok",
    "CCC": "override:CC(=O)Oc1ccccc1C(=O)O"
  }
}
```

> 设计要点:`override` 是 1-to-1 替换。若 LLM 想多评几个,可同时用 `propose`(注入 pool,本轮**不**评)并在后续轮次被 GP 选上,或直接 `override:NEW_SMILES` 把 slot 抢过来。

#### 3.4.2 `propose` — 注入新 SMILES(Stage A1)

注入的 SMILES 立即加入 pool(下轮可见)。**不**自动入本轮评分。

```python
@dataclass
class ProposeBlock:
    type: Literal["propose"] = "propose"
    rationale: str                          # ≤ 400 chars
    smiles: list[str]                       # 1..10 条
    rationale_per_mol: dict[str, str]       # {smiles: 一句话解释},可省略
```

#### 3.4.3 `reject` — 移除 pool 成员(Stage A1)

```python
@dataclass
class RejectBlock:
    type: Literal["reject"] = "reject"
    rationale: str                          # ≤ 200 chars
    targets: list[str]                      # 1..50 条;必须在 pool 中
    reason: Literal[
        "too_similar_to_history",
        "likely_toxic",
        "synthetically_infeasible",
        "out_of_scope_pharmacophore",
        "no_signal_for_target",
    ]
```

#### 3.4.4 `analog` — 触发 ReaSyn(Stage A1)

ReaSyn 产出的模拟物进入 orchestrator 内的局部 `new_analogs` 列表,**本 round 同步进入 Stage A2 review**(不再 carry 到下轮)。`rescore_with_different_params` 判定:orchestrator 直接丢弃原产物,本轮不入 pool(下轮 LLM 若想重生成,自行在 Stage A1 重新发 `analog` 块)。

```python
@dataclass
class AnalogBlock:
    type: Literal["analog"] = "analog"
    rationale: str                          # ≤ 400 chars
    seeds: list[str]                        # 1..10 条;可来自 pool / bo_suggestions / history
    generator_hint: Literal["conservative", "aggressive", "scaffold_hop"] | None
    n_per_seed: int = 5
    reasyn_config_override: dict | None     # 可选;本轮的 ReasynConfig 超参
```

#### 3.4.5 `review_analogs` — 审核待 review 的 ReaSyn 产物(Stage A2 必出)

**触发条件**:Stage A1 输出的 `analog` 块产出了非空 `new_analogs`(本 round 同步 review)。

```python
@dataclass
class ReviewAnalogsBlock:
    type: Literal["review_analogs"] = "review_analogs"
    rationale: str                          # ≤ 400 chars
    decisions: dict[str, AnalogueVerdict]   # 必含所有 new_analogs 的 analogue_smiles
    # AnalogueVerdict ∈ {"keep", "reject", "rescore_with_different_params"}
    # 缺失 → "reject"(fail-closed)

AnalogueVerdict = Literal["keep", "reject", "rescore_with_different_params"]
```

**`rescore_with_different_params` 语义**:该 analog 本 round 丢弃(不加入 pool)。下轮 LLM 若想重新评估,自行在 Stage A1 重新发 `analog` 块,以新 ReaSyn config 重生成。orchestrator 不维护跨轮 `rescore_queue` 状态。

#### 3.4.6 `noop` — 显式声明无池动作(Stage A1)

```python
@dataclass
class NoopBlock:
    type: Literal["noop"] = "noop"
    rationale: str                          # ≤ 200 chars
```

LLM 在 Stage A1 完全可以单独发 `noop`(当 LLM 觉得上轮 GP 选得很好,无需 propose/reject/analog)。但**当 `len(pool) < pool_min_size` 时,`noop` 必被拒** — orchestrator 反馈"必须 propose/analog 补池"。

### 3.5 类型联合

```python
LLMBlock = Union[
    ReviewBOBlock,
    ProposeBlock,
    RejectBlock,
    AnalogBlock,
    ReviewAnalogsBlock,
    NoopBlock,
]

# Stage-allow sets (authoritative — validator reads these, not literals)
PHASE_A_ACTIONS_ALLOWED:         tuple[str, ...] = ("propose", "reject", "analog", "noop")
PHASE_A_REVIEW_ANALOGS_ALLOWED:  tuple[str, ...] = ("review_analogs",)
PHASE_B_SUGGESTIONS_ALLOWED:     tuple[str, ...] = ("review_bo",)

def validate_blocks_phase(blocks: list[Block], stage: str) -> None:
    """stage ∈ {"A_actions", "A_review_analogs", "B_suggestions"}."""
    _STAGE_MAP = {
        "A_actions":         PHASE_A_ACTIONS_ALLOWED,
        "A_review_analogs":  PHASE_A_REVIEW_ANALOGS_ALLOWED,
        "B_suggestions":     PHASE_B_SUGGESTIONS_ALLOWED,
    }
    if stage not in _STAGE_MAP:
        raise ValueError(f"stage must be one of {sorted(_STAGE_MAP)}, got {stage!r}")
    allowed = set(_STAGE_MAP[stage])
    bad = [b for b in blocks if b.type not in allowed]
    if bad:
        types = sorted({b.type for b in bad})
        raise SemanticError(
            f"Stage {stage} disallows block types {types}; allowed: {sorted(allowed)}"
        )
```

---

## 4. Robustness:重试 + Fallback + Exception Catcher

### 4.1 错误分类

```python
class ParseError(Exception):          # re.findall 抽出 0 块,或 JSON 损坏
    pass

class SchemaError(Exception):         # jsonschema 校验失败(必出块缺、字段错、type 不匹配)
    pass

class SemanticError(Exception):       # 业务校验失败
    """Examples: SMILES RDKit 拒、reject targets 不在 pool、override 多个 BO slot
    都用同一 SMILES 时给出提示(去重后仍合法但需提示)、analog seeds 全部非法、
    块类型不属于本 stage 允许集合、noop 在 len(pool) < pool_min_size 时出现。"""
    pass
```

### 4.2 有限重试 + 错误反馈(每 stage 独立)

```python
class LLMAdvisor:
    def __init__(self, llm: LLMClient, max_retries: int = 3, **kwargs):
        self.llm = llm
        self.max_retries = max_retries

    def decide_actions(
        self, state: PreActionState,
    ) -> tuple[list[LLMBlock], list[LLMAttemptRecord], bool]:
        """Returns: (final_blocks, attempts, fallback_used)."""
        return self._decide_with_retry(
            state=state, stage="A_actions",
            system=SYSTEM_ACTIONS,
            user_renderer=render_user_actions,
            fallback_fn=fallback_actions,
            pool_min_size=state.pool_min_size,
        )

    def decide_review_analogs(
        self, state: PreReviewAnalogsState,
    ) -> tuple[list[LLMBlock], list[LLMAttemptRecord], bool]:
        """Returns: (final_blocks, attempts, fallback_used)."""
        return self._decide_with_retry(
            state=state, stage="A_review_analogs",
            system=SYSTEM_REVIEW_ANALOGS,
            user_renderer=render_user_review_analogs,
            fallback_fn=fallback_review_analogs,
        )

    def decide_review_suggestions(
        self, state: PostSuggestionState,
    ) -> tuple[list[LLMBlock], list[LLMAttemptRecord], bool]:
        """Returns: (final_blocks, attempts, fallback_used)."""
        return self._decide_with_retry(
            state=state, stage="B_suggestions",
            system=SYSTEM_REVIEW_SUGGESTIONS,
            user_renderer=render_user_suggestions,
            fallback_fn=fallback_review_suggestions,
        )

    def _decide_with_retry(self, *, state, stage, system,
                           user_renderer, fallback_fn,
                           pool_min_size=None):
        attempts: list[LLMAttemptRecord] = []
        previous_errors: list[str] = []

        for attempt_idx in range(1, self.max_retries + 1):
            t0 = time.monotonic()
            state_with_errors = dataclasses.replace(
                state, previous_errors=previous_errors, attempt=attempt_idx,
            )
            user_prompt = user_renderer(state_with_errors)
            try:
                raw = self.llm.chat(
                    system=system.format(round_idx=state.round_idx),
                    user=user_prompt, json_mode=True,
                )
            except Exception as exc:                           # 网络 / 超时
                err = f"transport: {type(exc).__name__}: {exc}"
                attempts.append(LLMAttemptRecord(
                    attempt=attempt_idx, raw_response="",
                    parsed_blocks=[], validation_errors=[err],
                    duration_ms=(time.monotonic() - t0) * 1000,
                ))
                previous_errors.append(err)
                continue

            try:
                blocks = parse_blocks(raw)                     # re.findall + json.loads
                validate_blocks_phase(blocks, stage)           # stage 允许集
                validate_semantics(
                    blocks,
                    pool=(state.pool or None) if stage == "A_actions" else None,
                    phase=stage,
                    pool_min_size=pool_min_size if stage == "A_actions" else None,
                )
                attempts.append(LLMAttemptRecord(
                    attempt=attempt_idx, raw_response=raw,
                    parsed_blocks=[b.to_dict() for b in blocks],
                    validation_errors=[],
                    duration_ms=(time.monotonic() - t0) * 1000,
                ))
                return blocks, attempts, False
            except (ParseError, SchemaError, SemanticError) as exc:
                err = format_error_for_prompt(exc)
                attempts.append(LLMAttemptRecord(
                    attempt=attempt_idx, raw_response=raw,
                    parsed_blocks=[], validation_errors=[err],
                    duration_ms=(time.monotonic() - t0) * 1000,
                ))
                previous_errors.append(err)
                LOGGER.warning("LLM stage=%s attempt %d/%d failed: %s",
                               stage, attempt_idx, self.max_retries, err)

        # 全部用完 → fallback
        blocks = fallback_fn(state)
        return blocks, attempts, True
```

**用户可见行为**:每次失败都会把错误信息写进 `state.previous_errors`,在下一轮 prompt 中显式呈现:

```
### Previous errors (you must fix these)
1. block 0 (review_bo): decisions missing for SMILES ["CCC"]
2. block 2 (reject): target "XYZ" not in pool
3. block 4 (review_analogs): "keep" verdict missing for analogue "..."
```

**跨 stage 隔离**:每 stage 的 `previous_errors` 互不共享。`state` 是 frozen dataclass,advisor 用 `dataclasses.replace` 每次构造新对象,绝不跨 stage 反馈。

### 4.3 Fallback 决策表(按 stage 拆分)

每个 stage 都有一个独立 fallback 函数,在 `max_retries` 用完后被调用。

| Stage | 触发 | Fallback 行为 |
|---|---|---|
| Stage A1 (actions) | LLM 持续输出非法(`max_retries` 用完) | 一个 `NoopBlock` — 本轮不动池;若 `len(pool) < pool_min_size`,pool-size loop 会在下一 iter 继续重试 |
| Stage A2 (review_analogs) | LLM 持续输出非法 | 一个 `ReviewAnalogsBlock` 把所有 `new_analogs` 全部判定为 `keep`(最大化池增长;若 LLM 不可靠时优先扩池) |
| Stage A2 | `new_analogs` 为空(Stage A1 没出 analog 块) | 不进入 Stage A2,`fallback_review_analogs` 不被调用 |
| Stage B (review_bo) | LLM 持续输出非法 | 一个 `ReviewBOBlock` 把所有 BO 候选判定为 `ok`(用 BO 原推荐) |
| Stage B | `bo_suggestions` 为空(GP 退化到 prior 或 pool 空) | `fallback_review_suggestions` 返回空列表;本轮无评分 |
| 通用 | 任何致命异常(ReaSyn 进程崩溃、磁盘写失败等) | Exception Catcher:落 `*.json` + `*.error.json` + re-raise |

> Stage A2 fallback 选 `keep-all`(而非 fail-closed reject-all)是有意设计:在 LLM 不可靠时优先扩池,配合 pool-size loop;`reject-all` 会让 Stage A1 的 analog 投入归零,违反"扩池是 LDM 价值核心"的设计意图。

### 4.4 Exception Catcher(致命异常)

```python
class OrchestratorError(Exception):
    """Orchestrator 主动抛出的致命异常;trajectory 已落盘。"""
    pass

def run_bo_with_llm(...) -> list:
    recorder = TrajectoryRecorder(trajectory_path)
    recorder.begin_run(...)
    try:
        for round_idx in range(n_iterations):
            with recorder.round_context(round_idx) as rr:
                _run_one_round(state, rr)         # 任何异常都会先 commit 再上抛
        return _format_history(history, n_obj)
    except Exception as exc:
        # 1. 落 trajectory(包含已完成的 rounds + 本次错误)
        recorder.record_fatal_error(
            round_idx=current_round,
            exc_type=type(exc).__name__,
            message=str(exc),
            traceback=traceback.format_exc(),
        )
        # 2. 落一个独立的 error 文件(便于监控 / 告警)
        recorder.dump_emergency_json()
        # 3. 再 raise
        raise
    finally:
        # 成功或失败都会写最终 JSON(见 §10)
        recorder.write_final()
```

**`record_fatal_error` 与 `dump_emergency_json`**:
- `record_fatal_error`:把异常追加到当前 round 的 errors 字段并 commit
- `dump_emergency_json`:把已 commit 的所有 round + 当前 round 写到独立的 `*.error.json` 文件,与正常 trajectory 同一目录
- 主 `trajectory.json` 在 `write_final()` 时根据 `state.status ∈ {completed, fatal_error}` 决定是否写 success 字段

---

## 5. 状态机(per BO round)

```
                ┌───────────────────────┐
                │ Round start            │
                │ pool_t                 │
                └──────────┬────────────┘
                           │
                           ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ Stage A1 — LLM 池管理 (LLM call #1,每轮必调)                │
   │   pre_state = PreActionState(pool_t, pool_min_size, ...)     │
   │   blocks_A1 = llm.decide_actions(pre_state)                 │
   │     可选: propose / reject / analog / noop                  │
   │     (noop 在 len(pool) < pool_min_size 时被拒,触发 loop)    │
   └────────────────────────────┬─────────────────────────────────┘
                                ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ Pool-size loop (orchestrator 内部)                           │
   │   应用 blocks_A1                                              │
   │     propose → pool.add(...)                                   │
   │     reject  → pool.discard(...)                               │
   │     analog  → run analog_fn(seeds) → new_analogs (local)     │
   │   if new_analogs:                                             │
   │     → 进入 Stage A2 (synchronous)                            │
   │   if len(pool) < pool_min_size and iter < max_pool_size_iters│
   │     → 把"必须 propose/analog 补池"错误塞进 prompt,重试 A1   │
   │   else:                                                       │
   │     → 退出 loop,进入 BO                                     │
   └────────────────────────────┬─────────────────────────────────┘
                                ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ Stage A2 — LLM review analogs (LLM call #2,条件调)          │
   │   仅当 Stage A1 产出非空 new_analogs 时才调                  │
   │   pre_state = PreReviewAnalogsState(pool_t', new_analogs)   │
   │   blocks_A2 = llm.decide_review_analogs(pre_state)          │
   │     必出 1 个: review_analogs                                │
   │     keep → pool.add(analogue)                                │
   │     reject / rescore → drop                                  │
   └────────────────────────────┬─────────────────────────────────┘
                                ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ BO 步骤 (LLM 不参与,每轮都跑)                                │
   │   fit GPSurrogate on history                                 │
   │   acquisition(EI/PI/UCB/EHVI/Chebyshev) → top-k             │
   │   bo_suggestions = top-k (在 mutated pool 上)                │
   └────────────────────────────┬─────────────────────────────────┘
                                ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ Stage B — LLM review_bo (LLM call #3,每轮必调)              │
   │   post_state = PostSuggestionState(pool_t', bo_suggestions) │
   │   blocks_B = llm.decide_review_suggestions(post_state)      │
   │     必出 1 个: review_bo                                      │
   │     (其他块丢弃并 warn)                                       │
   └────────────────────────────┬─────────────────────────────────┘
                                ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ 应用 review_bo                                                │
   │   final_candidates, overrides = apply_review_bo(...)        │
   └────────────────────────────┬─────────────────────────────────┘
                                ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ 评分                                                          │
   │   if final_candidates:                                       │
   │     scores = safe_score_n(scorers, final_candidates)         │
   │     history_t+1 = history_t ∪ scores                         │
   │     pool -= final_candidates                                 │
   └────────────────────────────┬─────────────────────────────────┘
                                ▼
                ┌───────────────────────┐
                │ Round end → next round│
                └───────────────────────┘
```

**关键设计变化**:
- **没有 `phase_a_period`**:Stage A1 **每轮必调**(LLDM 路径)。池扩展由 pool-size loop 处理(在同一 round 内循环),不跨 round 跳过。
- **没有 `rescore_queue`**:Stage A2 review 是**同步**(在同 round 的 Stage A1 之后),`rescore_with_different_params` 直接丢弃原产物;若 LLM 想重生成,下轮 Stage A1 自行发 `analog` 块。
- **没有 `pending_analogs` state 字段**:它是 Stage A1 内部的局部 `new_analogs`,不入 state,也不在 round 之间 carry。

**与现有循环的关系**:原 `bayesian_analog_search` 的 BO 主体 = `select_candidates → score → analog_fn(candidates)`;新流程把 `select_candidates` 推迟到 Stage A1(+A2)之后,把 `score` 推迟到 LLM override 之后(Stage B 后),把 `analog_fn` 拆成 LLM 决策(Stage A1 的 analog 块) + 同步 Stage A2 review。

---

## 6. JSON Schema

### 6.1 输入契约(`RoundState` → prompt 渲染,**按 stage 分三套**)

每 stage 都有独立的 system + user prompt 模板;共享的 `_FORMAT_HEADER`(响应格式说明)嵌入到 system prompt 顶部。

#### 6.1.1 Stage A1 system prompt (actions)

```python
SYSTEM_ACTIONS = _FORMAT_HEADER + """
You are a medicinal-chemistry co-pilot steering a Bayesian-optimization
loop over a SMILES candidate pool. The pool contains the current
candidates being searched — you should actively expand and curate it.

You are in STAGE A1 (actions) of round {round_idx}.
BO will run AFTER your decisions are applied. Your job is to decide
how to change the pool. Emit ONLY action blocks:
  - propose  (add new SMILES to the pool)
  - reject   (remove SMILES from the pool)
  - analog   (expand existing pool members via ReaSyn generation)
  - noop     (do nothing — only if pool is already large enough)

You MUST actively expand the pool if it is small. A noop is rejected
when pool size is below the minimum — the system will loop back to you.

EDGE CASE: pool is already large enough → emit a noop block.
"""
```

#### 6.1.2 Stage A2 system prompt (review analogs)

```python
SYSTEM_REVIEW_ANALOGS = _FORMAT_HEADER + """
You are a medicinal-chemistry co-pilot reviewing newly generated
ReaSyn analogues. Each analogue was produced by expanding an existing
pool member. For each analogue, decide whether to:
  - "keep"   — add it to the pool
  - "reject" — discard it
  - "rescore_with_different_params" — discard; LLM may re-generate next round

You are in STAGE A2 (review analogs) of round {round_idx}.
Emit exactly ONE review_analogs block with a decision for EVERY
analogue listed below.
"""
```

#### 6.1.3 Stage B system prompt (review suggestions)

```python
SYSTEM_REVIEW_SUGGESTIONS = _FORMAT_HEADER + """
You are a medicinal-chemistry co-pilot steering a Bayesian-optimization
loop over a SMILES candidate pool.

You are in STAGE B (review suggestions) of round {round_idx}.
BO has already run on the post-mutation pool. Output exactly ONE
review_bo block.

EDGE CASE: no BO suggestions → emit a review_bo block with EMPTY
decisions dict and a brief rationale.
"""
```

#### 6.1.4 Three per-stage user templates

```python
_USER_TEMPLATE_ACTIONS = """
## Round {round_idx}/{n_total_rounds} — Stage A1: actions
### Objective legend / Pool / History / GP summary / Pareto / Stagnation / Diversity
### PDF context
{pool_size_requirement_hint}    # 仅当 len(pool) < pool_min_size 时显示
### Response format reminder (allowed: propose / reject / analog / noop)
"""

_USER_TEMPLATE_REVIEW_ANALOGS = """
## Round {round_idx}/{n_total_rounds} — Stage A2: review analogs
### Newly generated analogues ({n_analogs} items)
{analogs_block}
### Response format reminder (allowed: review_analogs)
"""

_USER_TEMPLATE_SUGGESTIONS = """
## Round {round_idx}/{n_total_rounds} — Stage B: review suggestions
### Pool / History / GP summary / BO suggestions (top-{k_bo} from acq={acq_function})
### Pareto / Stagnation / Diversity / PDF context
### Response format reminder (allowed: review_bo)
"""
```

完整块定义 (review_bo / propose / reject / analog / review_analogs / noop) 与之前相同(见 §3.4);Stage A1 prompt 展示所有 4 个 action 块的 schema;Stage A2 只展示 `review_analogs`;Stage B 只展示 `review_bo`。

### 6.2 输出 schema(机器可校验)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "advisor_blocks.v1.json",
  "definitions": {
    "ReviewBOBlock": {
      "type": "object",
      "required": ["type", "rationale", "decisions"],
      "properties": {
        "type": {"const": "review_bo"},
        "rationale": {"type": "string", "maxLength": 600},
        "decisions": {
          "type": "object",
          "minProperties": 1,
          "additionalProperties": {
            "type": "string",
            "pattern": "^(ok|skip|override:.+)$"
          }
        }
      },
      "additionalProperties": false
    },
    "ProposeBlock": {
      "type": "object",
      "required": ["type", "rationale", "smiles"],
      "properties": {
        "type": {"const": "propose"},
        "rationale": {"type": "string", "maxLength": 400},
        "smiles": {
          "type": "array", "minItems": 1, "maxItems": 10,
          "items": {"type": "string", "pattern": "^[A-Za-z0-9@+\\-\\[\\]()=#$%/.]+$"}
        },
        "rationale_per_mol": {
          "type": "object",
          "additionalProperties": {"type": "string"}
        }
      },
      "additionalProperties": false
    },
    "RejectBlock": {
      "type": "object",
      "required": ["type", "rationale", "targets", "reason"],
      "properties": {
        "type": {"const": "reject"},
        "rationale": {"type": "string", "maxLength": 200},
        "targets": {
          "type": "array", "minItems": 1, "maxItems": 50,
          "items": {"type": "string"}
        },
        "reason": {
          "enum": [
            "too_similar_to_history",
            "likely_toxic",
            "synthetically_infeasible",
            "out_of_scope_pharmacophore",
            "no_signal_for_target"
          ]
        }
      },
      "additionalProperties": false
    },
    "AnalogBlock": {
      "type": "object",
      "required": ["type", "rationale", "seeds"],
      "properties": {
        "type": {"const": "analog"},
        "rationale": {"type": "string", "maxLength": 400},
        "seeds": {
          "type": "array", "minItems": 1, "maxItems": 10,
          "items": {"type": "string"}
        },
        "generator_hint": {
          "enum": ["conservative", "aggressive", "scaffold_hop", null]
        },
        "n_per_seed": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
        "reasyn_config_override": {
          "type": ["object", "null"],
          "properties": {
            "search_width": {"type": "integer", "minimum": 1, "maximum": 64},
            "num_cycles": {"type": "integer", "minimum": 1, "maximum": 32},
            "num_editflow_samples": {"type": "integer", "minimum": 1, "maximum": 500},
            "time_limit": {"type": "integer", "minimum": 10, "maximum": 3600}
          },
          "additionalProperties": false
        }
      },
      "additionalProperties": false
    },
    "ReviewAnalogsBlock": {
      "type": "object",
      "required": ["type", "rationale", "decisions"],
      "properties": {
        "type": {"const": "review_analogs"},
        "rationale": {"type": "string", "maxLength": 400},
        "decisions": {
          "type": "object",
          "minProperties": 1,
          "additionalProperties": {
            "type": "string",
            "enum": ["keep", "reject", "rescore_with_different_params"]
          }
        }
      },
      "additionalProperties": false
    },
    "NoopBlock": {
      "type": "object",
      "required": ["type", "rationale"],
      "properties": {
        "type": {"const": "noop"},
        "rationale": {"type": "string", "maxLength": 200}
      },
      "additionalProperties": false
    }
  },
  "oneOf": [
    {"$ref": "#/definitions/ReviewBOBlock"},
    {"$ref": "#/definitions/ProposeBlock"},
    {"$ref": "#/definitions/RejectBlock"},
    {"$ref": "#/definitions/AnalogBlock"},
    {"$ref": "#/definitions/ReviewAnalogsBlock"},
    {"$ref": "#/definitions/NoopBlock"}
  ]
}
```

注:LLM 一次响应可包含**多个**这样的块,每个块独立按上表校验。`validate_blocks_phase(blocks, phase)` 进一步按阶段过滤;非法阶段块抛 `SemanticError`。

---

## 7. Orchestrator 主流程(伪代码)

```python
# llm_advisor/orchestrator.py
def run_bo_with_llm(
    *, seed_smiles, scorer, llm, analog_fn, reasyn_pool, config, trajectory_path,
):
    path = resolve_trajectory_path(trajectory_path, method=config.method, seed=config.seed)
    recorder = TrajectoryRecorder(path=path, method=config.method, seed=config.seed)
    recorder.begin_run(config=_config_to_dict(config), llm_model=llm.model_name)
    advisor = LLMAdvisor(llm=llm, max_retries=3, use_rdkit=True)

    # init pool + history
    pool: list[str] = list(dict.fromkeys(seed_smiles))     # dedup, preserve order
    history: OrderedDict[str, tuple[float, ...]] = OrderedDict()
    last_gp_summary = GPSummary(n_train=0, in_prior_mode=True)
    stagnation_counter, last_best = 0, None

    try:
        # ---- init: LDM 路径保留 seed SMILES 在 pool(让 Stage A1 扩增)----
        pool_min = config.pool_min_size
        if pool_min > 1:
            pass        # LDM: pool is seeds; Stage A1's pool-size loop will expand
        else:
            init_chosen = list(pool)[: config.init_size]
            for smi in init_chosen:
                pool.remove(smi)
            init_scores = _score_via_scorer(scorer, init_chosen)
            for s in init_chosen:
                history[s] = (init_scores[s],)

        # ---- BO rounds ----
        for round_idx in range(config.n_iterations):
            with recorder.round_context(round_idx) as rr:
                _run_one_round(
                    round_idx=round_idx, config=config, pool=pool,
                    history=history, last_gp_summary=last_gp_summary,
                    stagnation_counter=stagnation_counter, last_best=last_best,
                    scorer=scorer, llm=llm, advisor=advisor,
                    analog_fn=analog_fn, reasyn_pool=reasyn_pool, recorder=rr,
                )
            last_gp_summary = _last_summary_from(rr)

        recorder.set_status("completed")
        recorder.set_final_history([(s, sc[0]) for s, sc in history.items()])
        return [(s, sc[0]) for s, sc in history.items()]
    except Exception as exc:
        recorder.record_fatal_error(round_idx=recorder.current_round, exc=exc)
        recorder.set_final_history([(s, sc[0]) for s, sc in history.items()])
        recorder.dump_emergency_json()
        raise
    finally:
        recorder.write_final()


def _run_one_round(
    *, round_idx, config, pool, history, last_gp_summary,
    stagnation_counter, last_best, scorer, llm, advisor,
    analog_fn, reasyn_pool, recorder,
):
    pool_min = config.pool_min_size
    all_attempts_A1: list[LLMAttemptRecord] = []
    all_attempts_A2: list[LLMAttemptRecord] = []
    final_blocks_A1: list[LLMBlock] = []
    final_blocks_A2: list[LLMBlock] = []
    fb_A1 = fb_A2 = False

    # Build Stage A1 snapshot (taken once, mutated inside the loop)
    snap_dict = _snapshot_actions(pool, history, last_gp_summary, config, round_idx)
    action_state = _action_state_from_snapshot(snap_dict)
    recorder.pre_state_snapshot = snap_dict

    # === Pool-size loop: Stage A1 (+ optional A2) ===
    for _iter in range(config.max_pool_size_iters):
        blocks_A1, attempts_A1, fb_A1 = advisor.decide_actions(action_state)
        all_attempts_A1.extend(attempts_A1)
        final_blocks_A1 = blocks_A1

        # Apply A1 actions: propose / reject / analog → returns new_analogs (local)
        new_analogs = _apply_actions(
            blocks=blocks_A1, pool=pool,
            analog_fn=analog_fn, reasyn_pool=reasyn_pool,
        )

        # === Stage A2 (synchronous, only if new_analogs non-empty) ===
        if new_analogs:
            review_state = _build_review_analogs_state(
                action_state=action_state, new_analogs=new_analogs,
                pool=pool, history=history, round_idx=round_idx, config=config,
            )
            blocks_A2, attempts_A2, fb_A2 = advisor.decide_review_analogs(review_state)
            all_attempts_A2.extend(attempts_A2)
            final_blocks_A2 = blocks_A2
            _apply_review_analogs(blocks=blocks_A2, pool=pool, new_analogs=new_analogs)

        # Exit loop if pool is large enough OR we hit a fallback (give up)
        if len(pool) >= pool_min or fb_A1:
            break

        # Inject pool-size error for next iteration of the loop
        LOGGER.warning("Pool-size loop iter %d: pool has %d SMILES < min %d", ...)
        pool_err = SemanticError(
            f"pool has {len(pool)} SMILES (< min {pool_min}); "
            f"you MUST emit `propose` with new SMILES or `analog` to expand."
        )
        action_state = dataclasses.replace(
            action_state, pool=tuple(pool),
            previous_errors=tuple(
                list(action_state.previous_errors) + [format_error_for_prompt(pool_err)]
            ),
            attempt=1,
        )

    # Record stage_a1 / stage_a2 in the trajectory
    recorder.llm_interactions["stage_a1"] = {
        "executed": True,
        "attempts": serialize_attempts(all_attempts_A1),
        "fallback_used": fb_A1,
        "final_blocks": serialize_blocks(final_blocks_A1),
        "pool_size_loop_final_pool_size": len(pool),
    }
    recorder.llm_interactions["stage_a2"] = {
        "executed": bool(all_attempts_A2),
        "attempts": serialize_attempts(all_attempts_A2),
        "fallback_used": fb_A2,
        "final_blocks": serialize_blocks(final_blocks_A2),
    }
    recorder.pool_after_phase_a = list(pool)

    # === BO step (every round) ===
    rng = RNG(seed=config.seed + round_idx)
    pick_records, summary = _run_bo_step(
        pool=pool, history=history, bo_config=config.bo_config,
        rng=rng, top_k=config.batch_size,
    )
    recorder.bo_suggestions = [p.to_dict() for p in pick_records]
    global _LAST_GP_SUMMARY
    _LAST_GP_SUMMARY = summary

    # === Stage B (every round) ===
    post_snap = _snapshot_suggestions(
        pool=pool, history=history, bo_picks=pick_records,
        acq_function=str(config.bo_config.acquisition), summary=summary,
        config=config, round_idx=round_idx,
    )
    post_state = _suggestion_state_from_snapshot(post_snap)
    blocks_B, attempts_B, fb_B = advisor.decide_review_suggestions(post_state)
    review_bo_block = _first_of_type(blocks_B, "review_bo")
    final_candidates, overrides = _apply_review_suggestions(review_bo_block, pick_records)
    recorder.llm_interactions["stage_b"] = {
        "executed": True, "attempts": serialize_attempts(attempts_B),
        "fallback_used": fb_B, "final_blocks": serialize_blocks(blocks_B),
        "review_bo_block": review_bo_block.to_dict() if review_bo_block else None,
        "final_candidates": final_candidates, "overrides": overrides,
    }

    # === Score + remove from pool ===
    if final_candidates:
        scores = _score_via_scorer(scorer, final_candidates)
        for s in final_candidates:
            history[s] = (scores[s],)
            if s in pool: pool.remove(s)
        recorder.scores = {s: ([float(scores[s])] if scores[s] is not None else [None])
                          for s in final_candidates}
    else:
        recorder.scores = {}

    # Update stagnation / best
    new_best = min(
        (sc[0] for sc in history.values() if sc and sc[0] is not None), default=None,
    )
    if new_best is not None and (last_best is None or new_best < last_best):
        last_best = new_best
        stagnation_counter = 0
    else:
        stagnation_counter += 1

    recorder.pool_after = list(pool)
```

---

## 8. LLM Client 抽象(OpenAI 兼容)

```python
# llm_advisor/client.py
class LLMClient(Protocol):
    model_name: str

    def chat(self, system: str, user: str, *, json_mode: bool = True) -> str: ...

class OpenAIChatClient:                     # 默认实现
    def __init__(self, model: str, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.2,
                 max_retries: int = 3, timeout: float = 60): ...

class MockLLMClient:                        # 测试/回放
    def __init__(self, scripted: list[list[LLMBlock]]): ...
    # scripted[i] 是第 i 轮的 block 列表(每轮两个 phase 共用)
```

**调用约束**:

- `chat(json_mode=True)` 强制模型输出合法 JSON(OpenAI 的 `response_format={"type":"json_object"}`)
- **多块解析**:orchestrator 用 `re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)` 抽出所有 JSON 块,逐块 schema 校验
- 超时 / JSON 解析失败 / schema 校验失败 → **进入 advisor 内部重试**(见 §4),不直接算 fatal

---

## 9. 文件落地清单(已建)

```
strbo_v1/
  llm_advisor/
    __init__.py
    client.py            # LLMClient + OpenAIChatClient + MockLLMClient
    blocks.py            # 6 个 Block dataclass + LLMBlock 联合 + to_dict()
                          # + PHASE_A_ACTIONS_ALLOWED /
                          #   PHASE_A_REVIEW_ANALOGS_ALLOWED /
                          #   PHASE_B_SUGGESTIONS_ALLOWED
    state.py             # GPSummary / PickRecord / AnalogueRecord
    round_state.py       # PreActionState / PreReviewAnalogsState / PostSuggestionState
    schema.py            # JSON schema 字符串(可被 jsonschema 加载)
    parser.py            # parse_blocks + 异常类(ParseError/SchemaError/SemanticError)
                          + validate_blocks_phase(blocks, stage)
    prompt.py            # SYSTEM_ACTIONS / SYSTEM_REVIEW_ANALOGS /
                          # SYSTEM_REVIEW_SUGGESTIONS +
                          # render_user_actions / render_user_review_analogs /
                          # render_user_suggestions
    fallback.py          # fallback_actions / fallback_review_analogs /
                          # fallback_review_suggestions
    advisor.py           # LLMAdvisor(decide_actions / decide_review_analogs /
                          # decide_review_suggestions;共一个 _decide_with_retry 私有)
    trajectory.py        # TrajectoryRecorder + RoundRecord + LLMAttemptRecord
    reasyn_pool.py       # ReasynConfigPool + pick_reasyn_config(...)
    config.py            # LLMClientConfig + load_env() + DEFAULT_LLM_MODEL
    orchestrator.py      # run_bo_with_llm(...) ← 唯一 public 入口
                          # (含 pool-size loop、_run_one_round、snapshot builders)
  bayesian_ldm_search.py # bayesian_ldm_search() ← run_search.py 的 bo-*-ldm 入口

docs/
  llm_advisor_design.md  # 本设计文档

tests/
  test_llm_blocks.py            # dataclass 序列化 + 三个 stage allow set
  test_llm_parser.py            # parse_blocks + validate_blocks_phase + 业务校验
  test_llm_fallback.py          # 三 stage fallback 决策表覆盖
  test_llm_advisor.py           # MockLLMClient 跑 retry + fallback + 跨 stage 隔离
  test_llm_orchestrator.py      # 端到端(用 mock LLM + mock scorer + pool-size loop)
  test_llm_prompts.py           # 三 stage 各自的 system + user prompt
  test_llm_client.py            # OpenAI 客户端 + Mock 客户端
  test_llm_trajectory.py        # 录盘后 re-parse 还原
  test_bayesian_ldm_search.py   # bayesian_ldm_search() 公共入口
```

**`strbo_v1/bayesian_analog_search.py` / `acquisition.py` / `gp.py` 全部不改**。

---

## 10. Trajectory 记录(单一最终 JSON + Exception Catcher)

### 10.1 输出文件约定(对齐 `run_search.py`)

CLI 形如(经由 `run_search.py` 入口,bo-*-ldm 方法):
```bash
python run_search.py --method bo-tanimoto-ldm --seed 0 \
    --output output/bo_ldm/ \
    --llm-trajectory-dir output/bo_ldm  # 默认 = 写 tmpdir 并清理
```

**`resolve_trajectory_path(trajectory_dir, method, seed)` 语义**:
- 目录 → `{trajectory_dir}/{method}_seed={seed}_trajectory.json`
- 文件路径 → 原文使用

### 10.2 JSON 文件结构(单一文件,运行结束时写)

```jsonc
{
  "status": "completed" | "fatal_error",
  "run_metadata": {
    "method": "bo-tanimoto-ldm",
    "seed": 0,
    "llm_model": "DeepSeek-V4-Flash",
    "started_at": "2026-06-25T14:00:00Z",
    "finished_at": "2026-06-25T14:23:17Z",
    "duration_seconds": 1397.4
  },
  "config": { /* BayesianAnalogSearchConfig + LLMClientConfig + OrchestratorConfig 的 echo */ },
  "history": [
    // 与 run_search.write_json 兼容
    {"index": 0, "smiles": "CCO", "score": -7.2},
    ...
  ],
  "rounds": [
    {
      "round_idx": 0,
      "timestamp": "2026-06-25T14:00:30Z",
      // ---- 三 stage 拆分 ----
      "llm_interactions": {
        "stage_a1": {
          "executed": true,                          // Stage A1 必调
          "attempts": [
            {
              "attempt": 1,
              "raw_response": "<full LLM output text>",
              "parsed_blocks": [...],
              "validation_errors": [],
              "duration_ms": 1823.4
            }
            // 失败重试会追加 attempt
          ],
          "fallback_used": false,
          "final_blocks": [
            {"type": "propose", "smiles": ["..."], ...},
            {"type": "analog", "seeds": ["..."], ...}
          ],
          "pool_size_loop_final_pool_size": 12      // pool-size loop 退出时的 pool 大小
        },
        "stage_a2": {
          "executed": false,                         // 当 Stage A1 没产出 analog 时 false
          "attempts": [...],                         // 调过才有内容
          "fallback_used": false,
          "final_blocks": [...]
        },
        "stage_b": {
          "executed": true,
          "attempts": [...],
          "fallback_used": false,
          "final_blocks": [
            {"type": "review_bo", "rationale": "...", "decisions": {...}}
          ],
          "review_bo_block": {...},
          "final_candidates": ["CCO", "CCN", "CC(=O)Oc1ccccc1C(=O)O"],
          "overrides": {"CCC": "CC(=O)Oc1ccccc1C(=O)O"}
        }
      },
      // ---- Stage A1 起点快照(给 pool-size loop 用)----
      "pre_state_snapshot": {
        "pool": ["CCO", "CCN", ...],
        "history": [...],
        "gp_summary": {...}
      },
      // ---- Stage A1(+A2) 后的 pool + BO 步骤 ----
      "pool_after_phase_a": [...],
      "bo_suggestions": [
        {"smiles": "CCO", "mu": -7.1, "sigma": 0.3, "acq_value": 0.45,
         "nearest_history": [["CCN", -6.9], ...]},
        ...
      ],
      // ---- 评分 ----
      "scores": {"CCO": [-7.2], "CCN": [-6.9], "CC(=O)Oc1ccccc1C(=O)O": [-8.1]},
      // ---- 准备下一轮 ----
      "pool_after": [...],
      // ---- 健康度 ----
      "warnings": [],
      "errors": []
    },
    ...
  ],
  "fatal_error": {  // 仅在 status == "fatal_error" 时存在
    "round_idx": 7,
    "exc_type": "ReaSynSubprocessError",
    "message": "...",
    "traceback": "..."
  }
}
```

### 10.3 异常时的双写策略(Exception Catcher)

| 场景 | 主文件 | 旁路文件 | 状态 |
|---|---|---|---|
| 正常运行完成 | `*.json` 写一次(完整) | 不写 | `status=completed` |
| 致命异常 | `*.json` 写一次(含已完成的 rounds + `fatal_error` 字段) | `*.error.json` 单独写(同样的内容,便于监控脚本快速识别) | `status=fatal_error` |
| LLM 部分失败(重试用完) | `*.json` 写(每个 round 内 `fallback_used=true`) | 不写 | `status=completed` |
| 任何 round 抛错 | `*.json` 写 + `dump_emergency_json` | 同上 | `status=fatal_error` |

**写入时机**:
- 每个 round 结束 → `recorder.commit_in_flight()` 写一个 `.in_flight.json`(临时,崩溃可恢复)
- 正常结束 → `recorder.write_final()` 删 `.in_flight.json`,写正式 `*.json`
- 异常 → `recorder.write_final()` 同样写(带 fatal_error),同时 `dump_emergency_json` 写 `*.error.json`

### 10.4 TrajectoryRecorder 接口

`RoundRecord` 在 `__enter__` 时返回,orchestrator 在每 stage 结束后**直接给 `recorder.llm_interactions[stage_key]` 赋值**(用 dict-of-dict 协议,而不是显式 `record_*` 方法)。这简化了 recorder 内部状态机。

```python
class TrajectoryRecorder:
    def __init__(self, path: str | Path, *, method: str, seed: int): ...
    @contextmanager
    def round_context(self, round_idx: int) -> Iterator[RoundRecord]: ...
    def begin_run(self, config: dict, llm_model: str): ...
    def set_final_history(self, history: list): ...
    def commit_in_flight(self): ...                    # 每次 round_context 退出时
    def record_fatal_error(self, *, round_idx, exc): ...
    def dump_emergency_json(self): ...                 # 写 *.error.json
    def set_status(self, status: Literal["completed", "fatal_error"]): ...
    def write_final(self): ...                          # 写正式 *.json

@dataclass
class RoundRecord:
    round_idx: int
    timestamp: str
    pre_state_snapshot: dict
    pool_after_phase_a: list
    bo_suggestions: list
    llm_interactions: dict          # {"stage_a1": {...}, "stage_a2": {...}, "stage_b": {...}}
    scores: dict
    pool_after: list
    warnings: list
    errors: list
```

`recorder.llm_interactions[stage_key]` 字段:
- `stage_a1`:`{executed: true, attempts, fallback_used, final_blocks, pool_size_loop_final_pool_size}`
- `stage_a2`:`{executed: true|false, attempts, fallback_used, final_blocks}`
- `stage_b`:`{executed: true, attempts, fallback_used, final_blocks, review_bo_block, final_candidates, overrides}`

---

## 11. 风险与缓解

| 风险 | 缓解 |
|---|---|
| LLM 输出非合法 JSON | `json_mode=True` + 后端 `response_format`;`re.findall` 抽多块;逐块 `jsonschema` 校验;**进入 advisor 内部重试**,错误信息反馈到下一轮 prompt |
| LLM 幻觉 SMILES(化学不合法) | RDKit `MolFromSmiles` 过滤;非法条目丢弃(逐块) |
| LLM 提议的 SMILES 与历史/pool 重复 | orchestrator 去重后入 pool |
| LLM 漏写必出块(review_bo / review_analogs) | 必出块缺失 → 自动补 `ok` / 默认 verdict;`warn_once` |
| LLM override 多个 BO slot 都用同一 SMILES | 允许,去重后评分;`SemanticError` 提示 |
| LLM analog seeds 指向非法 SMILES | 丢弃非法 seed,其余继续 |
| LLM 不审 ReaSyn 产物 | Stage A2 fallback **keep-all**(扩池优先);`warn_once` |
| LLM 持续输出非法 | `max_retries=3` 用完 → 三 stage 各自的 fallback |
| 致命异常(ReaSyn 进程崩溃、磁盘写失败) | **Exception Catcher**:落 `*.json` + `*.error.json` + re-raise |
| 大量 token 消耗 | 池/历史在 prompt 渲染前截断(pool≤50, history≤20, PDF≤1500 tokens) |
| LLM 调用阻塞 BO | `decide_actions / review_analogs / review_suggestions` 默认同步;可换 `AsyncLLMClient` + 并行评分 |
| **Stage A1 + A2 + B 每轮最多 3 次 LLM 调用,token ×3、latency ×3** | pool-size loop 上限 `max_pool_size_iters=5` 兜底;Stage A2 仅在 analog 非空时调 |
| **LLM 误把 `review_bo` 块写到 Stage A1 / A2** | `validate_blocks_phase(blocks, stage)` 严格枚举 stage-allow set,抛 `SemanticError`,丢弃并 warn |
| **池管块写到 Stage B** | 同上 |
| **跨 stage 的 retry 预算污染** | 每 stage 独立 `max_retries` 与 `previous_errors`(frozen dataclass + `dataclasses.replace` 每次构造新对象),绝不跨 stage 反馈 |
| **pool < pool_min_size 时 LLM 死循环发 noop** | pool-size loop:`SemanticError("pool has X SMILES < min Y; must propose/analog")` 反馈到下一 iter;上限 5 iter 兜底 |
| **Stage A2 看到 mutated pool 但 LLM 误以为是自己改的** | `SYSTEM_REVIEW_ANALOGS` 显式声明"你 review 的 analogues 是 Stage A1 产出的,pool 已被 Stage A1 改过" |
| **Stage B 看到 mutated pool 但 LLM 仍按旧 pool 推理** | `SYSTEM_REVIEW_SUGGESTIONS` 显式声明"本 stage 看到的是 Stage A1(+A2) 改完后的 pool;BO 已在 mutated pool 上跑过" |
| **Stage A1 内部 pool-size loop 把 Stage A2 误跳过** | 每次 iter 都重新调用 `advisor.decide_actions` 并把"必须补池"作为新 `previous_errors` 注入;只有 pool 达标或 fallback 才退出 |

---

## 12. 验收标准(已达)

1. **MockLLMClient 跑通**:用脚本化 LLM 跑完 N-round 合成数据,输出 history 与纯 GP 对照(应不差)
2. **OpenAIChatClient 真模型跑通**:3-round 小规模(10 init + 5 BO),输出截图 + token 消耗统计
3. **`pytest tests/test_llm_*.py` + `tests/test_bayesian_ldm_search.py` 全绿**(244+ tests),覆盖:
   - 非法 JSON / 部分 JSON 块合法部分非法
   - 非 RDKit 合法 SMILES
   - 漏出 review_bo / 漏出 review_analogs
   - override SMILES 不在 pool 中
   - 多块输出、顺序错乱
   - ReaSynConfig 超参越界
   - **Stage A1 出 review_bo 块 → SemanticError**
   - **Stage A1 出 review_analogs 块 → SemanticError**
   - **Stage B 出 propose / reject / analog 块 → SemanticError**
   - **三 stage previous_errors 互不污染**
   - **Stage A1 noop 在 `len(pool) < pool_min_size` 时被拒**
   - **Pool-size loop 在 max_pool_size_iters 内重试补池**
4. **Exception Catcher 验证**:模拟 BO step 崩溃,确认 `*.json` 与 `*.error.json` 都被写、且 re-raise
5. **Fallback 验证**:MockLLMClient 永远返回非法,确认三 stage 都走 fallback 完成所有 round
6. **Trajectory 还原**:re-parse `*.json` 后能精确还原每轮的 `pre_state_snapshot` / `pool_after_phase_a` / `bo_suggestions` / `final_blocks`(stage_a1/a2/b)/ `scores` 序列
7. **Pool-size loop 集成测试**:`pool_min_size=3` + 1 初始 SMILES,验证 Stage A1 至少被调 2 次才能让 pool 达标

---

## 13. 后续可拓展(本设计不实现,留口子)

- **多目标权重的 LLM 调制**:在 n_obj ≥ 2 时让 LLM 调整 `ref_point` 与 `minimize` 软权重(可作为 `acq_override` 拓展)
- **Stagnation 主动重启**:连续 N 轮无新 best 时由 LLM 触发"硬重启"——把 pool 清空,从 PDF 抽取的 SMILES 与 LLM propose 的新种子重注
- **跨目标迁移**:把不同 target 的 Pareto 前沿缓存进 prompt,引导 scaffold 复用
- **Embedding-based memory**:用 GP 内核距离把历史 SMILES 投影到 2D,LLM 视觉化决策

## 14. Public API and `run_search` integration

### 14.1 Public entry point: `bayesian_ldm_search`

`strbo_v1/bayesian_ldm_search.py` exposes a single public function,
:func:`bayesian_ldm_search`, that mirrors the shape of
:func:`strbo_v1.bayesian_analog_search.bayesian_analog_search`:

```python
history, trajectory = bayesian_ldm_search(
    seed_smiles=...,
    scorer=...,
    analog_fn=...,
    config=BayesianLDMSearchConfig(...),
    rng=None,            # optional
    llm=None,            # optional; default = OpenAIChatClient(config.llm_config)
)
```

* `history` is a list of `(smi, score_or_scores)` tuples in
  evaluation order, matching the existing `bayesian_analog_search`
  return contract (float for `n_obj == 1`, tuple for `n_obj >= 2`).
* `trajectory` is the LLM advisor's per-round trajectory dict
  (mirroring `strbo_v1.llm_advisor.trajectory.TrajectoryRecorder`'s
  final JSON shape), or `None` if no `trajectory_dir` was provided.
* `llm` is an optional pre-built LLM client (e.g. `MockLLMClient`
  for tests). If `None`, an `OpenAIChatClient` is constructed from
  `config.llm_config`.

### 14.2 Method names: `bo-tanimoto-ldm` and `bo-strkernel-ldm`

Two new methods in `run_search.py`:

* `bo-tanimoto-ldm`: Tanimoto fingerprint GP + LDM advisor
* `bo-strkernel-ldm`: SMILES string-kernel GP + LDM advisor

The GP impl is auto-derived from the method suffix; users do **not**
need to set `--llm-gp-impl` separately.

### 14.3 New CLI flags (bo-*-ldm only)

```
--llm-model              str   default=DeepSeek-V4-Flash (hardcoded)
--llm-base-url           str   default="" (= use env LLM_BASE_URL)
--llm-api-key            str   default="" (= use env LLM_API_KEY)
--pool-min-size          int   default=1; reused for LDM (auto-set to
                                  --batch-size if not provided)
--llm-trajectory-dir     str   default="" (= no sidecar; main JSON
                                  still has llm_trajectory via a
                                  tempdir managed by run_search)
```

**Removed in three-stage refactor**:
- `--llm-phase-a-period` — Stage A1 runs every round for LDM (no skipping)
- `--llm-warmup-init-pool` — replaced by pool-size loop inside Stage A1

### 14.4 Environment variables

Credentials live in `.env` (loaded via `dotenv` by
`strbo_v1.llm_advisor.config.load_env`):

* `LLM_API_KEY`
* `LLM_BASE_URL`

**There is no `LLM_MODEL` env var.** The model is a code-level
choice, hardcoded to `DeepSeek-V4-Flash` in
`strbo_v1.llm_advisor.config.DEFAULT_LLM_MODEL`. The `--llm-model`
CLI flag overrides it for individual runs.

`.env` is the single source of truth. Two-tier fallback: env vars
in `os.environ` take precedence (via `load_dotenv(override=False)`);
`.env` provides the default values when env vars are unset. If
`.env` is missing AND env vars are unset, `LLMClientConfig.from_env()`
raises a clear `ValueError("LLM_API_KEY is empty; set it in .env or
export LLM_API_KEY in the environment")` on the next access.

### 14.5 Trajectory merge into the main JSON

For `bo-*-ldm` methods, the main JSON gains a top-level
`"llm_trajectory"` key holding the per-round LLM/BO log:

```json
{
  "config": {...},
  "history": [...],
  "llm_trajectory": {
    "status": "completed",
    "run_metadata": {"method": "bo-tanimoto-ldm", "seed": 0, ...},
    "rounds": [
      {
        "round_idx": 0,
        "llm_interactions": {
          "stage_a1": {"final_blocks": [...], "fallback_used": false,
                       "pool_size_loop_final_pool_size": 12, ...},
          "stage_a2": {"executed": true|false, ...},
          "stage_b":  {"review_bo_block": {...}, "final_candidates": [...]}
        },
        "bo_suggestions": [...],
        "scores": {...},
        ...
      },
      ...
    ]
  }
}
```

This is read by `plot_search_results.py` as a no-op (it only uses
`config` and `history`); the trajectory is the audit trail for
debugging the LLM advisor's behaviour.

### 14.6 `run_search.sh` configuration

The shell script has a `METHODS=()` array and an `LLM_*` block:

```bash
LLM_MODEL="DeepSeek-V4-Flash"     # local var; default for --llm-model
LLM_POOL_MIN_SIZE=10              # "" = auto-set to --batch-size for LDM
LLM_TRAJECTORY_DIR=""             # path; "" = no sidecar

METHODS=("bo-tanimoto" "bo-strkernel" "bo-tanimoto-ldm" "bo-strkernel-ldm")
```

**Removed in three-stage refactor**: `LLM_PHASE_A_PERIOD` (Stage A1
runs every round), `LLM_WARMUP_INIT_POOL` (replaced by Stage A1's
pool-size loop). `--pool-min-size` is now reused for both random
search and LDM (single flag for both pool-refill paths).

Edit `METHODS=()` to choose which experiments to run. The per-seed
loop dispatches via `case "$METHOD"` so the same script supports
any subset of the 6 methods.

### 14.7 `plot_search_results.py` integration

The plotter gains two new entries in `METHOD_COLORS` and
`METHOD_LABELS`:

```python
"bo-tanimoto-ldm":  "tab:red",
"bo-strkernel-ldm": "tab:purple",
```

No other plotter changes — the `n_obj` dispatch (single-obj BSF /
2-obj HV / 3+ per-obj BSF) is method-agnostic and reads from
`config.n_objectives` in the JSON.
