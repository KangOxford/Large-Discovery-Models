# bo/ldm — Large Discovery Model (BO + LLM)

LLM-controlled Bayesian Optimisation for CDRH3 antibody sequence design.

## What it does

Each BO iteration:

1. The orchestrator asks an LLM for a Trust Region DSL (where to sample)
   and a Bias DSL (how to score candidates).
2. The LLM may output `{}` to keep the current DSLs, or update one or
   both fields independently.
3. The orchestrator validates each DSL:
   - **TR**: depth ≤ 8, |TR| ≤ 1M sequences, iter timeout ≤ 10s
   - **Bias**: depth ≤ 4
4. If validation fails, the orchestrator retries (default 3 times) with
   the failure message fed back to the LLM.
5. After all retries fail: **fallback to original AntBO** (no TR filter,
   no bias); bias is preserved (cleared only via explicit LLM update).
6. DSL is applied to BO: TR filters candidates, bias shifts acquisition.

## Architecture

```
bo/ldm/
├── __init__.py              ← public API (whitelist)
├── config.py                ← DSLConfig dataclass (all params)
├── dsl/                     ← PRIVATE — atom classes & sandbox
│   ├── search_space.py      ← SearchSpaceAtom + HammingDistanceTo, Match, And, Or, Not
│   ├── bias.py              ← BiasAtom + MaxCysteine, MaxHydrophobicRun, ...
│   ├── operators.py         ← &, |, ~ operator binding
│   ├── sandbox.py           ← safe_exec_dsl (restricted namespace exec)
│   ├── validator.py         ← depth / size / timeout checks
│   ├── sampler.py           ← reject sampling
│   ├── iter_with_cap.py     ← bounded iter for size validation
│   └── exceptions.py        ← DSLSyntaxError, TRTooLarge, TRTimeout, ...
├── orchestrator/            ← PRIVATE — main loop & prompt
│   ├── status.py            ← OrchestratorStatus
│   ├── prompts.py           ← system + user prompt builders
│   ├── decision_log.py      ← append-only JSON file
│   ├── loop.py              ← Orchestrator (PUBLIC via __init__.py)
│   └── fallback.py          ← fallback_to_original_antbo
├── llm/                     ← PRIVATE — LLM clients
│   ├── client.py            ← LLMClient ABC (PUBLIC via __init__.py)
│   ├── litellm_backend.py   ← LiteLLMClient (concrete)
│   └── response_parser.py   ← parses {"update_trust_region"?: ..., "update_bias"?: ...}
├── prompts/
│   └── system.txt           ← 4-section system prompt template
└── integrate.py             ← BRIDGE: build_status / apply_decision / sample_candidates / score_with_bias
```

## Public API (`from bo.ldm import ...`)

```python
from bo.ldm import (
    DSLConfig,                      # config dataclass (all params)
    SearchSpaceAtom,                # ABC (concrete atoms are private)
    BiasAtom,                       # ABC
    Orchestrator,                   # main LDM controller
    OrchestratorStatus,             # input to step()
    OrchestratorDecision,           # output of step()
    LLMClient,                      # ABC for LLM backend
    build_status,                   # helpers (bo/ldm/integrate.py)
    apply_decision,
    sample_candidates,
    score_with_bias,
)
```

Anything else is PRIVATE and must NOT be imported from `bo/`.

## Quick start

```python
from bo.ldm import DSLConfig, Orchestrator
from bo.ldm.llm.litellm_backend import LiteLLMClient

config = DSLConfig.from_yaml(yaml_dict["ldm"])
client = LiteLLMClient()
orch = Orchestrator(config=config, llm_client=client,
                    decision_log_path="outputs/llm_decisions/exp.json")

# In BO loop, per iteration:
status = build_status(cat, antigen_id="1ADQ_A", antigen_seed=42, iter_seed=0)
decision = orch.step(status)
apply_decision(cat, decision)
# decision.search_dsl  -> SearchSpaceAtom | None (None = AntBO default)
# decision.bias_dsl    -> BiasAtom | None      (None = zero bias)
# decision.fallback_used -> True if all retries failed
```

## LLM output format

The LLM is expected to return a JSON object with two OPTIONAL keys:

```json
{
  "update_trust_region": "HammingDistanceTo('ARYYGSYWYFD', 2) & Match('AR**GS*W*FD')",
  "update_bias": "MaxCysteine(1) + NetChargeRange(-1.0, 1.0)"
}
```

- Missing key = keep current value.
- Empty `{}` = no update (pass).
- `update_trust_region` source must evaluate to a `SearchSpaceAtom`.
- `update_bias` source must evaluate to a `BiasAtom`.

## DSL grammar

### Trust Region atoms

| Atom | Python | Notes |
|---|---|---|
| Hamming ball | `HammingDistanceTo(center_str, max_distance_int)` | exact Hamming distance bound |
| Position match | `Match(pattern_str)` | length-11; `A-Z` exact, `*` wildcard |
| AND | `Atom & Atom` | intersection (driver-based iter) |
| OR | `Atom \| Atom` | union (dedup iter) |
| NOT | `~Atom` | complement (reject sampling iter) |

### Bias atoms

| Atom | Python | Notes |
|---|---|---|
| Max Cys | `MaxCysteine(value_int)` | penalty if Cys count > value |
| Max hydrophobic run | `MaxHydrophobicRun(value_int)` | penalty if longest run > value |
| Max aromatic | `MaxAromatic(value_int)` | penalty + bonus |
| Net charge range | `NetChargeRange(min_v, max_v)` | penalty if outside range |
| No N-glycosylation | `NoNGlycosylation()` | penalty if N-X-[ST] motif present |
| Sum | `Atom + Atom` | additive composition |

## Trust region → set semantics

A DSL is a **predicate**, not a set. The set is `{x : atom.__contains__(x)}`.

- `__contains__(seq)` — O(seq_len) membership test (11 atomic compares)
- `__iter__()` — lazy enumeration with caveats:
  - `HammingDistanceTo`, `Match`, `And`, `Or`: finite iteration
  - `Not`: infinite by rejection sampling (use `iter_with_cap`)
  - `And`: driver algorithm (smallest child → filter through others)

When `|TR|` exceeds the configured cap (default 1M), `safe_exec_dsl` → validator
raises `TRTooLarge`. The error message is sent back to the LLM so it can
revise its DSL.

## Parameters (all in `bo/config.yaml: ldm:` section, see DSLConfig)

| Param | Default | Effect |
|---|---|---|
| `llm_init_enabled` | `True` | LLM-guided init (TR + bias sampling) |
| `llm_loop_enabled` | `True` | LLM orchestrator during BO iterations |
| `llm_temperature` | `0.25` | LLM sampling temperature |
| `llm_call_max_retries` | `3` | retries on parse/validation failure |
| `llm_call_timeout_s` | `30` | per-call timeout |
| `llm_decisions_log` | `outputs/llm_decisions/exp.json` | append-only log |
| `history_max_in_prompt` | `100` | cap on history rows in prompt |
| `bias_weight` | `0.1` | weight of bias contribution to acquisition |
| `max_dsl_size` | `1_000_000` | |TR| hard cap |
| `max_nesting_depth` | `8` | max AND/OR/NOT depth |
| `sample_max_attempts` | `10000` | reject sampling cap (per call) |
| `sample_timeout_s` | `5.0` | reject sampling timeout |
| `iter_cap_timeout_s` | `10.0` | iter_with_cap timeout |
| `atoms_whitelist` | (10 atoms) | safe_exec_dsl whitelist |
| `fallback_strategy` | `"original_antbo"` | fallback after all retries fail |

## Tests

```bash
pytest tests/bo/ldm/             # 155 unit tests
pytest tests/integration/        # 5 end-to-end tests
python scripts/smoke/run_ldm_smoke.py
python scripts/smoke/run_bo_smoke.py
```

## Architectural rule

`bo/` external code (e.g. `bo/main.py`, `bo/localbo_cat.py`) MUST use only
the public API. Forbidden imports are detected by
`tests/bo/ldm/test_public_api.py::test_bo_outside_does_not_import_internal_modules`.