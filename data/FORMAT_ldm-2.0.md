# LDM Unified Data Format `ldm-2.0`

`ldm-2.0` is a unified SFT data format for Large Discovery Model style tasks.
It is intended to cover nanogpt training-program search, small-molecule design,
protein/antibody sequence design, and future black-box optimization or inverse
design tasks with the same structure. The model should learn not only how to
propose candidates inside a given design space, but also how to expand the
design space itself.

## 0. Design Principles

1. **Language first.** Structured fields provide the skeleton; semantics live in
   natural language. Every layer has `description`, and the action layer has
   `reasoning`, so each round can preserve the full decision rationale. The
   trained artifact is an LLM, not a JSON slot filler.
2. **The design space is state, not a header.** `design_space` lives inside
   `search_state` and can be changed by actions. This is the key difference
   from simpler static-table formats.
3. **Actions are a tagged union.** Candidate proposal and design-space expansion
   are first-class sibling actions; expansion is not a special case of proposal.
4. **Names should explain themselves.** Avoid paper notation such as `C_t` or
   `arm_set`; field names should say what they mean.
5. **Missing data is explicit.** Use `null` plus explicit flags such as
   `reasoning_available:false`. Do not fabricate placeholders or ask an LLM to
   backfill missing rationale.

## 1. Top-Level Structure

```jsonc
{
  "schema_version": "ldm-2.0",
  "task":         { ... },   // what the task is; static
  "search_state": { ... },   // current dynamic search state, including design space
  "request":      { ... },   // what the model is asked to do this round
  "action":       { ... }    // training target: what the model should output
}
```

`task` + `search_state` + `request` renders to the training **instruction**.
`action` renders to the training **output**.

## 2. `task`: Static Task Definition

```jsonc
{
  "id": "nanogpt",
  "domain": "training_program",
  "description": "<natural language: task, evaluation, and what good means>",
  "objectives": [
    {
      "name": "val_bpb",
      "direction": "minimize",
      "description": "validation bits per byte measured under a fixed budget"
    }
  ],
  "reasoning_available": true
}
```

| Field | Meaning |
|---|---|
| `objectives` | An array, so multi-objective tasks are native. Single-objective tasks use one item; smallmol uses two: minimize Vina and maximize activity. |
| `direction` | `minimize` or `maximize`, replacing prose such as "lower is better". |
| `reasoning_available` | Protein traces have no rationale, so they use `false` to make the missing rationale explicit. |

## 3. `search_state`: Dynamic Search State

```jsonc
{
  "round": 41,
  "num_evaluated": 80,
  "design_space":      { ... },
  "observations":      [ ... ],
  "best_so_far":       { ... },
  "surrogate_feedback": { ... },
  "progress":          { ... },
  "do_not_repeat":     [ ... ]
}
```

### 3.1 `design_space`: Expandable Design Space

The design space is the core of the format. It is state that actions can change.

```jsonc
{
  "representation": "parameter_edits",
  "active_parameters": [
    {"name":"HEAD_DIM","type":"choice","domain":[64,96,128],"edit_op":"set_choice"},
    {"name":"EMBEDDING_LR","type":"float","domain":[0.1,1.0],"scale":"log","edit_op":"set_numeric"}
  ],
  "inactive_parameters": [
    {"name":"WARMDOWN_RATIO","type":"float","domain":[0.0,0.95],"current_value":0.5}
  ],
  "expansion_history": [
    {"round":12,"activated":"WARMDOWN_RATIO","reason":"the schedule family has been exhausted and needs a new coordinate"}
  ],
  "allows_new_parameters": true,
  "description": "The surrogate observes only active parameters; activating an inactive parameter expands the feature vector for later candidates."
}
```

| Field | Meaning |
|---|---|
| `representation` | `parameter_edits` means candidates are incremental edits to a parent state, as in nanogpt. `complete_design` means candidates are full objects, as in smallmol and protein. |
| `active_parameters` | Dimensions currently visible to the surrogate and editable by the model. |
| `inactive_parameters` | Known dimensions that are not yet active and can be activated by `expand_design_space`. This is L1 expansion. |
| `expansion_history` | Prior expansions, so the model can see when and why the space changed. |
| `allows_new_parameters` | Whether the model may invent a new primitive not present in the schema. This is L2 expansion. |

Three expansion levels:

| Level | Action | Meaning |
|---|---|---|
| L0 | `propose` | Propose candidates inside the current space. |
| L1 | `expand_design_space` | Activate a known inactive dimension. |
| L2 | `add_new_parameter` | Invent a new primitive that does not exist in the schema, such as sparse value memory. |

### 3.2 `observations`: Historical Observations

```jsonc
[
  {
    "design": {...},
    "results": {"val_bpb": 0.985983},
    "round": 8,
    "roles": ["best_so_far","best_path"],
    "description": "Current best produced by changing batch 524288 to 262144 and head_dim 128 to 96."
  }
]
```

`roles` is an array of tags that unifies task-specific history views:

| Original concept | Normalized `roles` |
|---|---|
| smallmol `pareto_front` | `["pareto_front"]` |
| smallmol `top_low_vina` / `top_high_activity` | `["top_objective_0"]` / `["top_objective_1"]` |
| smallmol `balanced_elites` / `recent_selected` | `["elite"]` / `["recent"]` |
| protein `best[]` / `recent[]` | `["best"]` / `["recent"]` |
| nanogpt best path | `["best_so_far","best_path"]` |

One observation can carry multiple roles. This replaces the original smallmol
pattern of five overlapping history lists, which is redundant and hard to extend.

### 3.3 `surrogate_feedback`

This field may be `null`.

```jsonc
{
  "predicted_mean": 0.9871,
  "uncertainty": 0.0042,
  "acquisition_value": 0.31,
  "description": "Surrogate prediction and uncertainty for this branch."
}
```

Nanogpt has this feedback. Smallmol and protein use `null`.

### 3.4 `progress`: Stall Signal

```jsonc
{
  "stalled": true,
  "rounds_since_improvement": 12,
  "description": "No improvement for 12 rounds; the current active feature set may be too narrow."
}
```

Stall information is an important trigger for design-space expansion. It teaches
the model when to stop searching inside the current space.

## 4. `request`: Current Round Request

```jsonc
{
  "allowed_actions": ["propose","expand_design_space"],
  "num_candidates": 1,
  "max_edits_per_candidate": 2,
  "description": "Choose one: edit inside the current feature space, or activate one inactive parameter to expand the space."
}
```

`allowed_actions` explicitly tells the model which actions are legal. This is
necessary if the model is expected to learn expansion behavior.

## 5. `action`: Training Target

```jsonc
{
  "type": "propose",
  "reasoning": "<natural language: why this action was chosen, what was rejected, and which history supports it>",
  "payload": { ... },
  "summary": "<one-sentence summary>"
}
```

### 5.1 `propose` + `parameter_edits` for nanogpt

```jsonc
{
  "candidates": [{
    "parent": "state_0041",
    "edits": [{
      "parameter": "HEAD_DIM",
      "edit_op": "set_choice",
      "value": 96,
      "rationale": "Reduce head dimension to trade width for more steps under the fixed budget."
    }]
  }]
}
```

### 5.2 `propose` + `complete_design` for smallmol/protein

```jsonc
{
  "candidates": [
    {"design":"CCNC","rationale":"methylate CCN; explore branching"},
    {"design":"PQWNYVQPGCE","rationale":null}
  ]
}
```

### 5.3 `expand_design_space` (L1)

```jsonc
{"activate":"WARMDOWN_RATIO","initial_value":0.5}
```

### 5.4 `add_new_parameter` (L2)

```jsonc
{
  "parameter": {"name":"SPARSE_VALUE_MEMORY","type":"choice","domain":[0,1024,4096]},
  "code_sketch": "...",
  "why_new_axis": "Dense capacity has saturated under the fixed budget, so a cheaper capacity axis is needed."
}
```

## 6. Three-Task Mapping Table

| | nanogpt | smallmol | protein |
|---|---|---|---|
| `domain` | training_program | molecule | antibody_sequence |
| `objectives` | 1, minimize val_bpb | 2, minimize Vina and maximize activity | 1, minimize binding energy |
| `representation` | parameter_edits | complete_design | complete_design |
| `active_parameters` | 11-dimensional schema | empty; hygiene rules go in description | length=11 plus 20AA alphabet |
| `inactive_parameters` | 3 dimensions | none | none |
| `allows_new_parameters` | true | true, new scaffold | false |
| `allowed_actions` | propose, expand_design_space | propose | propose |
| `surrogate_feedback` | present | null | null |
| `do_not_repeat` | none | avoid_exact_smiles | do_not_repeat |
| `reasoning_available` | true | true | false |
| `num_candidates` | 1 | 8 | 1 |

## 7. Known Data Issues

| Issue | Location | Handling |
|---|---|---|
| Stale/leaking `required_output` | protein instruction | Five records always contain `['ADGHTKQNPRA']`: record 0 leaks the answer, and the other four contain wrong instructions. Drop this field during conversion. |
| `expand_design_space` is extremely rare | nanogpt | Full run has only 9 / 1596 = 0.56 percent. SFT will likely treat it as noise, so the model will not learn expansion without dedicated samples or oversampling. |
| Degenerate action distribution | nanogpt | HEAD_DIM and WINDOW_PATTERN account for 79-81 percent of edits; 75 percent of two-operator proposals use the same pair. |
| protein has no rationale | protein output | Use `null` plus `reasoning_available:false`; do not ask an LLM to backfill rationale. |
| duplicate candidates | smallmol | Preserve faithfully by default. Use `--dedup-candidates` only when explicit deduplication is wanted. |

## 8. Rendering for LlamaFactory

`ldm-2.0` is an intermediate representation, not the direct training format.
The renderer converts it to Alpaca:

```text
task + search_state + request  --render-->  instruction with natural language and embedded JSON
action                         --render-->  output JSON
```

Rendering modes:

- `--render prose` (default): structured fields become natural-language sections
  plus embedded JSON. This is recommended for LLM training.
- `--render json`: serialize the entire `ldm-2.0` record. This is useful for
  debugging or highly structured experiments.

## 9. Extending to Similar Tasks

| New task | representation | Meaning of expansion |
|---|---|---|
| material formulation design | complete_design | `add_new_parameter` introduces a new precursor |
| prompt optimization | parameter_edits | `expand_design_space` activates a new prompt component |
| directed protein evolution | parameter_edits | `add_new_parameter` opens a new mutation site |
| catalyst screening | complete_design | `expand_design_space` activates a new reaction-condition dimension |

