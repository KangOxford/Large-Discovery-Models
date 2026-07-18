# HANDOFF: Unified LDM Data Processing and LlamaFactory SFT

> **For the next AI agent:** This file is your task brief. The goal is to
> convert heterogeneous Large Discovery Model (LDM) task trajectories into the
> `ldm-2.0` intermediate representation, render them into LlamaFactory Alpaca
> format, and run SFT.
>
> Read `FORMAT_ldm-2.0.md` for the schema. The converter is
> `scripts/build_ldm2.py`; it has been validated on real data, so do not rewrite
> it from scratch.
>
> Read section 5 carefully. Those pitfalls can destroy training quality.

## 1. Background

LDM means an LLM proposes candidates, a Gaussian-process surrogate computes an
acquisition score, and Bayesian optimization chooses which candidates to
evaluate. We want to distill the large teacher model's proposal behavior into a
smaller proposer model.

The three tasks are the same learning problem at the interface level: given
observed history, generate the next exploration candidates, and do not predict
objective values. Only the domain payload changes.

| Task | Domain | Candidate shape | Objective |
|---|---|---|---|
| `nanogpt` | training-program search | edits to `train.py` parameters | minimize val_bpb, single objective |
| `smallmol` | molecular design | SMILES strings | minimize Vina and maximize activity, multi-objective |
| `protein` | antibody CDRH3 design | 11-amino-acid strings | minimize binding energy, single objective |

## 2. Deliverables

```text
.
|-- README_HANDOFF.md          # this file
|-- FORMAT_ldm-2.0.md          # format spec; read first
|-- scripts/
|   |-- build_ldm2.py          # converter, Python 3.8+, no third-party deps
|   |-- fabricate.py           # rule-based augmentation with provenance
|   `-- verify.py              # independent verifier; 13 checks
|-- examples/
|   |-- ir_sample.jsonl        # 15 ldm-2.0 IR records, 3 tasks x 5
|   |-- sft_sample.jsonl       # 15 rendered Alpaca records
|   `-- dataset_info.json      # LlamaFactory dataset registry example
`-- train/
    `-- ldm_lora_sft.yaml      # sample LlamaFactory LoRA SFT config
```

## 3. Core Design

One sentence: the design space is state, not a header. The model must learn not
only how to propose inside the current space, but also how to expand the space.

Nanogpt's original prompt states the key contract:

> "Choose exactly one action: either stick with the current active feature space
> or expand it."

Therefore `action` is a tagged union with three expansion levels:

| Level | `action.type` | Meaning |
|---|---|---|
| L0 | `propose` | Propose candidates in the current design space. |
| L1 | `expand_design_space` | Activate a known inactive parameter. |
| L2 | `add_new_parameter` | Invent a new primitive absent from the schema. The schema supports this, but current data does not contain examples. |

Pipeline:

```text
raw trace --[adapter]--> ldm-2.0 IR --[renderer]--> LlamaFactory Alpaca
```

The IR is not fed directly to training. This keeps task-specific adapters
separate from shared rendering, auditing, and verification logic.

## 4. Tasks

### Task A: Convert All Data

There are two source categories. The user should provide paths.

**A1. Sample file**

Shape: `ldm_data_sample.json`, with top-level keys
`{"smallmol":[...], "nanogpt":[...], "protein":[...]}`. Each item has
`{instruction,input,output,system}`.

```bash
python scripts/build_ldm2.py from-sample \
  --in <path>/ldm_data_sample.json \
  --out-ir ir_sample.jsonl
```

**A2. Full nanogpt run directory**

Shape: `expanded_ldm_bon_N4H4_03/`, containing
`states/state_XXXX/{prompt.md,response.md,operations.json,meta.json}` plus
`manifest.jsonl`.

```bash
# High-quality set: candidates that really ran and have real val_bpb.
python scripts/build_ldm2.py from-nanogpt-run \
  --run-dir <path> \
  --out-ir ir_ng_eval.jsonl \
  --min-status evaluated

# Larger set: includes generated candidates with GP surrogate scores.
python scripts/build_ldm2.py from-nanogpt-run \
  --run-dir <path> \
  --out-ir ir_ng_gen.jsonl \
  --min-status generated
```

`--min-status` trust ranking:

```text
generation_error < seed < generated < surrogate_scored < crash < evaluated
```

If the user has multiple run directories, such as `_01`, `_02`, and `_03`,
process and merge all of them. This is the best first defense against the
distribution issues in section 5.2.

### Task B: Audit

Always run the audit.

```bash
python scripts/build_ldm2.py audit --in-ir ir_ng_gen.jsonl
```

It prints action distribution, parameter-edit concentration, and warnings about
degenerate behavior. Report the audit result to the user.

Reference baseline for one run:

```text
by action: propose 1526 (99.48%) | expand_design_space 8 (0.52%)
edited parameters: WINDOW_PATTERN 39.4% | HEAD_DIM 39.3% -> top-2 concentration: 78.6%
[!] space-expanding actions are 0.52% of data - SFT will likely drop this behaviour.
```

### Task B2: Optional Augmentation

Use this to address the distribution issues in section 5.2.

```bash
python scripts/fabricate.py \
  --in-ir ir_ng_gen.jsonl \
  --out-ir ir_ng_aug.jsonl \
  --f1 150 --f2 72 --f3 80 --f4 300 --f5 40 --seed 0
```

The five operators derive synthetic examples from real data plus explicit BO
rules. Every synthetic record carries provenance.

| Operator | Creates | Teaches |
|---|---|---|
| F1 exhaustion | Evidence that dominant knobs have been exhausted and scores are flat. | Stall plus exhaustion should trigger design-space expansion. |
| F2 plateau | Real context with a rule-relabeled action. | Do not repeat ineffective edits. |
| F3 rotation | Masks dominant knobs and transplants real rare-parameter actions. | Which alternatives to use when dominant knobs are unavailable. |
| F4 jitter | Resamples numeric parameters inside their domains. | Numeric domains are continuous. |
| F5 transplant | Moves real expansion actions to other stalled contexts. | Link expansion to stall signals, not one specific context. |

Reference effect on one run:

```text
expand_design_space: 0.52% -> 9.6%
top-2 concentration: 78.6% -> 64.4%
synthetic share: 29.5%
```

Built-in augmentation checks:

1. F1 flatness claims must match the synthetic numbers.
2. F1 retained prose must not contradict synthetic history.
3. F3 frozen knobs must not still appear in prose as editable.
4. No structural shortcut: a field's mere presence must not perfectly predict an action.
5. No self-contradictory constraints: the target action must not violate its own `do_not_repeat`.
6. Stall pathology: if `progress.stalled=True`, `propose` should not dominate above 85 percent.
7. Action validity: parameters active, values in domain, edit budget respected, activate target inactive.
8. No no-op edits: proposed value must differ from the current value.

Red lines:

- Do not use `--f2 all`. It creates too many synthetic rows, pushes synthetic
  share above half the dataset, and creates a new `WARMDOWN_RATIO` bias.
- Do not flatten top-2 concentration to uniform. The dominant knobs really
  caused most improvements; the problem is repeating them after a plateau.
- Validate only on real held-out data. Synthetic rows are training-only.
- If used in a paper, disclose operators and mixture weights.
- Some duplicate `(context, action)` synthetic pairs are acceptable, but do not
  increase F1 further without checking loss overweighting.

### Task B3: Verify

Strongly recommended:

```bash
python scripts/verify.py all \
  --run-dir <RUN> \
  --in-ir ir_ng_aug.jsonl \
  --sft data/ldm_sft.jsonl \
  --dataset-info data/dataset_info.json \
  --cutoff-len 16384
```

The verifier has four groups and 13 checks. Each exists because it caught a real
bug.

| Group | Checks | Example bug caught |
|---|---|---|
| `coverage` | Source prompt sections reach rendered output. | A 26KB `train.py` silently disappeared; current values were parsed but not rendered. |
| `validity` | Every action can be executed by the runner. | Training targets came from rejected attempts with out-of-range values or too many edits. |
| `alpaca` | LlamaFactory can load the file. | Bad field mapping, invalid JSON, output contract outside cutoff. |
| `leakage` | No answers, provenance, or structural shortcuts leak into prompts. | `## Observed history` appeared only on synthetic records, making it a shortcut. |

Failure returns exit code 1. Subcommands can also be run separately:

```text
verify.py coverage | validity | alpaca | leakage
```

### Task C: Render to Alpaca

```bash
# Merge all IR.
cat ir_sample.jsonl ir_ng_eval.jsonl > ir_all.jsonl

python scripts/build_ldm2.py render \
  --in-ir ir_all.jsonl \
  --out data/ldm_sft.jsonl \
  --render prose \
  --dataset-info data/dataset_info.json
```

Render modes:

- `--render prose` is the default and recommended mode. Structured fields become
  natural-language sections plus embedded JSON.
- `--render json` serializes the whole IR and is mainly for debugging.
- `--strip-parent-artifact` drops the embedded parent `train.py`; this saves many
  tokens but breaks train/inference prompt parity, so leave it off unless memory
  requires it.

### Task D: Connect to LlamaFactory

1. Put `data/ldm_sft.jsonl` and `data/dataset_info.json` in LlamaFactory's
   `data/` directory, or merge the entry into an existing `dataset_info.json`.
2. Use `train/ldm_lora_sft.yaml` as a starting config.
3. Check `cutoff_len` before training. See section 5.3.

## 5. Known Issues

### 5.1 Stale or Leaking `required_output` in Protein

Protein instructions contain a `required_output` field. In the source data, five
records always contain `['ADGHTKQNPRA']`: the first leaks the answer and the
other four are wrong instructions.

`adapt_protein()` drops this field. Do not add it back.

### 5.2 Degenerate Action Distribution in Nanogpt

One run shows:

- `expand_design_space`: 8 / 1534 = 0.52 percent. SFT will likely treat this as
  noise, so the small model will not learn expansion.
- `HEAD_DIM` plus `WINDOW_PATTERN`: 78.6 percent of all edits.
- 75 percent of two-operator proposals use the same pair.
- Of 80 real evaluation rounds, only 8 refresh the best score; the other 72 are
  plateau rounds repeatedly hitting the same knobs.

The likely cause is the teacher model's weak feedback sensitivity, a known
LLM-as-optimizer failure mode. Naive distillation will inherit that behavior.

Concentration is not automatically bad: the same two knobs drove most real
improvements. The bad behavior is repeating them after the search has stopped
improving. Do not force a uniform distribution over all parameters.

Mitigations:

| Option | Method | Cost |
|---|---|---|
| Repeat cap | Cap identical `(parameter,value)` suggestions at k, e.g. 3, while preserving improvement rounds. | Single-run ceiling: about 81% to 72%. |
| Merge multiple runs | Process all available run directories. | Best first option; naturally adds coverage and improvement rounds. |
| Oversample expansion actions | Duplicate the rare expansion rows. | Can overfit to the three specific inactive parameters. Use carefully. |
| Keep only improvement rounds | Filter by whether val_bpb improves best-so-far. | Single run leaves only about 8 rows. Too small by itself. |

Confirm the mitigation with the user before applying it.

### 5.3 Context Length

Nanogpt instructions embed the full parent `train.py` of about 26KB. Measured
rendered prompt sizes:

```text
median: 42519 characters
p95:    43433 characters
max:    43501 characters
estimated tokens: median about 13287, max about 13594
```

Use `cutoff_len >= 8192`; `16384` is recommended and has been measured to fit
the max example. Otherwise the trailing "Your move" contract can be truncated.

`--strip-parent-artifact` reduces prompt length by about 75 percent, but it
breaks training/inference consistency if inference still includes `train.py`.
Disclose this tradeoff if you use it.

### 5.4 Protein Has No Rationale

Protein outputs are bare sequence strings. Use `rationale: null` and
`task.reasoning_available: false`. Do not generate synthetic rationales with an
LLM; that trains the model to invent explanations.

### 5.5 Instruction/Contract Conflict

Original nanogpt prompts used the legacy tool contract
`propose_train_operations`, while `ldm-2.0` uses a unified JSON action. The
converter restates the contract in `ldm-2.0` terms and stores the legacy text in
`raw_context.legacy_return_format`.

Rendered instructions should not contain `propose_train_operations`.

### 5.6 System Field Consistency

The renderer adds a fixed `system` field per task. If the inference environment
does not supply a separate system message, training with `system` can create a
train/inference mismatch. Confirm the inference format with the user.

### 5.7 Inference Must Use the Same Renderer

`ldm-2.0` prompts are structurally different from the original task prompts.
They are reorganized as Task, Objectives, Design space, Observed history, and
Your move, with a unified JSON output contract.

After fine-tuning, the runner must construct `ldm-2.0` IR and call the same
renderer used during SFT. Do not train on `ldm-2.0` and infer with legacy prompts.

### 5.8 Nanogpt Has No `do_not_repeat`

Protein and smallmol have real hard constraints: `do_not_repeat` and
`avoid_exact_smiles`. Nanogpt does not. Do not synthesize this field for nanogpt.
Exhaustion evidence belongs in `observations` and `progress`.

### 5.9 `response.md` Can Contain Rejected Attempts

`response.md` is a transcript and may include rejected attempts:

```text
attempt 1 rejected: EMBEDDING_LR=0.05 outside [0.1, 1.0]
attempt 2 rejected: EMBEDDING_LR=0.075 outside [0.1, 1.0]
attempt 3 accepted
```

Taking the first tool call would train the model to emit invalid actions. The
converter uses the accepted `operations.json` set with type-sensitive matching.
If no transcript call matches, it trusts the runner's validated record.

### 5.10 Markdown Section Titles Must Match Exactly

`_sections()` splits prompts by exact headings. A one-character mismatch can
silently merge a section into the previous one. This has previously lost the
26KB parent `train.py`, and it also once dropped current active values.

The converter now attaches `current_value` to active parameters and renders it
as `current=...`. It also captures prior operations as
`design_space.applied_this_transition`.

When adding a new task, compare source and rendered information volume.

## 6. LlamaFactory Alpaca Format

Each line is one record:

```json
{
  "system": "You are a scientific search agent ...",
  "instruction": "# Task: nanogpt ...\n## Design space ...\n## Your move ...",
  "input": "",
  "output": "{\"type\":\"propose\",\"reasoning\":\"...\",\"payload\":{...},\"summary\":\"...\"}",
  "source": "nanogpt",
  "action_type": "propose"
}
```

`source` and `action_type` are extra fields for filtering and mixture control.
LlamaFactory ignores them.

`dataset_info.json`:

```json
{
  "ldm_bo_sft": {
    "file_name": "ldm_sft.jsonl",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "system": "system"
    }
  }
}
```

Use the registered dataset name `ldm_bo_sft`, not the file name, in training.

## 7. Acceptance Checklist

- [ ] All sources have been adapted; conversion failures are reported.
- [ ] Audit results have been reported to the user.
- [ ] Rendered instructions do not contain `propose_train_operations`.
- [ ] Rendered IR and output do not contain `required_output`.
- [ ] Every `output` is valid JSON with `type` and `payload`.
- [ ] `dataset_info.json` points to the actual file name.
- [ ] `cutoff_len` is at least 8192, preferably 16384 for full nanogpt prompts.
- [ ] The section 5.2 mitigation has been confirmed with the user.
- [ ] `verify.py all` returns exit code 0.
- [ ] Longest examples have been checked with the real tokenizer, not only the
  rough 3.2 chars/token estimate.

## 8. Open Questions for the User

1. How many run directories are available? With only one, section 5.2 cannot be
   fully fixed.
2. Which mitigation should be used for section 5.2? Recommended: merge multiple
   runs plus repeat cap k=3, without uniformizing the parameter distribution.
3. Does inference use a separate system message?
4. Should L2 `add_new_parameter` examples be collected from reflection logs?
5. Which base model and memory budget will be used? This determines `cutoff_len`
   and whether stripping the parent artifact is necessary.

## 9. Quick Smoke Check

```bash
python scripts/build_ldm2.py from-sample \
  --in <sample>.json \
  --out-ir /tmp/ir.jsonl

python scripts/build_ldm2.py render \
  --in-ir /tmp/ir.jsonl \
  --out /tmp/sft.jsonl \
  --render prose

python - <<'EOF'
import json
recs = [json.loads(line) for line in open('/tmp/sft.jsonl')]
assert all('propose_train_operations' not in r['instruction'] for r in recs)
assert all('required_output' not in r['instruction'] for r in recs)
assert all(json.loads(r['output']).get('type') for r in recs)
print('ok', len(recs))
EOF
```
