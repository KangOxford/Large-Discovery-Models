# LDM RL environment (Slime)

This package turns an LDM task campaign into a reinforcement-learning
environment for [Slime](https://github.com/THUDM/slime) (vendored as the
`rl/slime` git submodule). The environment itself is framework-agnostic; only
`bridge.py` talks to Slime, and it does so lazily inside the rollout worker.

## Why this is easy

LDM tasks are already abstracted behind a small, stable seam:

- `LDMTaskSpec` declares objectives, the candidate domain, the proposal
  response space (including a declared parser), and the reservoir contract;
- `CandidateDomainAdapter.admit` validates/normalizes proposals;
- `CandidateEvaluator.evaluate` measures a candidate;
- `ObjectiveSet` compares metrics.

One campaign round is exactly one RL transition:

| LDM campaign concept        | RL concept                          |
| --------------------------- | ----------------------------------- |
| one campaign (`LDMEngine`)  | one episode                         |
| one expansion round         | one `step`                          |
| policy text → candidates    | action                              |
| rendered history feedback   | observation                         |
| objective improvement       | reward                              |
| iteration budget            | truncation (episode end)            |
| repeated empty reservoirs   | termination (task concluded)        |
| candidate admission + dedup | environment dynamics (shared logic) |

`LDMEnv.step` reuses the shared `ReservoirBuilder`, the task's own
`CandidateDomainAdapter` and `CandidateEvaluator`, and mirrors the engine's
evaluation error wrapping and reservoir-order selection, so environment
semantics match campaign semantics.

## Layout

```text
rl/ldm_rl/
├── env.py             # LDMEnv: reset/step/run, EnvConfig, EnvStep, EpisodeResult
├── parsing.py         # loads task-declared response-space parsers
├── prompts.py         # policy-facing observation rendering
├── episodes.py        # EpisodeSpec JSON + Slime prompt-data generator CLI
├── factories.py       # build_env(task_id, mode="mock") per-task adapter wiring
├── bridge.py          # Slime custom generate() / reward_func()
├── run_ldm_rl.sh      # example Slime GRPO launch script
└── tests/             # env, factory, and bridge (fake-backend) tests
```

## The environment contract

```python
from ldm_rl import EnvConfig, LDMEnv
from ldm_rl.factories import build_env

env = build_env(
    "ai4bio_mutation_effect_prediction",
    mode="mock",
    config=EnvConfig(iterations=4, reservoir_size=2, reward="improvement"),
)
observation = env.reset()          # str: task, objectives, schema, instructions
step = env.step(action_text)       # str -> EnvStep
# step.observation  feedback transcript (loss_mask=0 in Slime)
# step.reward       objective improvement over the previous incumbent
# step.done         terminated (empty-reservoir limit) or truncated (budget)
# step.info         structured JSON: rejections, evaluations, incumbent, ...
```

Reward policies (`EnvConfig.reward`):

- `improvement` (default): increase of the oriented objective over the
  previous best, clipped at 0. For multi-objective tasks each objective is
  tracked component-wise against its own best-so-far and the improvements are
  summed.

  **Pass `reward_ref_point` whenever the objective is expensive or minimised.**
  It is a floor under the incumbent on every round, not just a round-1 seed.
  Without it the per-round gains telescope to `(first measured - best
  measured)`, so a policy is paid for opening with a deliberately bad candidate
  and then "recovering"; with it the episode total is
  `max(0, reference - best measured)`, which is path independent. Same defect,
  same fix, as the moving nadir removed from `hypervolume`. The reference is in
  *oriented* (maximisation) space, so a minimised metric needs its negation:
  `reward_ref_point=(-reference_val,)`.
- `raw`: oriented objective value of the best newly evaluated candidate.
- `binary`: 1.0 only when the new candidate strictly improves the incumbent.
- `acquisition`: the GP acquisition score (mean + beta * std for GP-UCB) of
  the evaluated candidate(s), taken from the selector run *before*
  evaluation — i.e. the decision-time expected utility of the proposal.
  Requires a `selector` + `surrogate_encoder` pair, which the task factories
  wire automatically (`RBFGPUCBSelector` + the task's own encoder, exactly as
  the campaign does).

Parse errors and fully-rejected rounds get `reward_invalid`; evaluations that
ran but failed get `reward_failure` (both default to 0.0).

When a selector is configured, candidate selection inside each round uses the
same acquisition logic as the engine (`fit` on history, encode candidates,
highest acquisition first). Without one, selection is reservoir-order. Every
step exposes the full selection record (including per-candidate acquisition
scores) under `EnvStep.info["selection"]`.

## Tests

```bash
# from the repository root, using the project venv
.venv/bin/python -m pytest rl/ldm_rl/tests -q
```

- `test_env.py`: environment semantics on synthetic adapters, including a
  parity check against a real `LDMEngine` run.
- `test_factories.py`: builds mock environments from the registered
  `ai4bio_mutation_effect_prediction`, `causal_discovery_discrete`, and
  `small_molecule` tasks and runs short episodes against their real
  admission/evaluation code.
- `test_bridge.py`: drives `bridge.generate` with a fake Slime backend and a
  fake tokenizer (no Slime/GPU required) and checks the transcript, loss mask,
  and episode reward.

## RL case example

`examples/small_molecule_rl_case.py` runs a full mock RL episode over the
small-molecule task using its own deterministic mock proposer
(`ExpandingMockCase2LLM`) and mock scorers (vina + KRAS activity), no
torch/GP/docking required:

```bash
python rl/ldm_rl/examples/small_molecule_rl_case.py 6
```

Each round prints the proposed SMILES, its two scores, and the improvement
reward; the final line is a JSON summary (rounds, total reward, best scores).

## Running Slime training

See `run_ldm_rl.sh` (needs a GPU cluster). In short:

```bash
# 1. initialize the submodule (done in this checkout)
git submodule update --init rl/slime

# 2. generate episode prompt data
python rl/ldm_rl/episodes.py --output rl_episodes.jsonl \
    --task ai4bio_mutation_effect_prediction --mode mock \
    --count 64 --iterations 8 --reservoir-size 2

# 3. launch training with (among others):
#    --custom-generate-function-path ldm_rl.bridge.generate
#    --custom-rm-path ldm_rl.bridge.reward_func
#    --prompt-data rl_episodes.jsonl --input-key prompt --label-key label
#    and PYTHONPATH=<repo>/rl:<repo>
```

Each prompt row is one `EpisodeSpec` JSON; `bridge.generate` builds the
environment from it, runs the multi-turn propose/evaluate loop against sglang,
and fills `sample.reward` directly (Slime skips its own reward model when the
custom generate function already set it).

## Current scope and limits

- Mock-mode factories are tested end to end; real-mode factories are wired but
  require upstream benchmark paths (`upstream_root`, `data_dir`, `cv_dir` /
  `upstream_root`) and have not been exercised here.
- The two wired tasks configure the campaign's own `RBFGPUCBSelector` +
  surrogate encoder, so selection and the `acquisition` reward use the same
  GP acquisition as the LDM campaign.
- The env step is synchronous, but `bridge.generate` awaits it via
  `asyncio.to_thread`, so a slow evaluation no longer stalls the rollout
  worker's event loop and sibling episodes keep their evaluations in flight.
  This matters once a step costs minutes: `nanogpt` trains a real model for its
  full wall-clock budget per step. A true actor-based backend would still be
  better for cross-process scheduling.
- `nanogpt` is wired in both modes, and is the reference for an **expensive,
  measured** objective: one step is one real training run scored by
  `val_bpb`, with no analytic stand-in. See
  `rl/slime_launch/NANOGPT_HANDOFF.md` -- in particular that the trainer and
  the evaluator must hold disjoint GPUs, and that the reward needs a measured
  `reward_ref_point`.
- `antibody` still resolves to a placeholder adapter and fails fast.
