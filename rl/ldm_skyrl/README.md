# `ldm_skyrl` — a second RL line on SkyRL

A SkyRL implementation of the small-molecule acquisition RL loop, running beside
the existing slime line rather than replacing it. Both lines are judged by the
same number: Pareto hypervolume at budget 80 on G12D and G12C, SFT+RL against
SFT-only, through the same evaluation stack.

The roadmap, the evidence behind it, and the reading of every figure are in
[`skyrl_port_plan.ipynb`](skyrl_port_plan.ipynb). The machine-readable phase
register is [`phases.json`](phases.json); the frozen numbers the notebook reads
are [`facts.json`](facts.json).

Modelled on [ApexAgents-SkyRL-Recipe](https://github.com/Mercor-Intelligence/ApexAgents-SkyRL-Recipe),
which is import-only against released SkyRL with no forks. This package keeps
that discipline: SkyRL is a dependency, never a vendored tree.

## Where this sits relative to the other branches

`rl/ldm_rl` — the framework-neutral environment — lives on the `rl` line, whose
published tip is `pr/reward-zero-fixes`. This package is on `main`, so the
roadmap is visible on the default branch. That means `ldm_skyrl` must not import
`ldm_rl` at module load time, and it does not: `_deps.require_ldm_rl` defers the
import and, when it fails, says which branch carries the package rather than
only that a module was not found.

To run anything that touches the environment, put a checkout of the `rl` line's
`rl/` directory on `PYTHONPATH`.

## The port map

| slime seam | SkyRL seam | this package |
|---|---|---|
| `ldm_rl/bridge.py::generate` (custom rollout fn) | `GeneratorInterface.generate` | `generator.py::LDMAcquisitionGenerator` |
| `ldm_rl/bridge.py::reward_func` (custom RM path) | reward returned inside `GeneratorOutput` | `generator.py::EpisodeTrace.reward` |
| `--prompt-data <jsonl> --input-key prompt` | prompt dataset object | `dataset.py::EpisodeDataset` |
| `SGLANG_ARGS` / `GRPO_ARGS` shell arrays | `SkyRLTrainConfig` overrides | `config.py::ALGORITHM_DEFAULTS` |
| ad-hoc `vina_max_workers` | `generator.rate_limit.*` | `reward.py::RewardConcurrency` |
| `--export=ALL` inheritance | Ray job runtime env | `ray_env.py::FORWARDED_ENV_VARS` |
| `train.py` argv assembly | entrypoint module | `entrypoints/main_ldm_*.py` |

`ldm_rl/env.py` is not in this table because it does not change. It is 803 lines
and names a framework on exactly one of them; the coupling lives almost entirely
in `bridge.py`.

## Why token-in-token-out

SkyRL's own note on `InferenceEngineOutput` is that
`decode(response_ids) == responses` holds but the reverse does not: several token
sequences render the same text. A multi-turn loop that re-tokenises the
transcript each turn therefore drifts from what the engine sampled, and the loss
mask stops describing the tokens the policy produced — with no error anywhere.
`LDMAcquisitionGenerator` appends the engine's own ids and tokenises only
environment feedback, which is masked out regardless.

`tests/test_generator.py::test_engine_sees_the_growing_transcript_not_a_fresh_prompt`
is the test that would catch a regression here. It is worth knowing that this
failure is otherwise invisible: a loop that re-prompts from scratch still
produces plausible tokens, a plausible reward, and a well-formed mask.

## Failing loudly

Every path that depends on a phase item that has not passed raises
`PhaseGateError`, and the message is built from `phases.json` — so the criterion
in the traceback is the same string as the criterion in the roadmap, and the two
cannot drift apart.

```
$ python -m ldm_skyrl.entrypoints.main_ldm_train
PhaseGateError: phase item A1 (A) has not passed: uv resolve of the SkyRL 0.3.0
  pin set on aarch64
  criterion: uv lock succeeds and the resolved torch is a +cu128 or +cu129 build
             (never +cu130, which makes torch.cuda.is_available() return False on
             driver 565)
  artifact:  rl/ldm_skyrl/uv.lock plus the resolver log
  budget:    CPU only
```

`runner.py` is gated on A5 rather than written speculatively. Wiring an
unexercised trainer would mean the first genuine failure surfaces inside Ray with
nothing to say which assumption broke.

## Running the tests

No GPU, no network, no SkyRL:

```bash
PYTHONPATH=rl python -m pytest rl/ldm_skyrl/tests -q
```

The whole of phase B is testable this way, which is why it does not wait on
phase A.

## The cluster pins, and why they are not preferences

| | value | why |
|---|---|---|
| driver / CUDA | 565 / 12.7 | what this cluster runs |
| SkyRL | 0.3.0, commit `b8a5caaa` | `main` requires CUDA 13.0 and driver r580 |
| wheels | cu128 / cu129 | a cu130 build installs and then reports `torch.cuda.is_available() == False` |
| vLLM | `0.23.0+cu129` | SkyRL 0.3.0's inference backend is vLLM only; the aarch64 wheel exists on the pinned index |

## Phase A first, and A5 before anything with LDM in it

`scripts/run_1gpu_smoke.sh` runs a generator that returns a constant reward and
contains no LDM code. A failure there is a failure of the stack and cannot be
confused with a defect in the wiring phase B builds.
