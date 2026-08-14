# Real LDM Campaign Examples

These plots are compact, versioned examples from real evaluator-backed runs.
They are evidence that the three adapters execute end to end and that their
optimization trajectories improve. They are not a controlled ablation study.

## Provenance

| Plot | Campaign | Snapshot | Result |
| --- | --- | --- | --- |
| `antibody_ucb_100.png` | Antibody `policy_max`, UCB, antigen `1ADQ_A`, seed 42 | Complete: 100 real Absolut evaluations, 20 initialization plus 80 UCB selections | Best binding energy improved from -88.56 after initialization to -96.72. |
| `small_molecule_ehvi_100.png` | Direct-LLM molecule generation, EHVI, seed 42 | Complete: 100 real Vina plus G12D activity evaluations | Pareto hypervolume reached 22.8080517046179. |
| `nanogpt_lcb_100.png` | Operation-tool best-of-N N4H4, LCB, seed 42 | Complete: 20 warm-up attempts and 100 LCB iterations; 99 outer candidates reached real training, yielding 116 finite observations overall | Best `val_bpb` improved from 0.986220 in finite warm-up to 0.981844. |

The nanoGPT launcher completed with return code 0. Iterations are contiguous
from 1 through 100, and all selected and best states are persisted. The x-axis
includes finite observations: three failed warm-up evaluations and the invalid
outer candidate selected at iteration 83 are recorded in the run artifacts but
excluded from the GP buffer and plot. Iteration 83 did not launch real training,
so the completed campaign contains 119 real training attempts rather than 120.

## Evidence Boundary

- Antibody uses the real Absolut evaluator.
- Small molecule uses real AutoDock Vina plus the configured G12D activity model.
- nanoGPT uses real 300-second GPU training evaluations.
- All three use an OpenAI-compatible served LLM for proposal generation.
- One seed per task is shown here. Without random, pure-LLM, BO-only, and
  multi-seed controls, the plots establish optimization progress but do not by
  themselves isolate the causal contribution of every LDM component.

Raw trajectories are intentionally not committed. The small-molecule round
trace alone is approximately 239 MB. Use `scripts/plot_campaigns.py` to render
plots from persisted run artifacts.
