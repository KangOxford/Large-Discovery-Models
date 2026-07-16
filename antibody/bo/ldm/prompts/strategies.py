"""Strategy registry — controls the strategy section of the system prompt.

Each strategy is a string that replaces the ``{strategy_section}``
placeholder in ``system.txt``.  The strategy tells the LLM how to
choose trust regions, bias, and review decisions.

Registered strategies:
    ldm-default : flexible LLM-driven BO (the original design)
    antbo-mock  : strict mimicry of AntBO's original local search
"""
from __future__ import annotations

LDM_DEFAULT = """\
[5. STRATEGY]

Early BO or insufficient evidence:
  Use multiple diverse centers spread across sequence space.
  Soft TR (radius=None) recommended — GP sees all data.

Improving:
  GP finding good regions. Tighten gradually.
  Fewer centers, smaller radius, focused local search.

Stalled:
  Current region exhausted. Add new centers from unexplored clusters.
  More local search restarts, broader radius.

Severely stalled or poor best:
  Abandon current region. Use many diverse centers.
  Broad radius, or NeighborSampling / LatinHyperCube for diversity.

Note on stochasticity:
  BO evaluates {batch_size} candidate(s) per iteration. Occasional
  non-improving samples are normal. If the current trust region is
  producing reasonable candidates, omit update_trust_region to keep it.
  Drastic shifts should be considered sparingly.
  n_evals_in_current_tr shows how many evaluations have been done
  under the current trust region.

Note on soft vs hard TR:
  - radius=None (soft): candidates sampled/searched from neighborhood,
    but GP trains on ALL data — good for exploration.
  - radius set (hard): GP trains ONLY within the radius —
    good for exploitation when you trust the region.
"""

ANTBO_MOCK = """\
[5. STRATEGY — AntBO MIMICRY]

You must replicate AntBO's original local search with adaptive restart.
Follow these rules EXACTLY on every BO iteration.

SEARCH PLAN:

  Normal mode (n_iters_without_improvement < 5):
    Provide update_trust_region as:
      LocalSearch('<best_sequence>', radius=<R>, restart=3, steps=199)
    Where <R> adapts based on n_iters_without_improvement:
      0 stalls  → R = 2
      1-3 stalls → R = 3
      4 stalls  → R = 4

  Restart mode (n_iters_without_improvement >= 5):
    Read the observation history CSV in the prompt.
    Pick the 3 best-scoring sequences (lowest score = best).
    Provide update_trust_region as:
      Or(LocalSearch('<seq1>', radius=4, restart=1, steps=199),
         LocalSearch('<seq2>', radius=4, restart=1, steps=199),
         LocalSearch('<seq3>', radius=4, restart=1, steps=199))
    Budget = 3 * 200 = 600.

  When a restart finds improvement (n_iters_without_improvement resets
  to 0), switch back to Normal mode.

BIAS:
  NEVER provide update_bias. Omit the key entirely.

REVIEW:
  Always respond with:
    {{"action": "take", "ids": [0]}}
  Always TAKE the top candidate (highest acquisition score).
"""

STRATEGIES: dict[str, str] = {
    "ldm-default": LDM_DEFAULT,
    "antbo-mock": ANTBO_MOCK,
}
