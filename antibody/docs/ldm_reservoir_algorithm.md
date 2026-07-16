# LDM Reservoir Algorithm Proposal

This is a standalone implementation plan for the new LDM variant. It is placed
next to the existing AntBO/LDM code and does not overwrite existing files.

## Feasibility

The algorithm is feasible because AntBO already separates three pieces cleanly:
LLM-produced DSL trust regions, GP/acquisition scoring, and discrete candidate
sampling/local search. The new method only changes the selection protocol: the
LLM now proposes five strategy-level trust regions, each region yields one
representative candidate, and the final query is selected by acquisition-based
argmax or softmax.

## Algorithm

At BO iteration `t`, build context `C_t` from antigen context, full history,
best sequence, score trend, and current BO status. The LLM planner returns five
search strategies `z_1...z_5`, each as an AntBO DSL atom such as
`LocalSearch(center, radius, restart, steps)` or
`NeighborSampling(center, radius, mut_pr, budget)`. These strategies are the
structured prior `p_llm(x | C_t)`.

For each strategy, run the existing AntBO acquisition executor to create a local
candidate pool and score all candidates with GP posterior and acquisition. Keep
one best representative candidate per strategy. This produces up to five protein
candidates. The final candidate is selected by either direct acquisition argmax
or softmax sampling over acquisition values:

`P(x_i) proportional to exp(eta * acq(x_i))`.

Bias atoms can still be used inside each pool through
`combined = acq + bias_weight * bias(seq)`, while the final selection can use
pure acquisition or the combined score depending on configuration.

## Edge Tests

The new tests cover: five strategy argmax selection, softmax probability
normalization, fallback when the LLM returns fewer than five strategies, and
budget capping when a strategy requests too many evaluations. The smoke script
uses a fake GP/acquisition to verify the discrete loop without Absolut or an LLM
API key.

## Files

- `bo/ldm_reservoir/config.py`: standalone configuration.
- `bo/ldm_reservoir/planner.py`: LLM JSON -> five DSL strategy atoms.
- `bo/ldm_reservoir/session.py`: five pools -> five candidates -> final selection.
- `scripts/smoke/run_ldm_reservoir_smoke.py`: no-API smoke runner.
- `tests/bo/ldm_reservoir/`: unit tests for planner and session.
