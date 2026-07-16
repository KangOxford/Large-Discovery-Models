# Reservoir LDM prototype

This package is a standalone reimplementation of the LDM idea without editing
or replacing the existing `bo/ldm` files.

Algorithm per BO step:

1. `ReservoirPlanner` asks the LLM for `K=5` search strategies `z` conditioned
   on history/context `C_t`.
2. Each strategy is an AntBO DSL trust region such as `LocalSearch(...)` or
   `NeighborSampling(...)`.
3. `ReservoirAcquisitionSession` executes each strategy independently and gets
   one representative protein candidate from each strategy pool.
4. The final candidate is selected from the five representatives by either
   `argmax(acquisition)` or softmax sampling over acquisition values.

The implementation reuses original AntBO components:

- DSL atoms: `bo.ldm.dsl.search_space`
- Bias atoms: `bo.ldm.dsl.bias`
- GP/acquisition batch evaluation: `bo.ldm.acquisition.parallel_search`

It deliberately does not modify `bo/localbo_cat.py`, `bo/ldm`, or `bo/config.yaml`.
