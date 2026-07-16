# M1 Oversample Methods Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and evaluate three M1 variants: BO oversampling, one-step LLM-only, and LLM-seeded analogue oversampling.

**Architecture:** Reuse the existing tilted case2 loop, candidate records, prompt schemas, and CLI. Add a shared pool-maintenance step that reduces oversized reservoirs to `max_candidates_per_round` by q0-weighted sampling before BO/EHVI, add one direct-LLM sequential baseline, and add one LLM seed plus analogue reservoir builder. The analogue method must use the configured analogue generator path; it must not silently fall back to a different molecule generator.

**Tech Stack:** Python, pytest, RDKit-backed canonicalization, existing OpenAI-compatible LLM client, existing ReaSyn/analog adapter.

---

### Task 1: Pool Maintenance

**Files:**
- Create: `strbo_v1/ldm_tilted_case2/pool_maintenance.py`
- Modify: `strbo_v1/ldm_tilted_case2/loop.py`
- Test: `tests/test_tilted_pool_maintenance.py`

- [ ] Add a failing test proving an oversized candidate list is reduced to the configured pool size using q0 mass, not first-N truncation.
- [ ] Implement q0-weighted Gumbel top-k maintenance and trace metadata for original and maintained pool size.
- [ ] Verify targeted tests pass.

### Task 2: Direct LLM Variants

**Files:**
- Modify: `strbo_v1/ldm_tilted_case2/config.py`
- Modify: `strbo_v1/ldm_tilted_case2/methods/direct_llm.py`
- Modify: `strbo_v1/ldm_tilted_case2/loop.py`
- Modify: `scripts/run_case2_three_methods.py`
- Test: `tests/test_tilted_methods_m1.py`
- Test: `tests/test_run_case2_three_methods.py`

- [ ] Add failing tests for `m1_stratified_direct_llm_oversample_sir` and `m1_llm_one_step`.
- [ ] Make the oversample method use stratified direct LLM batches and BO/EHVI selection.
- [ ] Make the one-step method request exactly one LLM candidate per round and skip BO/EHVI.
- [ ] Verify CLI accepts both methods.

### Task 3: LLM-Seed Analogue BO Method

**Files:**
- Create: `strbo_v1/ldm_tilted_case2/methods/llm_seed_analog.py`
- Modify: `strbo_v1/ldm_tilted_case2/prompts.py`
- Modify: `strbo_v1/ldm_tilted_case2/config.py`
- Modify: `strbo_v1/ldm_tilted_case2/loop.py`
- Modify: `scripts/run_case2_three_methods.py`
- Test: `tests/test_tilted_methods_m1_analog.py`

- [ ] Add failing tests proving the method asks LLM for seeds, expands each seed through `analog_fn`, and builds an oversized reservoir.
- [ ] Implement the builder using existing seed schema and q0 source mass from analogue source counts.
- [ ] Add CLI knobs for seed count and total analogue budget.
- [ ] Verify mock loop and CLI tests pass.

### Task 4: Remote Evaluation

**Files:**
- Remote-only launch scripts under `/mnt/data0/shared/ldm_tilted_case2_three_methods/`

- [ ] Sync the branch to an isolated remote code directory.
- [ ] Run targeted tests remotely.
- [ ] Launch three isolated real evaluations with DashScope `deepseek-v4-flash`, SK kernel, and `/mnt/data0` TMPDIR.
- [ ] For the analogue method, verify that the remote run is using the configured ReaSyn analogue generator and fail loudly if ReaSyn is unavailable.
- [ ] Confirm all three processes start, write logs, and produce initial rounds.
