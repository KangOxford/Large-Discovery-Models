# temp/

Files moved here during the 2026-06-27 repository reorganisation. None of them
are required for AntBO to run; they are kept for traceability only. Entire
directory is **git-ignored**.

## Contents

| File | Origin | Notes |
|---|---|---|
| `incomplete_root_results/BO_transformed_overlap/antigen_1ADQ_A_..._seed_42_.../llm_antigen_context.json` | `results/` at repo root | Single incomplete run — only the initial antigen-context JSON was saved before interruption. Full 5-seed run is in `outputs/full_llm_5seeds_init100000_ninit50_iter100/`. |
| `main.py.rej` | `bo/` | Rejected hunk from the original LDM patch application. Kept for traceability. |
| `config.yaml.before_section_ablation_20260625_150918` | `bo/` | Snapshot of `bo/config.yaml` taken right before the section-ablation experiments on 2026-06-25 15:09. |
| `llm_generated_policy.json` | `bo/` | One-off output of a sample LLM policy call. Not referenced by any code path. |
| `example_antigen_context_prompt_snapshot.json` | `bo/` | Reference example of an LLM prompt snapshot. Not referenced by any code path. |
| `llm_antbo_uncommitted_changes_20260626_173540.patch` | repo root | The patch that introduced the LDM extensions. Now fully applied; kept as historical record. |
| `table_search_test.csv` | repo root | 502 KB CSV used to exercise the `tabular_search_csv` config option. |
| `duplicate_antigen_context.py` | `plots/` | Byte-identical copy of `bo/antigen_context.py` (now `bo/ldm/antigen_context.py`). Was accidentally placed in `plots/`. |

## When to delete

If `cache/init_dataset` bootstrap or any future regression requires space,
this directory can be removed in full without affecting AntBO. The smoke tests
do not depend on any file in `temp/`.