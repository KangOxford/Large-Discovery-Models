# scripts/

Experimental, demo, plotting, and orchestration scripts. **Not part of the core
algorithm** — these files can be safely deleted without breaking AntBO itself.

All scripts assume the **repository root** (`AntBO/`) as the working directory
unless stated otherwise.

## Top-level scripts

| Script | Purpose | Run from |
|---|---|---|
| `run_full_llm_5seeds.sh` | Full-LLM AntBO sweep (5 seeds × 5 antigens). Outputs land in `outputs/full_llm_5seeds_init100000_ninit50_iter100/`. | `bash scripts/run_full_llm_5seeds.sh` |
| `run_section_ablation.sh` | Three ablation runs (Section 1 only / Section 2 only / Section 3). Outputs land in `outputs/section_ablation/`. | `bash scripts/run_section_ablation.sh` |
| `set_llm_sections.py` | Flip `llm_ranked_init` / `llm_trust_region` / `llm_antigen_context` flags in `bo/config.yaml`. Used by `run_section_ablation.sh`. | `python scripts/set_llm_sections.py true true true` |
| `regenerate_structure_symlinks.py` | Retarget Absolut structure symlinks from `/mnt/data0/shared/AntBO/Absolut/src/structures/...` to `/mnt/data0/ys/Absolut/src/structures/...`. Dry-run by default. | `python scripts/regenerate_structure_symlinks.py /path/to/link_dir --pattern '*Structures.txt'` |
| `call_llm_policy.py` | Sample LLM provider wrapper (reads prompt JSON, returns policy JSON). Wire up via `llm_policy_command` in `bo/config.yaml`. | manual invocation |
| `make_llm_vs_reproduction_table.py` | Build a wide comparison table of LLM sections vs reproduction. Reads from `outputs/reproduction/`, `outputs/section_ablation/`, `outputs/backup/`. Writes to `outputs/comparisons/tables/`. | `python scripts/make_llm_vs_reproduction_table.py` |
| `make_numbered_llm_comparison.py` | Produce a numbered/comma-cleaned version of the wide table. | `python scripts/make_numbered_llm_comparison.py` |
| `make_vertical_llm_vs_reproduction_csv.py` | Pivot the wide table into a vertical (long) format. | `python scripts/make_vertical_llm_vs_reproduction_csv.py` |
| `plot_best_so_far_5seeds.py` | Aggregate the 5-seed full-LLM run: per-antigen best-so-far mean ± std band, one subplot per antigen. Reads from `outputs/full_llm_5seeds_init100000_ninit50_iter100/`. Writes to `outputs/comparisons/plots/best_so_far_5seeds_subplots.png`. | `python scripts/plot_best_so_far_5seeds.py` |
| `plot_llm_sections_vs_reproduction.py` | Side-by-side plot of Section 1 / 2 / 3 / Full LLM vs reproduction. Writes into `outputs/comparisons/plots/llm_sections_vs_reproduction/`. | `python scripts/plot_llm_sections_vs_reproduction.py` |
| `plot_reproduction_vs_llm_5seeds.py` | 2×3 subplot grid overlaying reproduction (no-LLM, mean of 10 seeds) against the 5-seed LDM AntBO run (mean ± std band). Writes to `outputs/comparisons/plots/reproduction_vs_llm_5seeds_subplots.png`. | `python scripts/plot_reproduction_vs_llm_5seeds.py` |
| `plot_reproduction_vs_llm_from_aggregate.py` | Per-antigen single-PNG overlay (reproduction vs LDM-AntBO seed-42, single run). Writes into `outputs/comparisons/reproduction_vs_llm_5antigens/`. Kept for the older single-seed comparison; for the new aggregated view use `plot_reproduction_vs_llm_5seeds.py`. | `python scripts/plot_reproduction_vs_llm_from_aggregate.py` |
| `view_llm_vs_reproduction_vertical.py` | CLI viewer for the vertical CSV. | `python scripts/view_llm_vs_reproduction_vertical.py` |
| `clean_absolut_temp.sh` | Purge stale Absolut temp files left by crashed AntBO runs. Matches the `run{pid}_{epoch_ms}_` prefix plus exact suffix structure (`TempCDR3_*`, `TempBindingsFor*_t*_Part1_of_1`, `*FinalBindings_Process_*_Of_*`), so Absolut's own data files are never touched. Default threshold 30 min. | `AGE_MINUTES=60 bash scripts/clean_absolut_temp.sh` |

## smoke/

Minimal smoke tests that exercise the repository without invoking Absolut!.

```bash
python scripts/smoke/run_bo_smoke.py           # import + 1 BO trial under bbox.tool=random
python scripts/smoke/run_custom_init_smoke.py  # validate ./cache/init_dataset bootstrap
```

See [`smoke/README.md`](smoke/README.md) for details.

## Path conventions

Every script reads/writes relative to the **repository root**, regardless of
where it sits inside `scripts/`. The canonical paths are:

- Reproduction data: `outputs/reproduction/`
- LDM policies:      `outputs/ldm_policies/`
- Run outputs:        `outputs/llm_run_outputs/BO_transformed_overlap/` (set via `save_path` in `bo/config.yaml`)
- Section ablation:   `outputs/section_ablation/`
- Backup:             `outputs/backup/`
- Comparison tables:  `outputs/comparisons/tables/`
- Comparison plots:   `outputs/comparisons/plots/`
- Remote logs:        `outputs/logs/`
