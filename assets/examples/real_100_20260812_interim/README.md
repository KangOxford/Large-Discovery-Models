# Refactor Campaign Comparison (Interim)

These plots were generated from live real-evaluator campaign snapshots on
2026-08-12 with `scripts/plot_campaigns.py`. They use the same metrics and plot
style as `assets/examples/real_100_20260809/`, but they are not final 100-run
results because all three refactor campaigns were still running when captured.

## Snapshot

| Task | Captured progress | Interim result | Previous complete result |
| --- | --- | --- | --- |
| Antibody UCB | 57/100 Absolut evaluations | Best energy improved from -88.56 after initialization to -98.10 at evaluation 31. | -96.72 after 100 evaluations. |
| Small-molecule EHVI | 24/100 Vina plus activity evaluations | Pareto hypervolume reached 18.715996374108368. | 22.8080517046179 after 100 evaluations. |
| nanoGPT LCB N4H4 | 4 finite warm-up evaluations; 0/100 outer iterations | Best warm-up `val_bpb` was 0.986868. | Best finite `val_bpb` was 0.981844 after the complete campaign. |

`previous_vs_refactor_interim.png` places the previous complete campaigns in
the top row and these refactor snapshots in the bottom row. Do not compare the
new curve endpoints as if they had equal evaluation budgets. Regenerate the
plots after the active campaigns complete before using them as final evidence.

## Outputs

- `previous_vs_refactor_interim.png` and `.pdf`: direct six-panel comparison.
- `antibody_ucb_trajectory.png` and `.pdf`: refactor antibody snapshot.
- `small_molecule_ehvi_trajectory.png` and `.pdf`: refactor molecule snapshot.
- `nanogpt_lcb_trajectory.png` and `.pdf`: refactor nanoGPT snapshot.
- `ldm_three_tasks.png` and `.pdf`: combined refactor-only view.
- Corresponding CSV files contain the compact plotted curves.

Raw campaign artifacts are intentionally not copied into this directory.
