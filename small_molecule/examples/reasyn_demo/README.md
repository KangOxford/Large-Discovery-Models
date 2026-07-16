# ReaSyn Demo Inputs

This directory shows the smallest handoff from PDF-extracted seed SMILES and
literature activity values to ReaSyn hit expansion. It skips only the PDF
extraction step.

- `autodock_markush_demo_docking_results.csv` has a historical filename, but
  its contents are a small extraction-output seed table using compounds 37, 38,
  and 39 from
  `output/formula_workflow_outputs/reversible_kras_g13d_inhibitors_formula_to_smiles.csv`.
  It includes SMILES plus literature IC50 values so the ReaSyn bridge can pick
  seeds without re-running PDF extraction.

Prepare ReaSyn inputs from the demo seed table. If no `--smiles-path` is
provided, this demo CSV is used by default:

```bash
python3 docking_to_analog_search_example.py \
  --top-n 3 \
  --output-dir output/reasyn_demo_prepared
```

The script writes:

- `reasyn_input.txt`: a SMILES text file with a `SMILES` header.
- `reasyn_manifest.json`: the selected seed molecules and ReaSyn settings.
- `run_reasyn_commands.sh`: a reproducible command for the ReaSyn checkout.

Run ReaSyn when you have installed the upstream checkout, trained AR/Edit Bridge
checkpoints, and processed building-block indexes:

```bash
python3 docking_to_analog_search_example.py \
  --smiles-path examples/reasyn_demo/autodock_markush_demo_docking_results.csv \
  --top-n 3 \
  --reasyn-repo /path/to/ReaSyn \
  --model-paths data/trained_model/nv-reasyn-ar-166m-v2.ckpt,data/trained_model/nv-reasyn-eb-174m-v2.ckpt \
  --run-reasyn \
  --output-dir output/reasyn_demo_run
```

Then dock and rank the generated analogs:

```bash
python3 extract_and_dock.py dock \
  --csv output/reasyn_demo_run/reasyn_analogs.csv \
  --allow-unreviewed \
  --pdb-id 8UN5 \
  --chain-id A \
  --work-dir output/docking_work \
  --output-dir output/docking_work/reasyn_demo_analog_results \
  --exhaustiveness 4 \
  --num-modes 3 \
  --analog-top-k 3
```

The analog docking step writes `analog_group_topk.csv` and
`analog_overall_best.csv` in addition to the full docking result tables. It
prints the current best analog for each seed and keeps the same rows in
`docking_metadata.json` under `analog_summary.group_best` for future optimizer
integration.

The StrBO demo command uses the same seed CSV:

```bash
python3 reasyn_strbo_optimization.py \
  --smiles-path examples/reasyn_demo/autodock_markush_demo_docking_results.csv \
  --top-n-seeds 3 \
  --n-trials 3 \
  --top-k 3 \
  --reasyn-repo /path/to/ReaSyn \
  --model-paths data/trained_model/nv-reasyn-ar-166m-v2.ckpt,data/trained_model/nv-reasyn-eb-174m-v2.ckpt \
  --output-dir output/reasyn_demo_strbo
```
