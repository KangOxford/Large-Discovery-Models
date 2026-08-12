# Protein Inverse Folding Resources

`seed_design.py` is the starter editable region from the MLS-Bench
`ai4bio-protein-inverse-folding` task. `smoke_prelude.py` contains the fixed
geometry helpers needed to exercise that region without the datasets.
`baseline_leaderboard.csv` pins the three upstream baseline rows used to
calibrate bounded-power scoring.

The upstream budget checker at the pinned revision reports ProteinMPNN at
981,524 parameters, PiFold at 4,278,085, and GVP at 462,478. Its 1.05x-largest
rule therefore sets the candidate cap to 4,491,989 parameters.

Source task:
https://github.com/Imbernoulli/MLS-Bench/tree/main/tasks/ai4bio-protein-inverse-folding

The registration was implemented against upstream tree
`da06dffcc79826dc3d22dec53ead310c430b6535`. The full upstream
`custom_invfold.py` scaffold and the CATH/TS datasets are intentionally not
vendored. Supply them with `--scaffold-path` and `--data-root` for benchmark
runs. Runtime outputs belong under `../runs/` and are ignored.
