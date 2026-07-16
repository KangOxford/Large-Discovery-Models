# AntBO 

**A combinatorial Bayesian Optimisation framework enabling efficient in
silico design of the CDRH3 region.**

![AntBO overview](./figures/AntBO_illustrationPNG.PNG?raw=true)

This repo provides the official implementation of _AntBO_,
as well as all the code needed to reproduce the experiments presented in
[AntBO: Towards Real-World Automated Antibody Design with Combinatorial Bayesian Optimisation](https://www.sciencedirect.com/science/article/pii/S2667237522002764).


## Setup
The code has been tested on  `Ubuntu 18.04.4 LTS` with `python 3.9.7` 

#### Conda environment

```bash
conda env create -f environment.yaml 
conda activate DGM
```

#### Antigen simulator: _Absolut!_

Install [Absolut!](https://github.com/csi-greifflab/Absolut) by following instructions 
of their README: https://github.com/csi-greifflab/Absolut#installation. 

#### Install all pre-computed structures (~33GB)

```bash
PATH_TO_ABSOLUT=./Absolut # put the path of the directory where Absolut! is installed
cd $PATH_TO_ABSOLUT
aria2c -i ../urls.txt --auto-file-renaming=false --continue=true
```

#### Pre-computed initial dataset (`./cache/init_dataset`)

The `--custom_init` feature requires a pre-computed dataset bundle. The first
import of `bo.custom_init` resolves it from `./cache/`:

1. If `./cache/init_dataset/` exists as a directory, it is used directly.
2. Else if `./cache/init_dataset.zip` exists, it is auto-extracted.
3. Else a `FileNotFoundError` is raised, instructing you to download
   `init_dataset.zip` from the original AntBO repository into `./cache/`.

See [`cache/README.md`](cache/README.md) for details. The smoke test
`scripts/smoke/run_custom_init_smoke.py` exercises all three scenarios.

## Repository Layout

```
AntBO/
├── bo/                      core BO algorithm
│   └── ldm/                 LLM extensions (ranked_init, trust_region, antigen_context)
├── actor/  dataloader/  environment/  experiment/  genetic_algorithm/
├── model/  random_search/  task/  utilities/  utils_data/  visualise_results/
├── figures/  docking_visualisations/
├── scripts/                 experiments, demos, plotting, orchestration (not core algo)
│   └── smoke/               smoke tests (no Absolut! required)
├── outputs/                 all generated artefacts — git-ignored
├── cache/                   runtime cache for init_dataset — git-ignored
└── temp/                    untracked scratch files — git-ignored
```

The three git-ignored directories are produced/consumed as follows:

| Dir | Producer | Consumer |
|---|---|---|
| `outputs/ldm_policies/` | human / `scripts/call_llm_policy.py` | `bo/config.yaml` (`llm_*_policy_path`) |
| `outputs/reproduction/` | original AntBO reproduction notebooks | `scripts/make_*.py`, `scripts/plot_*.py` |
| `outputs/full_llm_5seeds_*/` | `scripts/run_full_llm_5seeds.sh` | plotting scripts |
| `outputs/section_ablation/` | `scripts/run_section_ablation.sh` | plotting scripts |
| `outputs/backup/` | `scripts/run_*.sh` pre-run snapshots | ad-hoc |
| `outputs/comparisons/` | `scripts/make_*.py`, `scripts/plot_*.py` | humans |
| `outputs/plots/` | original AntBO reproduction PDFs | humans |
| `outputs/logs/` | remote runs | debugging |
| `cache/init_dataset/` | auto-extracted from zip on first import of `bo.custom_init` | `bo/custom_init.py` |
| `cache/init_dataset.zip` | manually downloaded | auto-extracted |
| `temp/` | 2026-06-27 reorganisation leftovers | none |

## Demo

To run AntBO (as well as other baselines, Genetic Algorithm, 
and Random Search), the user needs to specify a `config.yaml` file containing the parameters of the run ([a default 
config file](./bo/config.yaml) is also provided).

Given a config file, AntBO can be run using the following command line:
```shell
 python ./bo/main.py --config ./bo/config.yaml --n_trials 5 --seed 42 --antigens_file ./dataloader/core_antigens.txt
```
- Using `--n_trials 5` and `--seed 42` implies that the optimisation will be run over 5 random seeds starting from 
seed 42.
- `--antigens_file ./dataloader/core_antigens.txt` indicates that **AntBO** will be run on each of
 the antigens listed in the file [./dataloader/core_antigens.txt](./dataloader/core_antigens.txt).
- The results will be stored in the folder specified in the `config.yaml` file (under the `save_path` field;
  default `outputs/llm_run_outputs`).

For LLM-augmented runs, see [`bo/ldm/README.md`](bo/ldm/README.md) and the orchestration scripts in
[`scripts/`](scripts/README.md) (`run_full_llm_5seeds.sh`, `run_section_ablation.sh`).

For smoke tests that exercise the BO entry point without invoking Absolut!:

```bash
python scripts/smoke/run_bo_smoke.py
python scripts/smoke/run_custom_init_smoke.py
```

#### Results visualisation
To plot the regret curve associated to one run of **AntBO** on an antigen, one can either run the following:

```python
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

from utilities.misc_utils import load_w_pickle
from typing import *

from bo.main import BOExperiments
from task.utils import plot_mean_std

# --- Select for which antigen you want to plot the regret curve --- #
antigen_name = ...

# --- These must match your config.yaml --- #
save_path: str = ... 
kernel_type = "transformed_overlap"  
cdr_constraints = 1
seq_len: int = 11
search_strategy = "local"

init_seed = 42
n_trials = 5

# ----------- Collect evolution of scores for each seed ----------- #

results = []
for seed in range(init_seed, init_seed + n_trials):
    
    result_path_root = BOExperiments.get_path(
        save_path=save_path,
        antigen=antigen_name,
        kernel_type=kernel_type, 
        seed=seed,
        cdr_constraints=cdr_constraints,
        seq_len=seq_len,
        search_strategy=search_strategy,
    )

    result_path = os.path.join(result_path_root, 'results.csv')

    if not os.path.exists(result_path):
        continue

    data = pd.read_csv(result_path).BestValue.values
    
    results.append(data)

# --------------------- Plot the regret curve --------------------- #

ax = plot_mean_std(results)
ax.set_xlabel("Num. evaluations", fontsize=16)
```

or rely on the [convergence curve plotting script](./visualise_results/plot_convergence_curve.py) fed with a 
[configuration file](visualise_results/convergence_curve_config.yaml).

```shell 
python ./visualise_results/plot_convergence_curve.py --config ./visualise_results/convergence_curve_config.yaml
```

## Absolut 3D Visualisation

Follow the instructions in visualise3d.txt




## Cite us
```bibtex
@article{KHAN2023100374,
title = {Toward real-world automated antibody design with combinatorial Bayesian optimization},
journal = {Cell Reports Methods},
pages = {100374},
year = {2023},
issn = {2667-2375},
doi = {https://doi.org/10.1016/j.crmeth.2022.100374},
url = {https://www.sciencedirect.com/science/article/pii/S2667237522002764},
author = {Asif Khan and Alexander I. Cowen-Rivers and Antoine Grosnit and Derrick-Goh-Xin Deik and Philippe A. Robert and Victor Greiff and Eva Smorodina and Puneet Rawat and Rahmad Akbar and Kamil Dreczkowski and Rasul Tutunov and Dany Bou-Ammar and Jun Wang and Amos Storkey and Haitham Bou-Ammar},
keywords = {computational antibody design, structural biology, protein engineering, Bayesian optimization, combinatorial Bayesian optimization, Gaussian processes, machine learning},
abstract = {Summary
Antibodies are multimeric proteins capable of highly specific molecular recognition. The complementarity determining region 3 of the antibody variable heavy chain (CDRH3) often dominates antigen-binding specificity. Hence, it is a priority to design optimal antigen-specific CDRH3 to develop therapeutic antibodies. The combinatorial structure of CDRH3 sequences makes it impossible to query binding-affinity oracles exhaustively. Moreover, antibodies are expected to have high target specificity and developability. Here, we present AntBO, a combinatorial Bayesian optimization framework utilizing a CDRH3 trust region for an in silico design of antibodies with favorable developability scores. The in silico experiments on 159 antigens demonstrate that AntBO is a step toward practically viable in vitro antibody design. In under 200 calls to the oracle, AntBO suggests antibodies outperforming the best binding sequence from 6.9 million experimentally obtained CDRH3s. Additionally, AntBO finds very-high-affinity CDRH3 in only 38 protein designs while requiring no domain knowledge.}
}
```

## Contributors
**Asif Khan**, **Alexander I. Cowen-Rivers**, **Antoine Grosnit**, **Derrick-Goh-Xin Deik**, 
**Philippe A. Robert**, **Victor Greiff**, **Eva Smorodina**, **Puneet Rawat**, **Rahmad Akbar**, 
**Kamil Dreczkowski**, **Rasul Tutunov**, **Dany Bou-Ammar**, **Jun Wang**, **Amos Storkey**,**Haitham Bou-Ammar**
-- _Huawei Noah's Ark lab_.
