import os
import zipfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

_BO_DIR = Path(os.path.realpath(__file__)).parent
REPO_ROOT = str(_BO_DIR.parent)
CACHE_DIR = os.path.join(REPO_ROOT, 'cache')
INIT_DATA_ZIP = os.path.join(CACHE_DIR, 'init_dataset.zip')
INIT_DATA_PATH = os.path.join(CACHE_DIR, 'init_dataset')


def _ensure_init_dataset() -> None:
    """Bootstrap the pre-computed initial dataset.

    Resolution order:
      1. If ``./cache/init_dataset/`` already exists as a directory, use it directly.
      2. Else if ``./cache/init_dataset.zip`` exists, extract it into ``./cache/``.
      3. Else raise ``FileNotFoundError`` instructing the user to download the zip
         from the original AntBO repository and place it under ``./cache/``.

    Called at module import time so any consumer of ``INIT_DATA_PATH`` (e.g.
    ``get_initial_dataset_path``) sees a ready-to-use directory.
    """
    if os.path.isdir(INIT_DATA_PATH):
        return
    if os.path.isfile(INIT_DATA_ZIP):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with zipfile.ZipFile(INIT_DATA_ZIP) as zf:
            zf.extractall(CACHE_DIR)
        return
    raise FileNotFoundError(
        "AntBO initial dataset not found.\n"
        f"Expected one of:\n"
        f"  - {INIT_DATA_PATH}/\n"
        f"  - {INIT_DATA_ZIP}\n"
        f"Please download 'init_dataset.zip' from the original AntBO repository "
        f"and place it under {CACHE_DIR}/."
    )


_ensure_init_dataset()


def get_n_per_cat(n_loosers: int, n_mascottes: int, n_heroes):
    return dict(Loosers=n_loosers, Mascotte=n_mascottes, Heroes=n_heroes)


def get_top_cut_ratio_per_cat(top_cut_ratio_loosers: int, top_cut_ratio_mascottes: int, top_cut_ratio_heroes):
    return dict(Loosers=top_cut_ratio_loosers, Mascotte=top_cut_ratio_mascottes, Heroes=top_cut_ratio_heroes)


class InitialBODataset:

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def get_categories(self) -> np.ndarray:
        return self.data['Type'].values

    def get_index_encoded_x(self) -> np.ndarray:
        return np.vstack(self.data['AA to ind'].values)

    def get_protein_names(self) -> pd.Series:
        return self.data['Protein']

    def get_protein_binding_energy(self) -> pd.Series:
        return self.data['Binding Energy']

    def __len__(self) -> int:
        return len(self.data)


def get_initial_dataset_path(antigen_name: str, n_per_cat: Dict[str, int], top_cut_ratio_per_cat: Dict[str, float],
                             seed: int) -> str:
    """

    Parameters
    ----------
    antigen_name: name of the antigen
    n_per_cat: dictionary {category: number_of_samples}
    top_cut_ratio_per_cat: dictionary {category: top_cut_ratio}
    seed: seed used to generate the dataset

    Returns
    -------

    """
    init_dataset_root = os.path.join(INIT_DATA_PATH, antigen_name, str(seed))
    init_dataset_id: str = ""
    for cat, n_sample in n_per_cat.items():
        if n_sample > 0:
            init_dataset_id += f"{cat}-{n_sample:d}_"
    for cat, top_cut_ratio in top_cut_ratio_per_cat.items():
        if top_cut_ratio > 0:
            init_dataset_id += f"{cat}-{top_cut_ratio:g}_"
    init_dataset_id = init_dataset_id[:-1]
    init_dataset_folder_path = os.path.join(init_dataset_root, init_dataset_id, "init_data")
    os.makedirs(os.path.dirname(init_dataset_folder_path), exist_ok=True)
    return init_dataset_folder_path


def get_initial_dataset_path_(antigen_name: str, top_category: str, n_samples: int, top_cat_top_cut_ratio: float,
                              seed: int) -> str:
    """

    Parameters
    ----------
    antigen_name: name of the antigen
    top_category: name of the top category of protein included in the initial dataset
    n_samples: number of samples in this initial dataset
    top_cat_top_cut_ratio:
    seed

    Returns
    -------

    """
    init_dataset_root = os.path.join(INIT_DATA_PATH, antigen_name, str(seed))
    init_dataset_id = f"{top_category}_n-{n_samples}_top-cat-cut-{top_cat_top_cut_ratio:g}"
    init_dataset_folder_path = os.path.join(init_dataset_root, init_dataset_id, "init_data")
    os.makedirs(os.path.dirname(init_dataset_folder_path), exist_ok=True)
    return init_dataset_folder_path
