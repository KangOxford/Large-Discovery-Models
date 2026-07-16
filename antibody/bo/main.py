import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, Any, Dict, get_args

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path.insert(0, ROOT_PROJECT)

from task import TableFilling
from task import BaseTool
from bo.ldm.antigen_context import (
    collect_absolut_antigen_context,
    save_json,
    save_llm_context_snapshot,
)
from utilities.misc_utils import log
from bo.custom_init import get_initial_dataset_path, InitialBODataset, get_top_cut_ratio_per_cat, get_n_per_cat
from bo.botask import BOTask
from bo.localbo_utils import ACQ_FUNCTIONS, SEARCH_STRATS, AA
from bo.optimizer import Optimizer
from bo.utils import save_w_pickle, load_w_pickle


def get_x_y_from_csv(csv_path: str) -> tuple[np.ndarray, torch.tensor]:
    data = pd.read_csv(csv_path)
    x = data["x"].values
    from bo.localbo_utils import AA_to_idx
    x = np.array([[AA_to_idx[cat] for cat in xx] for xx in x])
    y = torch.tensor(data["y"].values).reshape(-1, 1)
    return x, y


class BOExperiments:
    def __init__(self, config: Dict[str, Any], cdr_constraints: bool, seed: int) -> None:
        """
        Args:
             config: dictionary of parameters for BO
                 acq: choice of the acquisition function
                 ard: whether to enable automatic relevance determination
                 save_path: path to save model and results
                 kernel_type: choice of kernel
                 normalise: normalise the target for the GP
                 batch_size: batch size for BO
                 max_iters: maximum evaluations for BO
                 n_init: number of initialising random points
                 min_cuda: number of initialisation points to use CUDA
                 device: default 'cpu' if GPU specify the id
                 seq_len: length of seqence for BO
                 bbox: dictionary of parameters of blackbox
                     antigen: antigen to use for BO
                 vector_representation_table_csv: vector re
             seed: random seed
        """

        self.config = config
        self.table_of_antibodies_as_inds = None
        self.table_of_embeddings = None
        self.embedding_from_array_dict = None
        if self.config["tabular_search_csv"] is not None:
            print(f"Tab. BO setting: will select antibodies among available ones from: {config['tabular_search_csv']}")
            aux_tab = self.get_table_of_antibodies(tabular_search_csv=self.config["tabular_search_csv"])
            self.table_of_antibodies_as_inds, self.table_of_embeddings, self.embedding_from_array_dict = aux_tab
            if self.table_of_embeddings is not None:
                print("Will use pre-computing embeddings provided in the csv.")
        self.llm_initial_points = None
        self.seed = seed
        self.cdr_constraints = cdr_constraints
        # Sanity checks
        if self.config['acq'] not in get_args(ACQ_FUNCTIONS):
            raise ValueError(f"Unknown acquisition function choice {self.config['acq']} (not in {ACQ_FUNCTIONS})")
        self.search_strat = self.config.get('search_strategy', 'local')
        assert self.search_strat in get_args(SEARCH_STRATS), print(f"{self.search_strat} not in {SEARCH_STRATS}")

        print(f"Search Strategy {self.search_strat}")

        self.custom_initial_dataset: Optional[InitialBODataset] = None
        if self.custom_initial_dataset_path:
            print("Loading custom initial dataset")
            self.custom_initial_dataset = load_w_pickle(self.custom_initial_dataset_path)
            if not os.path.exists(self.custom_initial_dataset_path + '.pkl'):
                raise ValueError(self.custom_initial_dataset_path + '.pkl')
            if self.config['n_init'] != len(self.custom_initial_dataset):
                raise ValueError(f"{self.config['n_init']} != {len(self.custom_initial_dataset)}")

        if self.config['kernel_type'] is None:
            default_kernel = 'transformed_overlap'
            self.config['kernel_type'] = default_kernel
            print(f"Kernel Not Specified Using Default {default_kernel}")

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        print(f"Results of this run will be saved in {self.path}")

        self.res = pd.DataFrame(np.nan, index=np.arange(int(self.config['max_iters'] * self.config['batch_size'])),
                                columns=['Index', 'LastValue', 'BestValue', 'Time', 'LastProtein', 'BestProtein']) 

        self.nb_aas = len(AA)
        self.n_categories = np.array([self.nb_aas] * self.config['seq_len'])
        self.antigen_context = None
        _llm_cfg = self.config.get('llm', {})
        _llm_enabled = _llm_cfg.get('llm_init_enabled', True) or _llm_cfg.get('llm_loop_enabled', True)
        self.config['bbox']['seed'] = self.seed
        self.config['bbox']['seq_len'] = self.config['seq_len']
        _is_smoke = self.config['bbox'].get('antigen', '').upper().startswith('SMOKE')
        if self.config.get('llm_antigen_context', False) and _llm_enabled:
            if _is_smoke:
                self.antigen_context = {
                    "antigen_id": self.config['bbox']['antigen'],
                    "landscape_type": "synthetic_multi_peak",
                    "description": (
                        "Synthetic landscape for BO validation. Energy is "
                        "a sum of Hamming-distance-decayed peaks. This is "
                        "NOT a real antibody-antigen binding energy."
                    ),
                    "n_peaks": self.config['bbox'].get('n_peaks', 5),
                    "seq_len": self.config['seq_len'],
                    "amplitude_range": [-80.0, -50.0],
                    "scale_range": [1.5, 3.0],
                    "formula": "f(x) = sum_p amplitude_p * exp(-hamming(x, center_p) / scale_p)",
                }
            else:
                self.antigen_context = collect_absolut_antigen_context(
                    bbox_config=self.config['bbox'],
                    timeout_s=int(self.config.get('llm_antigen_context_timeout_s', 30)),
                    include_raw=bool(self.config.get('llm_antigen_context_include_raw', False)),
                )
            antigen_context_path = os.path.join(self.path, 'llm_antigen_context.json')
            save_json(self.antigen_context, antigen_context_path)
            print(f"Saved LLM antigen context in {antigen_context_path}")
        # llm_policy_command was removed in the LDM redesign (Phase 10): the
        # orchestrator now drives LLM calls directly via OpenAIClient.
        if self.config.get('llm_policy_command'):
            print("[note] llm_policy_command removed; orchestrator handles LLM calls.")
        # llm_ranked_init was removed in the LDM redesign (Q1=delete soft prior
        # in v6). The orchestrator now controls TR + bias; initial points use
        # plain random + CDR constraints (handled by Optimizer). The yaml key
        # is accepted but ignored to preserve compatibility.
        if self.config.get('llm_ranked_init', False):
            print("[note] llm_ranked_init removed in LDM redesign; using random init.")       
        self.start_itern = 0
        self.f_obj = BOTask(
            device=self.config['device'], n_categories=self.n_categories,
            seq_len=self.config['seq_len'], bbox=self.config['bbox'], normalise=False
        )

    @staticmethod
    def get_path(save_path: str, antigen: str, kernel_type: str, seed: int, cdr_constraints: int, seq_len: int,
                 search_strategy: str,
                 custom_init_dataset_path: Optional[str] = None, tabular_search_csv: Optional[str] = None):
        path: str = f"{save_path}/BO_{kernel_type}/antigen_{antigen}" \
                    f"_kernel_{kernel_type}_search-strat_{search_strategy}_seed_{seed}" \
                    f"_cdr_constraint_{bool(cdr_constraints)}_seqlen_{seq_len}"
        if tabular_search_csv is not None:
            path += f"_tabsearch-{os.path.basename(tabular_search_csv)[:-4]}"
        if custom_init_dataset_path:
            custom_init_id = os.path.basename(os.path.dirname(custom_init_dataset_path))
            custom_init_id_seed = os.path.basename(os.path.dirname(os.path.dirname(custom_init_dataset_path)))
            path += f"_custom-init-id-{custom_init_id}_seed_{custom_init_id_seed}"
        return os.path.abspath(path)

    @property
    def path(self) -> str:
        return self.get_path(
            save_path=self.config['save_path'],
            antigen=self.config['bbox']['antigen'],
            kernel_type=self.config['kernel_type'],
            search_strategy=self.config['search_strategy'],
            seed=self.seed,
            cdr_constraints=self.cdr_constraints,
            seq_len=self.config['seq_len'],
            custom_init_dataset_path=self.custom_initial_dataset_path,
            tabular_search_csv=self.config["tabular_search_csv"]
        )

    @property
    def custom_initial_dataset_path(self) -> Optional[str]:
        if not self.config.get('custom_init', False):
            return None
        return get_initial_dataset_path(
            antigen_name=self.config['bbox']['antigen'],
            n_per_cat=get_n_per_cat(
                n_loosers=self.config['custom_init_n_loosers'],
                n_mascottes=self.config['custom_init_n_mascottes'],
                n_heroes=self.config['custom_init_n_heroes']
            ),
            top_cut_ratio_per_cat=get_top_cut_ratio_per_cat(
                top_cut_ratio_loosers=self.config['custom_init_top_cut_loosers'],
                top_cut_ratio_mascottes=self.config['custom_init_top_cut_mascottes'],
                top_cut_ratio_heroes=self.config['custom_init_top_cut_heroes']
            ),
            seed=self.config['custom_init_seed']
        )

    @property
    def torch_rd_state_path(self) -> str:
        return os.path.join(self.path, 'torch_rd_state.pt')

    @property
    def np_rd_state_path(self) -> str:
        return os.path.join(self.path, "np_rd_state.pkl")

    @property
    def random_rd_state_path(self) -> str:
        return os.path.join(self.path, "random_rd_state.pkl")

    def load(self) -> Optimizer:
        res_path = os.path.join(self.path, 'results.csv')
        optim_path = os.path.join(self.path, 'optim.pkl')
        if os.path.exists(optim_path):
            optim = load_w_pickle(optim_path)
            if os.path.exists(self.torch_rd_state_path):
                torch_random_state = torch.load(self.torch_rd_state_path)
                torch.set_rng_state(torch_random_state)
            if os.path.exists(self.np_rd_state_path):
                np_rd_state = load_w_pickle(self.np_rd_state_path)
                np.random.set_state(np_rd_state)
            if os.path.exists(self.random_rd_state_path):
                rd_state = load_w_pickle(self.random_rd_state_path)
                random.setstate(rd_state)
            if os.path.exists(res_path):
                columns = ['Index', 'LastValue', 'BestValue', 'Time', 'LastProtein', 'BestProtein']
                self.res = pd.read_csv(res_path, usecols=columns)
                self.start_itern = (len(self.res) - self.res['Index'].isna().sum()) // self.config['batch_size']
            print(f"-- Resume -- Already observed {optim.casmopolitan.n_evals}")
            return optim

    def save(self, optim: Optimizer) -> None:
        optim_path = os.path.join(self.path, 'optim.pkl')
        res_path = os.path.join(self.path, 'results.csv')
        save_w_pickle(obj=optim, path=optim_path)
        self.res.to_csv(res_path)
        # save random states
        torch.save(torch.get_rng_state(), self.torch_rd_state_path)
        save_w_pickle(obj=np.random.get_state(), path=self.np_rd_state_path)
        save_w_pickle(obj=random.getstate(), path=self.random_rd_state_path)

    def results(self, optim: Optimizer, x: np.ndarray, itern: int, rtime: float) -> None:
        y = np.array(optim.casmopolitan.fx)
        if y[:itern + 1].shape[0] == 0:
            return

        antibodies = self.f_obj.idx_to_seq(x)

        def add_res(step: int, y_val: float, protein: str) -> None:
            argmin = np.argmin(y[:step + 1])
            x_best = ''.join([self.f_obj.fbox.idx_to_AA[ind] for ind in optim.casmopolitan.x[argmin].flatten()])
            self.res.iloc[step, :] = [step, y_val, float(np.min(y[:(step + 1)])), rtime, protein, x_best]

        for idx, j in enumerate(range(itern * self.config['batch_size'], (itern + 1) * self.config['batch_size'])):
            add_res(step=j, y_val=float(y[j].item() if hasattr(y[j], 'item') else y[j]), protein=antibodies[idx])

    def _build_orchestrator(self):
        """Construct the LDM Orchestrator from config.

        Returns ``None`` if the LDM is disabled (or no LLM client configured).
        """
        ldm_cfg_dict = self.config.get("llm", {})
        if not ldm_cfg_dict:
            return None
        if not (ldm_cfg_dict.get("llm_init_enabled", True) or ldm_cfg_dict.get("llm_loop_enabled", True)):
            return None
        # Inject batch_size from top-level config so prompts can reference it
        ldm_cfg_dict.setdefault("batch_size", self.config.get("batch_size", 1))
        # Lazy imports keep bo/main.py decoupled from bo.ldm internals
        # beyond the public API.
        from bo.ldm import DSLConfig, Orchestrator, OpenAIClient

        config = DSLConfig.from_yaml(ldm_cfg_dict)
        # OpenAIClient reads LLM_API_KEY / LLM_BASE_URL from .env (loaded by
        # python-dotenv). No CLI fallback, no litellm dependency.
        client = OpenAIClient()
        # Per-(antigen, seed) decision log co-located with results.csv.
        # The llm_decisions_log YAML key is ignored (kept only for backward
        # compat with old configs).
        decision_log_path = os.path.join(self.path, 'llm_decisions.json')
        return Orchestrator(
            config=config,
            llm_client=client,
            decision_log_path=decision_log_path,
        )

    def _build_llm_initial_points(self, orchestrator):
        """LLM-guided initialization via LHS + bias scoring.

        Calls the orchestrator once (iteration=0, empty history) to get a
        bias DSL. Then generates ``init_pool_size`` Latin Hypercube samples,
        filters by CDR constraints, scores with the bias DSL, and selects
        the top ``n_init``.

        Returns ``None`` (→ random init) if orchestrator is None, LLM
        returns no bias DSL, or all retries fail.
        """
        if orchestrator is None:
            return None

        _llm_cfg = self.config.get('llm', {})
        if not _llm_cfg.get('llm_init_enabled', True):
            print("[init] LLM init disabled; using random LHS init.")
            return None

        from bo.ldm import OrchestratorStatus

        _llm_cfg = self.config.get('llm', {})
        init_pool_size = int(_llm_cfg.get('init_pool_size', 100000))
        n_init = int(self.config['n_init'])

        # 1. Build init status (empty history, iteration 0)
        status = OrchestratorStatus(
            iteration=0,
            antigen_id=self.config['bbox']['antigen'],
            antigen_seed=self.seed,
            iter_seed=0,
            full_history=[],
            antigen_context=self.antigen_context or {},
        )

        # 2. Call orchestrator (LLM call + retries + fallback)
        print("[init] Calling LLM for initialization (TR + bias)...")
        decision = orchestrator.step(status)

        bias_dsl = decision.bias_dsl
        search_dsl = decision.search_dsl
        if bias_dsl is None and search_dsl is None:
            print("[init] No DSL from LLM; using random LHS init.")
            return None

        if search_dsl is not None:
            print(f"[init] LDM TR: {search_dsl!r}")
        if bias_dsl is not None:
            print(f"[init] LDM bias: {bias_dsl!r}")
        if decision.rationale:
            print(f"[init] Rationale: {decision.rationale}")

        # 3. Generate candidate pool
        from bo.localbo_utils import check_cdr_constraints

        if search_dsl is not None:
            from bo.ldm.dsl.search_space import Or as OrAtom, LatinHyperCubeSampling, NeighborSampling
            atoms = search_dsl.children if isinstance(search_dsl, OrAtom) else [search_dsl]
            pool_parts = []
            for atom in atoms:
                if isinstance(atom, (NeighborSampling, LatinHyperCubeSampling)):
                    n = atom.budget
                    print(f"[init] Sampling {n} from {atom!r}...")
                    samples = atom.sample(n=n, timeout_s=30.0)
                    pool_parts.extend(samples)
                else:
                    print(f"[init] Skipping non-sampling atom: {atom!r}")
            pool = np.array(pool_parts, dtype=int) if pool_parts else np.zeros((0, 11), dtype=int)
        else:
            # No TR: use LHS over the full space
            print(f"[init] Generating {init_pool_size} LHS candidates...")
            from bo.localbo_utils import latin_hypercube, from_unit_cube, onehot2ordinal
            from bo.utils import get_dim_info
            cat_dims = get_dim_info(self.n_categories)
            n_onehot = int(np.sum(self.n_categories))
            lb = np.zeros(n_onehot)
            ub = np.ones(n_onehot)
            x = latin_hypercube(init_pool_size, n_onehot)
            x = from_unit_cube(x, lb, ub)
            pool = onehot2ordinal(x, cat_dims)
            if hasattr(pool, 'numpy'):
                pool = pool.numpy()
            pool = pool.astype(int)

        # 4. Filter by CDR constraints
        if self.cdr_constraints:
            mask = np.array([check_cdr_constraints(p) for p in pool])
            pool = pool[mask]
            print(f"[init] CDR filtering: {mask.sum()}/{len(mask)} passed.")

        # 4b. Deduplicate (DSL sampling can produce duplicate candidates)
        pool = np.unique(pool, axis=0)

        if len(pool) < n_init:
            print(f"[init] Pool too small after CDR ({len(pool)} < {n_init}); using random init.")
            return None

        # 5. Score with bias DSL (if set)
        if bias_dsl is not None:
            scores = np.array([bias_dsl([int(v) for v in seq]) for seq in pool])
        else:
            scores = np.zeros(len(pool))

        # 6. Tiebreak: add tiny random noise to break exact ties
        noise = np.random.uniform(-1e-9, 1e-9, size=len(scores))
        top_idx = np.argsort(scores + noise)[-n_init:]

        initial_points = pool[top_idx]
        n_unique_scores = len(set(scores.round(6)))
        print(f"[init] Selected top-{n_init} from pool of {len(pool)}. "
              f"Score range: [{scores[top_idx].min():.3f}, {scores[top_idx].max():.3f}], "
              f"unique scores: {n_unique_scores}")
        return initial_points

    def run(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # --- Build LDM Orchestrator (if enabled via 'ldm' config section) ---
        orchestrator = self._build_orchestrator()

        kwargs = {
            'length_max_discrete': self.config['seq_len'],
            'device': self.config['device'],
            'seed': self.seed,
            'search_strategy': self.search_strat,
            'BERT_model_path': self.config.get('BERT_model_path', 'Rostlab/prot_bert_bfd'),
            'BERT_tokeniser_path': self.config.get('BERT_tokenizer_path', 'Rostlab/prot_bert_bfd'),
            'BERT_batchsize': self.config.get('BERT_batchsize', 128),
        }
        # Inject Orchestrator + antigen context for CASMOPOLITANCat.
        _llm_cfg = self.config.get('llm', {})
        if _llm_cfg.get('llm_loop_enabled', True):
            kwargs['orchestrator'] = orchestrator
        else:
            kwargs['orchestrator'] = None
            print("[loop] LLM loop disabled; plain BO (no orchestrator).")
        kwargs['antigen_id'] = self.config['bbox']['antigen']
        kwargs['antigen_seed'] = self.seed
        kwargs['antigen_context'] = self.antigen_context or {}
        # LLM config keys read by localbo_cat.py DSLConfig construction
        kwargs['acq_search_budget'] = _llm_cfg.get('acq_search_budget', 600)
        kwargs['acq_max_rounds'] = _llm_cfg.get('acq_max_rounds', 3)
        kwargs['num_llm_review'] = _llm_cfg.get('num_llm_review', 10)
        kwargs['max_retries'] = _llm_cfg.get('max_retries', 3)
        kwargs['bias_weight'] = _llm_cfg.get('bias_weight', 0.05)
        kwargs['sample_timeout_s'] = _llm_cfg.get('sample_timeout_s', 5.0)
        kwargs['llm_loop_enabled'] = _llm_cfg.get('llm_loop_enabled', True)
        kwargs['strategy'] = _llm_cfg.get('strategy', 'ldm-default')

        # --- LLM-guided initialization (if orchestrator enabled) ---
        if not self.config.get('resume'):
            self.llm_initial_points = self._build_llm_initial_points(orchestrator)

        if self.config['resume']:
            optim = self.load()
        else:
            optim = None

        if not optim:
            optim = Optimizer(
                config=self.n_categories, min_cuda=self.config['min_cuda'],
                n_init=self.config['n_init'], use_ard=self.config['ard'],
                acq=self.config['acq'],
                cdr_constraints=self.cdr_constraints,
                normalise=self.config['normalise'],
                kernel_type=self.config['kernel_type'],
                noise_variance=float(self.config['noise_variance']),
                alphabet_size=self.nb_aas,
                table_of_candidates=self.table_of_antibodies_as_inds,
                table_of_candidate_embeddings=self.table_of_embeddings,
                embedding_from_array_dict=self.embedding_from_array_dict,
                initial_points=self.llm_initial_points,
                **kwargs
            )

            if self.config.get("pre_evals") is not None:
                pre_eval_x, pre_eval_y = get_x_y_from_csv(self.config.get("pre_evals"))
                optim.suggest(len(pre_eval_x))  # exhaust init random suggestions
                optim.batch_size = self.config['batch_size']
                optim.casmopolitan.batch_size = optim.batch_size
                optim.casmopolitan.n_init = max([optim.casmopolitan.n_init, optim.batch_size])
                optim.observe(x=pre_eval_x, y=pre_eval_y)
                print(f"Observed {len(pre_eval_y)} already evaluated points")

        if self.antigen_context is not None:
            prompt_dir = os.path.join(self.path, 'llm_prompt_context')
            prompt_path = save_llm_context_snapshot(
                out_dir=prompt_dir,
                antigen_context=self.antigen_context,
                top_k=int(self.config.get('llm_prompt_top_k', 10)),
            )
            print(f"Saved initial LLM prompt context in {prompt_path}")

        # check if there are points that have been suggested and evaluated since the last antbo call
        if isinstance(self.f_obj.fbox, TableFilling) and os.path.exists(self.f_obj.fbox.path_to_eval_csv):
            table_of_results = pd.read_csv(self.f_obj.fbox.path_to_eval_csv, index_col=None)
            if np.all(table_of_results["Validate (0/1)"].values):
                print(f"Get already evaluated points from table {self.f_obj.fbox.path_to_eval_csv}")
                y = torch.tensor(table_of_results["Validate (0/1)"].values)
                x_seqs = table_of_results.Antibody.values
                # convert strings to array
                x_seqs_ind = np.array(
                    [np.array([self.f_obj.fbox.AA_to_idx[char] for char in x_seq]) for x_seq in x_seqs]
                )
                if optim.batch_size is None:
                    optim.batch_size = len(x_seqs)
                    optim.casmopolitan.batch_size = len(x_seqs)
                    optim.casmopolitan.n_init = max([optim.casmopolitan.n_init, optim.batch_size])
                    optim.restart()
                optim.observe(x=x_seqs_ind, y=y)
                self.results(optim=optim, x=x_seqs_ind, itern=self.start_itern, rtime=0)
                self.start_itern += 1
                self.save(optim=optim)
                self.f_obj.fbox.make_copy_eval_table()

        for itern in range(self.start_itern, self.config['max_iters']):
            start = time.time()
            x_next = optim.suggest(n_suggestions=self.config['batch_size'])
            if self.custom_initial_dataset and len(optim.casmopolitan.fx) < self.config['n_init']:
                # observe the custom initial points instead of the suggested ones
                n_random = min(x_next.shape[0], self.config['n_init'] - len(optim.casmopolitan.fx))
                x_next[:n_random] = self.custom_initial_dataset.get_index_encoded_x()[
                                    len(optim.casmopolitan.fx):len(optim.casmopolitan.fx) + n_random]
            y_next = self.f_obj.compute(x=x_next)
            optim.observe(x=x_next, y=y_next)
            end = time.time()
            self.results(optim=optim, x=x_next, itern=itern, rtime=end - start)

            seq_str = ''.join(['ACDEFGHIKLMNPQRSTVWY'[int(x)] for x in x_next[0]])
            y_val = y_next[0].item() if hasattr(y_next[0], 'item') else float(y_next[0])
            cumul_y = np.array(optim.casmopolitan.fx).flatten()
            best_val = float(cumul_y.min()) if len(cumul_y) > 0 else y_val

            parts = [f"Iter {itern + 1}/{self.config['max_iters']} ({end - start:.1f}s)",
                     f"seq={seq_str} y={y_val:.4f} best={best_val:.4f}"]

            decision = getattr(optim.casmopolitan, '_last_decision', None)
            if decision is not None:
                if decision.search_updated and decision.search_dsl is not None:
                    parts.append(f"LDM updated TR: {decision.search_dsl!r}")
                if decision.bias_updated and decision.bias_dsl is not None:
                    parts.append(f"LDM updated bias: {decision.bias_dsl!r}")

            self.log(" | ".join(parts))
            self.save(optim=optim)
            if self.antigen_context is not None:
                prompt_every = int(self.config.get('llm_prompt_save_every', 0))
                should_save_prompt = prompt_every > 0 and ((itern + 1) % prompt_every == 0)
                if should_save_prompt:
                    prompt_dir = os.path.join(self.path, 'llm_prompt_context')
                    prompt_path = save_llm_context_snapshot(
                        out_dir=prompt_dir,
                        antigen_context=self.antigen_context,
                        optim=optim,
                        f_obj=self.f_obj,
                        itern=itern + 1,
                        top_k=int(self.config.get('llm_prompt_top_k', 10)),
                    )
                    print(f"Saved LLM prompt context in {prompt_path}")

    def log(self, message: str, end: Optional[str] = None) -> None:
        header = f"BOExp - {self.config['bbox']['antigen']} - {self.config['kernel_type']} - seed {self.seed}"
        log(message=message, header=header, end=end)

    @staticmethod
    def get_table_of_antibodies(tabular_search_csv: str, normalize_embeddings: bool = True) \
            -> tuple[np.ndarray, Optional[np.ndarray], dict[str, np.ndarray]]:
        """ Return array of antigens where each row corresponds to an antibody given by the index of its AA

        Args:
            - tabular_search_csv: path to the csv file containing the AAs (and optionally vector representations)
            - normalize_embeddings: whether to min-max normalize the embeddings

        Returns:
            - aas_as_inds: array of aas (each entry is an array of AA indices)
            - embeddings: array of shape (n_antibodies, embedding size)
            - embedding_from_aas_as_inds_dict: dictionary mapping the antibody arrays to their embeddings
        """
        data = pd.read_csv(tabular_search_csv, index_col=None)
        if data.shape[-1] == 1:
            aas = data.values.flatten()
            embeddings = None
        else:
            assert np.all(data.columns[1:] == [f"d{i}" for i in range(1, data.shape[1])]), data.columns[1:]
            aas = data.values[:, 0]
            embeddings = data.values[:, 1:].astype(float)
            if normalize_embeddings:
                min_embeddings = embeddings.min(0)
                max_embeddings = embeddings.max(0)
                embeddings = (embeddings - min_embeddings) / (max_embeddings - min_embeddings)
        arr = np.array([list(c for c in x) for x in aas])
        aas_as_inds = BaseTool().convert_array_aas_to_idx(arr)
        if embeddings is not None:
            embedding_from_aas_as_inds_dict = {
                str(aas_as_inds[i].astype(int)): embeddings[i] for i in tqdm(range(len(aas_as_inds)))
            }
        else:
            embedding_from_aas_as_inds_dict = None
        return aas_as_inds, embeddings, embedding_from_aas_as_inds_dict


if __name__ == '__main__':
    from bo.utils import get_config

    parser = argparse.ArgumentParser(add_help=True, description='Antigen-CDR3 binding prediction using BO')
    parser.add_argument('--antigens_file', type=str, default=None,
                        help='List of Antigen to perform BO. Required if not in config YAML.')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Root path for results. Required if not in config YAML.')
    parser.add_argument('--seed', type=int, default=42, help='initial seed setting')
    parser.add_argument('--n_trials', type=int, default=3, help='number of random trials')
    parser.add_argument('--resume', type=bool, default=False, help='flag to resume training')
    parser.add_argument('--resume_trial', type=int, default=0, help='resume trial for training')
    parser.add_argument('--cdr_constraints', type=bool, default=True, help='constraint local search')
    parser.add_argument('--config', type=str, default='./bo/config.yaml',
                        help='Configuration File')
    args = parser.parse_args()
    config_ = get_config(os.path.abspath(args.config))
    config_['resume'] = args.resume

    # Resolve antigens source: CLI arg takes precedence, then YAML fallback,
    # then explicit error. save_path and antigens_file are operational
    # parameters that no longer live in the default config.yaml.
    if args.antigens_file is not None:
        antigens_path = args.antigens_file
    elif 'antigens_file' in config_:
        antigens_path = config_['antigens_file']
    else:
        parser.error('--antigens_file required (or set "antigens_file:" in config YAML).')

    if args.save_path is not None:
        config_['save_path'] = args.save_path
    elif 'save_path' not in config_:
        parser.error('--save-path required (or set "save_path:" in config YAML).')

    with open(antigens_path) as file:
        antigens = file.readlines()
        antigens = [antigen.rstrip() for antigen in antigens]

    print(f'Iterating Over All Antigens In File {antigens_path} \n {antigens}')
    # antigens = ['1ADQ_A', '1FBI_X', '1HOD_C', '1NSN_S', '1OB1_C', '1WEJ_F',
    # '2YPV_A', '3RAJ_A', '3VRL_C', '2DD8_S', '1S78_B', '2JEL_P']

    for antigen in antigens:
        start_antigen = time.time()
        seeds = list(range(args.seed, args.seed + args.n_trials))
        t = args.resume_trial
        while t < args.n_trials:
            print(f"Starting Trial {t + 1} for antigen {antigen}")
            config_['bbox']['antigen'] = antigen

            boexp = BOExperiments(config_, args.cdr_constraints, seeds[t])

            try:
                boexp.run()
            except FileNotFoundError as e:
                print(e.args)
                continue

            del boexp
            torch.cuda.empty_cache()
            end_antigen = time.time()
            print(f"Time taken for antigen {antigen} trial {t + 1} = {end_antigen - start_antigen}")
            t += 1
        args.resume_trial = 0
    print('BO finished')
