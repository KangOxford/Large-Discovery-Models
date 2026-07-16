"""Reservoir acquisition session.

Standalone 5-strategy LDM reservoir execution. The old ``bo.ldm`` code path is
not touched. One LLM plan should produce all K strategies before this class is
called; this session executes the K strategy pools as one parallel/batched step.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence
import time

import numpy as np
import torch

from bo.ldm.acquisition.parallel_search import _eval_batch, _sample_neighbour
from bo.ldm.dsl.alphabet import SEQ_LEN, hamming, idx_to_aa
from bo.ldm.dsl.search_space import (
    LatinHyperCubeSampling,
    LocalSearch,
    NeighborSampling,
    Or,
    SearchSpaceAtom,
)
from .config import ReservoirLDMConfig


def _seq_to_key(seq: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(v) for v in seq)


def _seq_to_str(seq: Sequence[int]) -> str:
    return "".join(idx_to_aa(int(v)) for v in seq)


def _center_to_str(center: Optional[Sequence[int] | str]) -> Optional[str]:
    if center is None:
        return None
    if isinstance(center, str):
        return center
    values = [int(v) for v in center]
    if len(values) != SEQ_LEN:
        raise ValueError(f"fallback_center must have length {SEQ_LEN}, got {len(values)}")
    return _seq_to_str(values)


@dataclass
class StrategyResult:
    strategy_idx: int
    atom_repr: str
    n_evaluated: int
    best: Optional[dict[str, Any]]
    error: Optional[str] = None


class ReservoirAcquisitionSession:
    """Run a parallel/batched 5-strategy LLM-reservoir acquisition step."""

    def __init__(self, config: Optional[ReservoirLDMConfig] = None, acq_name: str = "ei") -> None:
        self.config = config or ReservoirLDMConfig()
        self.acq_name = acq_name
        self.rng = np.random.default_rng(self.config.rng_seed)
        self.strategy_results: list[StrategyResult] = []
        self.last_record: dict[str, Any] = {}

    def run(
        self,
        strategies: SearchSpaceAtom | Sequence[SearchSpaceAtom],
        bias_dsl,
        gp,
        f_acq,
        batch_size: int,
        cat_config: np.ndarray,
        cdr_constraints: bool,
        device: torch.device,
        fallback_center: Optional[Sequence[int] | str] = None,
    ) -> np.ndarray:
        """Return selected categorical sequence array of shape (batch, 11)."""
        atoms = self._prepare_strategies(strategies, fallback_center)
        capped_atoms, records_by_strategy, errors = self._execute_strategy_pools_parallel(
            atoms=atoms,
            bias_dsl=bias_dsl,
            gp=gp,
            f_acq=f_acq,
            cat_config=cat_config,
            cdr_constraints=cdr_constraints,
            device=device,
        )

        self.strategy_results = []
        all_records: list[dict[str, Any]] = []
        representatives: list[dict[str, Any]] = []

        for idx, (atom, records) in enumerate(zip(capped_atoms, records_by_strategy)):
            records = self._unique_records(records)
            all_records.extend(records)
            best = self._best_record(records, self.config.pool_score)
            if best is not None:
                best = dict(best)
                best["strategy_idx"] = idx
                best["strategy_atom"] = repr(atom)
                representatives.append(best)
            self.strategy_results.append(StrategyResult(idx, repr(atom), len(records), best, errors[idx]))

        representatives = self._unique_records(representatives)
        if not representatives:
            representatives = self._fallback_global_candidates(gp, f_acq, bias_dsl, cat_config, cdr_constraints, device)
        if not representatives:
            raise RuntimeError("Reservoir LDM produced no valid candidate.")

        selected_ids, probs = self._select(representatives, batch_size)
        selected = [representatives[i] for i in selected_ids]

        if len(selected) < batch_size:
            selected_keys = {_seq_to_key(r["seq"]) for r in selected}
            extras = [
                r for r in self._rank_records(all_records, self.config.selection_score)
                if _seq_to_key(r["seq"]) not in selected_keys
            ]
            for rec in extras:
                selected.append(rec)
                selected_keys.add(_seq_to_key(rec["seq"]))
                if len(selected) >= batch_size:
                    break

        self.last_record = {
            "execution_mode": "parallel_batched",
            "llm_plan_calls_expected_per_iteration": 1,
            "n_strategies_requested": self.config.n_strategies,
            "n_strategies_executed": len(capped_atoms),
            "selection_mode": self.config.selection_mode,
            "pool_score": self.config.pool_score,
            "selection_score": self.config.selection_score,
            "probabilities": probs,
            "selected_ids": selected_ids,
            "representatives": [self._public_record(r) for r in representatives],
            "strategy_results": [s.__dict__ for s in self.strategy_results],
        }
        return np.array([r["seq"] for r in selected[:batch_size]], dtype=int)

    def _prepare_strategies(
        self,
        strategies: SearchSpaceAtom | Sequence[SearchSpaceAtom],
        fallback_center: Optional[Sequence[int] | str],
    ) -> list[SearchSpaceAtom]:
        if isinstance(strategies, Or):
            atoms = strategies.children
        elif isinstance(strategies, SearchSpaceAtom):
            atoms = [strategies]
        else:
            atoms = []
            for item in strategies:
                if isinstance(item, Or):
                    atoms.extend(item.children)
                elif isinstance(item, SearchSpaceAtom):
                    atoms.append(item)
                else:
                    raise TypeError(f"Expected SearchSpaceAtom, got {type(item).__name__}")

        atoms = atoms[: self.config.n_strategies]
        center = _center_to_str(fallback_center)
        while len(atoms) < self.config.n_strategies:
            if center is None:
                atoms.append(LatinHyperCubeSampling(num=self.config.per_strategy_budget))
            else:
                radius = self.config.fallback_radius_start + (len(atoms) % 3)
                atoms.append(NeighborSampling(center, radius=radius, mut_pr=self.config.fallback_mut_pr, budget=self.config.per_strategy_budget))
        return atoms

    def _execute_strategy_pools_parallel(
        self,
        atoms: list[SearchSpaceAtom],
        bias_dsl,
        gp,
        f_acq,
        cat_config: np.ndarray,
        cdr_constraints: bool,
        device: torch.device,
    ) -> tuple[list[SearchSpaceAtom], list[list[dict[str, Any]]], list[Optional[str]]]:
        capped_atoms = [self._cap_budget(atom) for atom in atoms]
        records_by_strategy: list[list[dict[str, Any]]] = [[] for _ in capped_atoms]
        errors: list[Optional[str]] = [None for _ in capped_atoms]

        sampling_items = [(idx, atom) for idx, atom in enumerate(capped_atoms) if isinstance(atom, (NeighborSampling, LatinHyperCubeSampling))]
        local_items = [(idx, atom) for idx, atom in enumerate(capped_atoms) if isinstance(atom, LocalSearch)]

        if sampling_items:
            self._execute_sampling_parallel(sampling_items, records_by_strategy, errors, gp, f_acq, bias_dsl, device)

        if local_items:
            try:
                local_records = self._parallel_local_search_by_strategy(
                    local_items, gp=gp, f_acq=f_acq, bias_dsl=bias_dsl,
                    config=cat_config, cdr_constraints=cdr_constraints, device=device,
                )
                for idx, records in local_records.items():
                    records_by_strategy[idx].extend(records)
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                for idx, _ in local_items:
                    errors[idx] = msg

        for idx, atom in enumerate(capped_atoms):
            if not isinstance(atom, (NeighborSampling, LatinHyperCubeSampling, LocalSearch)):
                errors[idx] = f"Unsupported atom type in reservoir session: {type(atom).__name__}"
        return capped_atoms, records_by_strategy, errors

    def _execute_sampling_parallel(
        self,
        sampling_items: list[tuple[int, NeighborSampling | LatinHyperCubeSampling]],
        records_by_strategy: list[list[dict[str, Any]]],
        errors: list[Optional[str]],
        gp,
        f_acq,
        bias_dsl,
        device: torch.device,
    ) -> None:
        def sample_item(item: tuple[int, NeighborSampling | LatinHyperCubeSampling]):
            idx, atom = item
            seed = int(self.rng.integers(0, 2**32 - 1))
            samples = atom.sample(n=atom.budget, rng=np.random.default_rng(seed), timeout_s=self.config.sample_timeout_s)
            return idx, atom, samples

        sampled: list[tuple[int, SearchSpaceAtom, list[list[int]]]] = []
        max_workers = max(1, min(len(sampling_items), self.config.n_strategies))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(sample_item, item): item[0] for item in sampling_items}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    sampled.append(future.result())
                except Exception as exc:
                    errors[idx] = f"{type(exc).__name__}: {exc}"

        all_seqs: list[list[int]] = []
        owners: list[tuple[int, str]] = []
        for idx, atom, samples in sampled:
            for seq in samples:
                all_seqs.append(seq)
                owners.append((idx, repr(atom)))

        scored = _eval_batch(all_seqs, gp=gp, f_acq=f_acq, bias_dsl=bias_dsl, bias_weight=self.config.bias_weight, device=device, acq_name=self.acq_name)
        for rec, (idx, atom_repr) in zip(scored, owners):
            rec = dict(rec)
            rec["strategy_idx"] = idx
            rec["source"] = f"ReservoirSampling(strategy={idx}, atom={atom_repr})"
            records_by_strategy[idx].append(rec)

    def _parallel_local_search_by_strategy(
        self,
        local_items: list[tuple[int, LocalSearch]],
        gp,
        f_acq,
        bias_dsl,
        config: np.ndarray,
        cdr_constraints: bool,
        device: torch.device,
    ) -> dict[int, list[dict[str, Any]]]:
        n_categories = list(config)
        deadline = time.time() + self.config.sample_timeout_s
        workers: list[dict[str, Any]] = []
        for local_idx, (strategy_idx, atom) in enumerate(local_items):
            center = atom.center_idx
            for restart in range(atom.restart):
                workers.append({
                    "local_idx": local_idx,
                    "strategy_idx": strategy_idx,
                    "restart": restart,
                    "atom": atom,
                    "center": center,
                    "radius": atom.radius,
                    "fixed_positions": atom.fixed_positions,
                    "max_steps": atom.steps,
                    "step": 0,
                    "x": list(center),
                    "best_combined": None,
                    "tol": 100,
                    "active": True,
                    "trajectory": set([tuple(center)]),
                })

        out: dict[int, list[dict[str, Any]]] = {idx: [] for idx, _ in local_items}
        if not workers:
            return out

        center_results = _eval_batch([w["x"] for w in workers], gp=gp, f_acq=f_acq, bias_dsl=bias_dsl, bias_weight=self.config.bias_weight, device=device, acq_name=self.acq_name)
        for i, worker in enumerate(workers):
            rec = dict(center_results[i])
            rec["strategy_idx"] = worker["strategy_idx"]
            rec["source"] = f"ReservoirLocalSearch(strategy={worker['strategy_idx']}, center={worker['atom'].center}, restart={worker['restart']}, step=0)"
            out[worker["strategy_idx"]].append(rec)
            worker["best_combined"] = rec[f"bias+{self.acq_name}"]

        max_steps = max(w["max_steps"] for w in workers)
        for step in range(1, max_steps + 1):
            if time.time() > deadline:
                break
            for worker in workers:
                if worker["active"] and worker["step"] >= worker["max_steps"]:
                    worker["active"] = False

            neighbours: list[list[int]] = []
            worker_ids: list[int] = []
            for i, worker in enumerate(workers):
                if not worker["active"]:
                    continue
                found = False
                attempts = 0
                while attempts < worker["tol"]:
                    attempts += 1
                    nb = _sample_neighbour(worker["x"], n_categories)
                    key = tuple(nb)
                    if any(nb[p] != worker["center"][p] for p in worker["fixed_positions"]):
                        continue
                    if worker["radius"] is not None and hamming(nb, worker["center"]) > worker["radius"]:
                        continue
                    if key in worker["trajectory"]:
                        continue
                    if cdr_constraints:
                        from bo.localbo_utils import check_cdr_constraints
                        if not check_cdr_constraints(nb):
                            continue
                    neighbours.append(nb)
                    worker_ids.append(i)
                    found = True
                    break
                if not found:
                    worker["active"] = False
            if not neighbours:
                break

            results = _eval_batch(neighbours, gp=gp, f_acq=f_acq, bias_dsl=bias_dsl, bias_weight=self.config.bias_weight, device=device, acq_name=self.acq_name)
            for j, rec in enumerate(results):
                worker = workers[worker_ids[j]]
                rec = dict(rec)
                rec["strategy_idx"] = worker["strategy_idx"]
                rec["source"] = f"ReservoirLocalSearch(strategy={worker['strategy_idx']}, center={worker['atom'].center}, restart={worker['restart']}, step={step})"
                out[worker["strategy_idx"]].append(rec)

            for j, worker_id in enumerate(worker_ids):
                worker = workers[worker_id]
                worker["trajectory"].add(tuple(neighbours[j]))
                worker["step"] += 1
                combined = results[j][f"bias+{self.acq_name}"]
                if combined > worker["best_combined"]:
                    worker["x"] = list(neighbours[j])
                    worker["best_combined"] = combined
        return out

    def _cap_budget(self, atom: SearchSpaceAtom) -> SearchSpaceAtom:
        budget = max(1, int(self.config.per_strategy_budget))
        if atom.budget <= budget:
            return atom
        if isinstance(atom, LatinHyperCubeSampling):
            return LatinHyperCubeSampling(num=budget)
        if isinstance(atom, NeighborSampling):
            return NeighborSampling(atom.center, fixed=atom.fixed, radius=atom.radius, mut_pr=atom.mut_pr, budget=budget)
        if isinstance(atom, LocalSearch):
            restart = min(atom.restart, budget)
            steps = max(1, budget // restart - 1)
            return LocalSearch(atom.center, fixed=atom.fixed, radius=atom.radius, restart=restart, steps=steps)
        return atom

    def _score_key(self, score_name: str) -> str:
        if score_name == "acq":
            return self.acq_name
        if score_name == "combined":
            return f"bias+{self.acq_name}"
        if score_name == "bias":
            return "bias"
        raise ValueError("score must be one of: acq, combined, bias")

    def _best_record(self, records: list[dict[str, Any]], score_name: str) -> Optional[dict[str, Any]]:
        if not records:
            return None
        key = self._score_key(score_name)
        return max(records, key=lambda r: float(r.get(key, -np.inf)))

    def _rank_records(self, records: Iterable[dict[str, Any]], score_name: str) -> list[dict[str, Any]]:
        key = self._score_key(score_name)
        unique = self._unique_records(records)
        return sorted(unique, key=lambda r: float(r.get(key, -np.inf)), reverse=True)

    def _unique_records(self, records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        key = self._score_key("combined")
        best_by_seq: dict[tuple[int, ...], dict[str, Any]] = {}
        for rec in records:
            if "seq" not in rec:
                continue
            seq_key = _seq_to_key(rec["seq"])
            old = best_by_seq.get(seq_key)
            if old is None or float(rec.get(key, -np.inf)) > float(old.get(key, -np.inf)):
                best_by_seq[seq_key] = rec
        return list(best_by_seq.values())

    def _select(self, representatives: list[dict[str, Any]], batch_size: int) -> tuple[list[int], list[float]]:
        score_key = self._score_key(self.config.selection_score)
        scores = np.array([float(r.get(score_key, -np.inf)) for r in representatives], dtype=float)
        finite = np.isfinite(scores)
        if not finite.any():
            probs = np.ones(len(representatives), dtype=float) / len(representatives)
        elif self.config.selection_mode == "argmax":
            probs = np.zeros(len(representatives), dtype=float)
            probs[int(np.nanargmax(scores))] = 1.0
        elif self.config.selection_mode == "softmax":
            safe_scores = np.where(finite, scores, scores[finite].min() - 1.0)
            eta = float(self.config.softmax_eta)
            if eta <= 0:
                probs = np.ones(len(representatives), dtype=float) / len(representatives)
            else:
                shifted = eta * (safe_scores - safe_scores.max())
                exp_scores = np.exp(shifted)
                probs = exp_scores / exp_scores.sum()
        else:
            raise ValueError("selection_mode must be 'softmax' or 'argmax'")

        k = min(int(batch_size), len(representatives))
        if self.config.selection_mode == "argmax":
            ids = list(np.argsort(scores)[::-1][:k])
        else:
            ids = list(self.rng.choice(len(representatives), size=k, replace=False, p=probs))
        return [int(i) for i in ids], [float(p) for p in probs]

    def _fallback_global_candidates(self, gp, f_acq, bias_dsl, cat_config: np.ndarray, cdr_constraints: bool, device: torch.device) -> list[dict[str, Any]]:
        atom = LatinHyperCubeSampling(num=max(1, self.config.per_strategy_budget))
        _, records_by_strategy, _ = self._execute_strategy_pools_parallel([atom], bias_dsl, gp, f_acq, cat_config, cdr_constraints, device)
        records = records_by_strategy[0] if records_by_strategy else []
        return self._rank_records(records, self.config.selection_score)

    def _public_record(self, rec: dict[str, Any]) -> dict[str, Any]:
        out = dict(rec)
        if "seq" in out:
            out["seq_str"] = _seq_to_str(out["seq"])
        return out
