"""bo/ldm/acquisition/session.py — interactive LLM acquisition session.

Multi-round protocol with batch support:
  Round 0: Execute atoms → pool results → top-k review → LLM TAKE/SEARCH
  Round 1+: LLM reviews results, outputs TAKE(ids) or SEARCH(new atoms)
  Batch filled → return. Budget/rounds exhausted → force-fill from top.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from bo.ldm.acquisition.parallel_search import execute_atoms
from bo.ldm.config import DSLConfig
from bo.ldm.dsl.search_space import SearchSpaceAtom
from bo.ldm.llm.function import ReviewFunction


@dataclass
class RoundRecord:
    round_idx: int
    atoms_repr: str
    n_evaluated: int
    budget_used: int
    review_topk: list[dict] = field(default_factory=list)
    taken_ids: list[int] = field(default_factory=list)
    llm_action: str = ""
    llm_rationale: str = ""


class AcquisitionSession:
    """Interactive acquisition session with LLM-guided search and review."""

    def __init__(self, llm_client, config: DSLConfig, acq_name: str = "ei"):
        self.llm = llm_client
        self.config = config
        self.acq_name = acq_name
        self.rounds: list[RoundRecord] = []
        self.review_prompts: list[str] = []
        self.llm_review_responses: list[str] = []

    def run(
        self,
        search_dsl: SearchSpaceAtom,
        bias_dsl,
        gp,
        f_acq,
        batch_size: int,
        cat_config: np.ndarray,
        cdr_constraints: bool,
        device: torch.device,
        status_data: dict | None = None,
    ) -> np.ndarray:
        rng = np.random.default_rng()
        budget_total = self.config.acq_search_budget
        budget_used = 0
        max_rounds = self.config.acq_max_rounds
        num_review = self.config.num_llm_review
        bias_weight = self.config.bias_weight
        combined_key = f"bias+{self.acq_name}"

        taken_ids: list[int] = []
        evaluated_pool: list[dict] = []
        current_search_dsl = search_dsl

        def _force_fill() -> list[int]:
            untaken = [r for r in evaluated_pool if r["id"] not in taken_ids]
            untaken.sort(key=lambda r: r[combined_key], reverse=True)
            needed = batch_size - len(taken_ids)
            ids = [r["id"] for r in untaken[:needed]]
            taken_ids.extend(ids)
            return ids

        for round_idx in range(max_rounds):
            remaining_slots = batch_size - len(taken_ids)
            if remaining_slots <= 0:
                break

            # Budget exhausted → force fill
            if budget_used >= budget_total and evaluated_pool:
                force_ids = _force_fill()
                self.rounds.append(RoundRecord(
                    round_idx=round_idx,
                    atoms_repr="(budget exhausted, force fill)",
                    n_evaluated=0,
                    budget_used=budget_used,
                    taken_ids=force_ids,
                    llm_action="take (budget exhausted)",
                ))
                break

            # Execute atoms
            results = execute_atoms(
                search_dsl=current_search_dsl,
                gp=gp, f_acq=f_acq, bias_dsl=bias_dsl,
                bias_weight=bias_weight,
                config=cat_config,
                cdr_constraints=cdr_constraints,
                rng=rng, timeout_s=self.config.sample_timeout_s,
                device=device, acq_name=self.acq_name,
            )

            start_id = len(evaluated_pool)
            for i, r in enumerate(results):
                r["id"] = start_id + i
                r["seq_str"] = "".join(
                    "ACDEFGHIKLMNPQRSTVWY"[int(x)] for x in r["seq"]
                )
            evaluated_pool.extend(results)
            proposed_budget = (
                current_search_dsl.budget
                if hasattr(current_search_dsl, "budget") else len(results)
            )
            budget_used += proposed_budget

            # Rank by bias+acq
            untaken = [
                r for r in evaluated_pool if r["id"] not in taken_ids
            ]
            untaken.sort(key=lambda r: r[combined_key], reverse=True)
            topk = untaken[:num_review]

            # Build review text
            review_text = self._build_review_text(
                topk, taken_ids, batch_size, budget_used, budget_total,
                round_idx, max_rounds, status_data,
            )

            # LLM review via ReviewFunction (handles retries internally)
            func = ReviewFunction(
                self.llm, self.config, self.acq_name,
            )
            remaining_budget = budget_total - budget_used
            remaining_slots = batch_size - len(taken_ids)
            action, rationale, payload = func(
                review_text=review_text,
                remaining_budget=remaining_budget,
                num_review=len(topk),
                remaining_slots=remaining_slots,
            )

            # Store for debugging
            if func.previous_attempts:
                self.review_prompts.extend(
                    [p for p, _ in func.previous_attempts] if False else
                    [func.construct_prompt(review_text=review_text,
                                           remaining_budget=remaining_budget)]
                )
            self.llm_review_responses.extend(
                [r for r, _ in func.previous_attempts] if func.previous_attempts
                else []
            )

            round_record = RoundRecord(
                round_idx=round_idx,
                atoms_repr=repr(current_search_dsl),
                n_evaluated=len(results),
                budget_used=budget_used,
                review_topk=topk[:num_review],
                llm_action=action,
                llm_rationale=rationale or "",
            )

            if action == "take":
                pos_ids = payload
                pool_ids = [topk[i]["id"] for i in pos_ids if i < len(topk)]
                for tid in pool_ids:
                    if tid not in taken_ids:
                        taken_ids.append(tid)
                round_record.taken_ids = pos_ids
                self.rounds.append(round_record)
                if len(taken_ids) >= batch_size:
                    break

            elif action == "search":
                round_record.taken_ids = []
                current_search_dsl = payload
                self.rounds.append(round_record)

        # Final safety fill
        if len(taken_ids) < batch_size:
            force_ids = _force_fill()
            self.rounds.append(RoundRecord(
                round_idx=len(self.rounds),
                atoms_repr="(final force fill)",
                n_evaluated=0,
                budget_used=budget_used,
                taken_ids=force_ids,
                llm_action="take (final force fill)",
            ))

        # Build output
        selected = []
        for tid in taken_ids[:batch_size]:
            if tid < len(evaluated_pool):
                selected.append(evaluated_pool[tid]["seq"])
        while len(selected) < batch_size and evaluated_pool:
            selected.append(evaluated_pool[taken_ids[0]]["seq"])

        return np.array(selected[:batch_size], dtype=int)

    def _build_review_text(
        self, topk, taken_ids, batch_size, budget_used, budget_total,
        round_idx, max_rounds, status_data,
    ) -> str:
        lines = []
        lines.append(f"# Acquisition Review — Round {round_idx + 1}/{max_rounds}")
        if status_data:
            lines.append(f"BO iteration: {status_data.get('iteration', '?')}")
            lines.append(f"Best value: {status_data.get('best_value', '?')}")
            strategy = status_data.get('strategy', 'ldm-default')
            if strategy == 'antbo-mock':
                lines.append("## Strategy: AntBO mimicry — always TAKE ids=[0]")
        lines.append("")
        lines.append(f"## Top candidates (by {self.acq_name}+bias)")
        lines.append(
            f"{'id':>3}  {'seq':<11}  {self.acq_name:>6}  {'mu':>7}  "
            f"{'sigma':>6}  {'bias':>6}  {self.acq_name}+bias"
        )
        for idx, r in enumerate(topk):
            lines.append(
                f"{idx:>3}  {r['seq_str']:<11}  "
                f"{r[self.acq_name]:>6.3f}  {r['mu']:>7.1f}  "
                f"{r['sigma']:>6.1f}  {r['bias']:>6.2f}  "
                f"{r[f'bias+{self.acq_name}']:>6.3f}"
            )
        lines.append("")
        lines.append("## Status")
        lines.append(f"- Taken: {len(taken_ids)} candidate(s)")
        lines.append(f"- Remaining slots: {batch_size - len(taken_ids)}")
        lines.append(f"- Budget used: {budget_used}/{budget_total}")
        lines.append(f"- Remaining budget: {budget_total - budget_used}")
        lines.append("")
        lines.append("## Actions")
        lines.append('  Take one or more:  {"action": "take", "ids": [0, 2]}')
        lines.append(
            '  Search more:       {"action": "search", '
            '"update_trust_region": "Or(LocalSearch(...))"}'
        )
        return "\n".join(lines)
