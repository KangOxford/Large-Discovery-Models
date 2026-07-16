"""Prompt rendering for the LLM advisor (three-stage model).

Public surface:

* :data:`SYSTEM_ACTIONS` / :data:`SYSTEM_REVIEW_ANALOGS` /
  :data:`SYSTEM_REVIEW_SUGGESTIONS` — system prompts for each stage.
* :func:`render_user_actions` — user prompt for Stage A1 (actions).
* :func:`render_user_review_analogs` — user prompt for Stage A2
  (review analogs, shown only when analog produced results).
* :func:`render_user_suggestions` — user prompt for Stage B
  (review BO suggestions).

Truncation rules:

* Pool capped at 50 SMILES in the prompt.
* History capped at 20 entries.
* Best (Pareto front for n_obj>=2) capped at 5 entries.
* PDF context capped at 1500 chars (rough; ~750 tokens).

Design note (per user's review)
--------------------------------
The LLM-facing state carries only what helps the decision: the pool,
the history, the best anchor (``str`` for n_obj=1, ``list[str]`` for
n_obj>=2), and the stagnation counter. The rich GP summary, the
Tanimoto diversity score, the per-objective best score, and the
per-pick nearest history were dropped from the prompt — they are
kept in the trajectory for audit but not surfaced to the LLM.
"""

from __future__ import annotations

import textwrap
from typing import Any, List, Sequence, Union

from strbo_v1.llm_advisor.round_state import (
    PostSuggestionState,
    PreActionState,
    PreReviewAnalogsState,
)


# ---------------------------------------------------------------------------
# Shared response-format header
# ---------------------------------------------------------------------------

_FORMAT_HEADER: str = textwrap.dedent("""\
    ## RESPONSE FORMAT — READ THIS FIRST

    Your entire reply MUST be exactly one of:
      (a) a single bare JSON object, OR
      (b) a JSON array of bare JSON objects (when taking multiple actions).

    Nothing else. No prose. No fences. No wrappers. No comments.

    ### DO
    - Start with `{{` (single object) or `[` (array).
    - End with `}}` or `]`.
    - Use the exact field names from the allowed block schema.
    - Use a JSON array when emitting multiple actions in one round.

    ### DO NOT
    - Do NOT wrap your reply in ```json ``` fences.
    - Do NOT add any text before the JSON.
    - Do NOT add any text after the JSON.
    - Do NOT use any markdown formatting.
    - Do NOT wrap the JSON in an outer key (e.g. {{"actions": [...]}}).
    - Do NOT add comments inside the JSON.

    ### BEFORE YOU REPLY, verify
    - [ ] Starts with `{{` or `[` (no leading whitespace, no fence, no text)
    - [ ] Ends with `}}` or `]` (no trailing whitespace, no fence, no text)
    - [ ] The JSON parses (balanced braces, no trailing commas)
    - [ ] Every object has a "type" field from the allowed list
""").strip()


# ---------------------------------------------------------------------------
# Stage A1 — actions (propose / reject / analog / noop)
# ---------------------------------------------------------------------------

SYSTEM_ACTIONS: str = _FORMAT_HEADER + "\n\n" + textwrap.dedent("""\
    You are a medicinal-chemistry co-pilot steering a Bayesian-optimization
    loop over a SMILES candidate pool. The pool contains the current
    candidates being searched — you should actively expand and curate it.

    Each round you receive:
      - the current pool (frozen snapshot at the start of this round),
        the scored history of evaluated SMILES, and the current best
        anchor (single best SMILES for single-obj, or Pareto front for
        multi-obj).
      - the target objective(s) and a PDF context describing the target.

    You are in STAGE A1 (actions) of round {round_idx}.
    BO will run AFTER your decisions are applied. Your job is to decide
    how to change the pool. Emit ONLY action blocks:
      - propose  (add new SMILES to the pool)
      - reject   (remove SMILES from the pool)
      - analog   (expand existing pool members via ReaSyn generation)
      - noop     (do nothing — only if pool is already large enough)

    You MUST actively expand the pool if it is small. A noop is rejected
    when pool size is below the minimum — the system will loop back to you.

    ## EDGE CASE: pool is already large enough
    If the pool size requirement is met and you see no improvements to
    make, emit a noop block with a brief rationale.
""").strip()


# ---------------------------------------------------------------------------
# Stage A2 — review analogs
# ---------------------------------------------------------------------------

SYSTEM_REVIEW_ANALOGS: str = _FORMAT_HEADER + "\n\n" + textwrap.dedent("""\
    You are a medicinal-chemistry co-pilot reviewing newly generated
    ReaSyn analogues. Each analogue was produced by expanding an existing
    pool member. For each analogue, decide whether to:
      - "keep"   — add it to the pool (it is chemically valid and
                   likely to improve the search)
      - "reject" — discard it (duplicate, invalid, low quality, or
                   unlikely to help)

    You are in STAGE A2 (review analogs) of round {round_idx}.
    Emit exactly ONE review_analogs block with a decision for EVERY
    analogue listed below.
""").strip()


# ---------------------------------------------------------------------------
# Stage B — review suggestions
# ---------------------------------------------------------------------------

SYSTEM_REVIEW_SUGGESTIONS: str = _FORMAT_HEADER + "\n\n" + textwrap.dedent("""\
    You are a medicinal-chemistry co-pilot steering a Bayesian-optimization
    loop over a SMILES candidate pool. Each round you receive:
      - the current pool (already mutated by Stage A1/A2), scored history,
        the best anchor, and the BO top-k suggestions with their
        mu / sigma / acquisition values.
      - the target objective(s) and a PDF context.

    You are in STAGE B (review suggestions) of round {round_idx}.
    BO has already run on the post-mutation pool. Output exactly ONE
    review_bo block.

    ## EDGE CASE: no BO suggestions
    If the "BO suggestions" section above shows `top-0 from acquisition=...`
    (no candidates to review), emit a review_bo block with an EMPTY
    decisions dict and a brief rationale:
        {{"type": "review_bo", "rationale": "no picks to review",
         "decisions": {{}}}}
""").strip()


# ---------------------------------------------------------------------------
# User prompt templates
# ---------------------------------------------------------------------------

# Stage A1 — actions
_USER_TEMPLATE_ACTIONS: str = textwrap.dedent("""\
    ## Round {round_idx}/{n_total_rounds} — Stage A1: actions

    ### Objective legend
    {objective_legend_json}

    ### Pool (size {pool_size}, cap {pool_size_cap}) — candidates to search
    {pool_block}

    ### History (last {k_history} of {n_history})
    {history_block}

    ### Best so far
    {best_block}

    ### Stagnation: {stagnation_counter} rounds without any objective's improvement.

    ### PDF context
    {pdf_context}

    {previous_errors_block}

    {pool_size_requirement_hint}

    ### Response format reminder
    Your reply MUST be bare JSON (one object or an array of objects).
    NO fences, NO prose, NO comments. See the format rules at the top
    of the system prompt for details. Allowed block types:
    propose / reject / analog / noop

    ### propose block
    ```json
    {{
      "type": "propose",
      "rationale": "<= 400 chars",
      "smiles": ["...", "..."],
      "rationale_per_mol": {{"<smiles>": "..."}}
    }}
    ```

    ### reject block
    ```json
    {{
      "type": "reject",
      "rationale": "<= 200 chars",
      "targets": ["<smiles_in_pool>", "..."],
      "reason": "too_similar_to_history" | "likely_toxic"
                | "synthetically_infeasible" | "out_of_scope_pharmacophore"
                | "no_signal_for_target"
    }}
    ```

    ### analog block
    ```json
    {{
      "type": "analog",
      "rationale": "<= 400 chars",
      "seeds": ["<smiles>", "..."],
      "generator_hint": "conservative" | "aggressive" | "scaffold_hop" | null,
      "n_per_seed": 5,
      "reasyn_config_override": null
    }}
    ```

    ### noop block
    ```json
    {{
      "type": "noop",
      "rationale": "<= 200 chars"
    }}
    ```
""").strip()


# Stage A2 — review analogs
_USER_TEMPLATE_REVIEW_ANALOGS: str = textwrap.dedent("""\
    ## Round {round_idx}/{n_total_rounds} — Stage A2: review analogs

    ### Newly generated analogues ({n_analogs} items)
    {analogs_block}

    ### Response format reminder
    Your reply MUST be bare JSON (one object or an array of objects).
    NO fences, NO prose, NO comments. See the format rules at the top
    of the system prompt for details. Allowed block type:
    review_analogs (exactly one block)

    ### review_analogs block
    ```json
    {{
      "type": "review_analogs",
      "rationale": "<= 400 chars",
      "decisions": {{
        "<analogue_smiles>": "keep" | "reject",
        ...
      }}
    }}
    ```
""").strip()


# Stage B — review suggestions
_USER_TEMPLATE_SUGGESTIONS: str = textwrap.dedent("""\
    ## Round {round_idx}/{n_total_rounds} — Stage B: review suggestions

    ### Objective legend
    {objective_legend_json}

    ### Pool (size {pool_size}, cap {pool_size_cap})
    {pool_block}

    ### History (last {k_history} of {n_history})
    {history_block}

    ### Best so far
    {best_block}

    ### BO suggestions (top-{k_bo} from acquisition={acq_function})
    {bo_suggestions_block}

    ### Stagnation: {stagnation_counter} rounds without any objective's improvement.

    ### PDF context
    {pdf_context}

    {previous_errors_block}

    ### Response format reminder
    Your reply MUST be bare JSON (one object or an array of objects).
    NO fences, NO prose, NO comments. See the format rules at the top
    of the system prompt for details. Allowed block type:
    review_bo (exactly one block)

    ### review_bo block
    ```json
    {{
      "type": "review_bo",
      "rationale": "<= 600 chars",
      "decisions": {{
        "<bo_smiles_1>": "ok" | "override:<NEW_SMILES>" | "skip",
        ...
      }}
    }}
    ```
""").strip()


# ---------------------------------------------------------------------------
# External guidance (steer the LLM's behaviour without changing code)
# ---------------------------------------------------------------------------


def _format_guidance_suffix(guidance: str) -> str:
    """Render the optional ``## EXTERNAL GUIDANCE`` block.

    The user supplies a free-form string (e.g. via the ``LLM_GUIDANCE``
    shell variable in ``run_search.sh``). It is appended verbatim to
    all three system prompts (Stage A1 actions, A2 review-analogs, B
    review-suggestions) so the LLM sees the same steering text in
    every stage. Empty / whitespace-only inputs are a no-op (return
    an empty string so the helper composes cleanly into a
    concatenation).
    """
    g = (guidance or "").strip()
    if not g:
        return ""
    return "\n\n## EXTERNAL GUIDANCE\n" + g


def format_system_actions(round_idx: int, guidance: str = "") -> str:
    """Render the Stage A1 (actions) system prompt, with optional
    external guidance appended."""
    return (
        SYSTEM_ACTIONS.format(round_idx=round_idx)
        + _format_guidance_suffix(guidance)
    )


def format_system_review_analogs(round_idx: int, guidance: str = "") -> str:
    """Render the Stage A2 (review-analogs) system prompt, with
    optional external guidance appended."""
    return (
        SYSTEM_REVIEW_ANALOGS.format(round_idx=round_idx)
        + _format_guidance_suffix(guidance)
    )


def format_system_review_suggestions(round_idx: int, guidance: str = "") -> str:
    """Render the Stage B (review-suggestions) system prompt, with
    optional external guidance appended."""
    return (
        SYSTEM_REVIEW_SUGGESTIONS.format(round_idx=round_idx)
        + _format_guidance_suffix(guidance)
    )


# ---------------------------------------------------------------------------
# Truncation constants
# ---------------------------------------------------------------------------


POOL_CAP: int = 50
HISTORY_CAP: int = 20
PARETO_CAP: int = 5
PDF_CAP_CHARS: int = 1500


# ---------------------------------------------------------------------------
# Internal formatters
# ---------------------------------------------------------------------------


def _format_pool(pool: Sequence[str], cap: int = POOL_CAP) -> str:
    pool_list = list(pool)
    if len(pool_list) <= cap:
        return "\n".join(f"  - {s}" for s in pool_list) or "  (empty)"
    head = "\n".join(f"  - {s}" for s in pool_list[:cap])
    return head + f"\n  ... (+{len(pool_list) - cap} more, truncated)"


def _format_history(
    history: Sequence, cap: int = HISTORY_CAP,
) -> str:
    """Format history. Each row is ``(smi, score)`` where ``score`` is
    ``float`` (n_obj=1) or ``list[float]`` (n_obj>=2)."""
    rows: List[str] = []
    for entry in history[-cap:]:
        smi, sc = entry
        if isinstance(sc, (list, tuple)):
            sc_str = ", ".join("None" if v is None else f"{float(v):.3f}" for v in sc)
            rows.append(f"  - {smi}  ->  [{sc_str}]")
        elif sc is None:
            rows.append(f"  - {smi}  ->  [None]")
        else:
            rows.append(f"  - {smi}  ->  [{float(sc):.3f}]")
    if not rows:
        return "  (empty)"
    suffix = (
        f"\n  ... (showing last {cap} of {len(history)})"
        if len(history) > cap else ""
    )
    return "\n".join(rows) + suffix


def _format_best(best: Union[str, List[str]], n_obj: int) -> str:
    """Format the best anchor for the prompt.

    n_obj == 1: a single SMILES (or "(none yet)" if empty).
    n_obj >= 2: a Pareto front (list of SMILES, truncated).
    """
    if n_obj == 1:
        if isinstance(best, str) and best:
            return f"  {best}"
        return "  (none yet — pool is fresh)"
    # Multi-obj: Pareto front
    if isinstance(best, (list, tuple)) and best:
        if len(best) <= PARETO_CAP:
            return "\n".join(f"  - {s}" for s in best)
        head = "\n".join(f"  - {s}" for s in best[:PARETO_CAP])
        return head + (
            f"\n  ... (+{len(best) - PARETO_CAP} more in the Pareto front)"
        )
    return "  (none yet — pool is fresh)"


def _format_mu_sigma(v: Any) -> str:
    """Format a ``mu`` / ``sigma`` value (float or list[float])."""
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(f"{x:.3f}" for x in v) + "]"
    return f"{float(v):.3f}"


def _format_bo_suggestions(bo_suggestions: Sequence) -> str:
    if not bo_suggestions:
        return "  (none — BO has not run for this round yet)"
    rows: List[str] = []
    for p in bo_suggestions:
        rows.append(
            f"  - {p.smiles}  mu={_format_mu_sigma(p.mu)}  "
            f"sigma={_format_mu_sigma(p.sigma)}  acq={p.acq_value:.3f}"
        )
    return "\n".join(rows)


def _format_analogs(analogs: Sequence) -> str:
    if not analogs:
        return "  (none)"
    rows: List[str] = []
    for a in analogs:
        rs = f"{a.reasyn_score:.3f}" if a.reasyn_score is not None else "?"
        rows.append(
            f"  - seed={a.seed_smiles}  analogue={a.analogue_smiles}  "
            f"reasyn_score={rs}  steps={a.num_steps}"
        )
    return "\n".join(rows)


def _truncate_pdf(text: str, cap: int = PDF_CAP_CHARS) -> str:
    text = text or ""
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n\n[... truncated, original length={len(text)} chars]"


def _format_previous_errors(errors: Sequence[str]) -> str:
    if not errors:
        return ""
    lines = ["### Previous errors (you must fix these)"]
    saw_parse_error = False
    for i, e in enumerate(errors, start=1):
        lines.append(f"{i}. {e}")
        if e.startswith("ParseError:"):
            saw_parse_error = True
    if saw_parse_error:
        lines.append(
            "\nFormat reminder: emit BARE JSON — one object, or a JSON "
            "array of objects. Do NOT wrap your reply in ```json fences "
            "or add any prose."
        )
    return "\n".join(lines)


def _format_objective_legend(legend: Sequence[dict]) -> str:
    if not legend:
        return "  (no objectives declared)"
    rows: List[str] = []
    for obj in legend:
        name = obj.get("name", "?")
        mn = obj.get("minimize", True)
        ref = obj.get("ref", None)
        ref_str = f", ref={ref}" if ref is not None else ""
        rows.append(f"  - {name}  minimize={mn}{ref_str}")
    return "\n".join(rows)


def _format_pool_size_requirement(state: PreActionState) -> str:
    """Render a pool-size requirement hint for Stage A1.

    When ``state.pool_min_size > 0`` and the current pool is below
    that minimum, instruct the LLM that ``noop`` is rejected and that
    it must emit ``propose`` or ``analog`` to refill the pool.

    The hint includes the exact current count and required minimum so
    the LLM knows exactly how many new SMILES are needed.
    """
    min_size = getattr(state, "pool_min_size", 1) or 1
    if min_size <= 0:
        return ""
    n_pool = len(state.pool)
    if n_pool >= min_size:
        return ""
    deficit = min_size - n_pool
    return (
        f"\n### Pool size requirement\n"
        f"CRITICAL: pool has {n_pool} SMILES, but the minimum is "
        f"{min_size} (deficit: {deficit}). BO cannot run with fewer "
        f"than {min_size} pool members. You MUST emit:\n"
        f"  - a `propose` block with at least {deficit} new SMILES, OR\n"
        f"  - an `analog` block to expand existing pool members until "
        f"the pool reaches {min_size}.\n"
        f"A bare `noop` block WILL be rejected — the system will "
        f"loop back to you until the pool is large enough.\n"
    )


# ---------------------------------------------------------------------------
# Public renderers
# ---------------------------------------------------------------------------

# We infer n_obj from the history shape: if any entry's score is a
# list/tuple → n_obj>=2; otherwise n_obj=1.
def _infer_n_obj_from_state(state) -> int:
    for _, sc in state.history:
        if isinstance(sc, (list, tuple)):
            return len(sc)
    return 1


def render_user_actions(state: PreActionState) -> str:
    """Render the user prompt for Stage A1 (actions)."""
    n_obj = _infer_n_obj_from_state(state)
    return _USER_TEMPLATE_ACTIONS.format(
        round_idx=state.round_idx,
        n_total_rounds=state.n_total_rounds,
        objective_legend_json=_format_objective_legend(state.objective_legend),
        pool_size=len(state.pool),
        pool_size_cap=state.pool_size_cap,
        pool_block=_format_pool(state.pool),
        k_history=min(len(state.history), HISTORY_CAP),
        n_history=len(state.history),
        history_block=_format_history(state.history),
        best_block=_format_best(state.best, n_obj),
        stagnation_counter=state.stagnation_counter,
        pdf_context=_truncate_pdf(state.pdf_context),
        previous_errors_block=_format_previous_errors(state.previous_errors),
        pool_size_requirement_hint=_format_pool_size_requirement(state),
    )


def render_user_review_analogs(state: PreReviewAnalogsState) -> str:
    """Render the user prompt for Stage A2 (review analogs)."""
    return _USER_TEMPLATE_REVIEW_ANALOGS.format(
        round_idx=state.round_idx,
        n_total_rounds=state.n_total_rounds,
        n_analogs=len(state.new_analogs),
        analogs_block=_format_analogs(state.new_analogs),
    )


def render_user_suggestions(state: PostSuggestionState) -> str:
    """Render the user prompt for Stage B (review suggestions)."""
    n_obj = _infer_n_obj_from_state(state)
    return _USER_TEMPLATE_SUGGESTIONS.format(
        round_idx=state.round_idx,
        n_total_rounds=state.n_total_rounds,
        objective_legend_json=_format_objective_legend(state.objective_legend),
        pool_size=len(state.pool),
        pool_size_cap=state.pool_size_cap,
        pool_block=_format_pool(state.pool),
        k_history=min(len(state.history), HISTORY_CAP),
        n_history=len(state.history),
        history_block=_format_history(state.history),
        best_block=_format_best(state.best, n_obj),
        k_bo=len(state.bo_suggestions),
        acq_function=state.acq_function,
        bo_suggestions_block=_format_bo_suggestions(state.bo_suggestions),
        stagnation_counter=state.stagnation_counter,
        pdf_context=_truncate_pdf(state.pdf_context),
        previous_errors_block=_format_previous_errors(state.previous_errors),
    )
