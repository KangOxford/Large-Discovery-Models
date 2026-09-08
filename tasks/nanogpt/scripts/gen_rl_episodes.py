#!/usr/bin/env python3
"""Write Slime prompt-data for nanoGPT RL episodes.

The shared ``ldm_rl.episodes`` CLI stamps every row with the *same*
``real_kwargs``, so every episode would be the same tuning problem. Instance
diversity is task-specific -- which knob subsets are worth isolating, which
budgets change the optimum -- so it lives here.

One row = one :class:`EpisodeSpec` = one instance:

* **free knobs**: either one/two knob *groups* (architecture, learning rates,
  schedule, batching) or the full 14. Group instances are the interesting ones:
  they ask "tune the learning rates for this architecture" rather than
  "everything at once", and they are what makes the prompts differ.
* **wall-clock budget**: varying it moves the optimum, because the budget is
  wall clock -- a bigger model sees fewer tokens in the same time.
* **reward_ref_point**: the measured reference for that budget, negated,
  because ``reward_ref_point`` is in oriented (maximisation) space and
  ``val_bpb`` is minimised.

Pinned knobs stay at their defaults, which makes the reference configuration
identical across instances, so one reference measurement per budget covers
every episode. (An earlier version of this task randomised pinned values to
close a baseline-copy exploit: the reward there was "0 = baseline" *and* the
prompt printed the baseline, so emitting the defaults was a guaranteed
non-loss. That exploit does not exist here -- emitting the reference scores
exactly 0, the same as any other non-improving proposal -- so the extra
reference runs randomisation would cost are not worth it. Pass
``--randomise-pinned`` if you want the extra prompt diversity and can afford
one reference run per instance.)

    python -m tasks.nanogpt.scripts.gen_rl_episodes \
        --output rl_episodes_nanogpt.jsonl --count 64 --iterations 4 \
        --references rl/nanogpt_reference.json \
        --real-kwargs '{"output_dir": "/scratch/nanogpt_rl", "eval_gpus": "4,5,6,7"}'
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

if __package__ in (None, ""):
    _REPO = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    for _path in (os.path.join(_REPO, "rl"), _REPO):
        if _path not in sys.path:
            sys.path.insert(0, _path)

from ldm_rl.episodes import EpisodeSpec, make_prompt_rows  # noqa: E402

from tasks.nanogpt.core import rl_knobs as knobs_mod  # noqa: E402

#: Knob groups an instance can hold free. Grouping is by what a practitioner
#: would tune together, which is also what makes each instance a coherent
#: question rather than a random mask.
KNOB_GROUPS: dict[str, list[str]] = {
    "architecture": ["DEPTH", "ASPECT_RATIO", "HEAD_DIM", "WINDOW_PATTERN"],
    "learning_rates": ["EMBEDDING_LR", "UNEMBEDDING_LR", "MATRIX_LR", "SCALAR_LR"],
    "schedule": ["WARMUP_RATIO", "WARMDOWN_RATIO", "FINAL_LR_FRAC", "WEIGHT_DECAY"],
    "batching": ["TOTAL_BATCH_SIZE", "DEVICE_BATCH_SIZE"],
}


#: Treat a continuous knob as unbounded for space-size purposes.
_INFINITE = float("inf")


_SPACE_CACHE: dict[tuple[str, ...], float] = {}


def legal_space_size(free: list[str], schema: dict) -> float:
    """How many distinct configs an instance's free knobs can express.

    Returns ``inf`` as soon as a continuous knob is free. Exact (not a product
    bound) for the all-discrete case, because the batching pair interacts: of
    the 3 x 4 = 12 (TOTAL_BATCH_SIZE, DEVICE_BATCH_SIZE) combinations only 9
    are legal, since DEVICE_BATCH_SIZE=96 divides none of the available total
    batch sizes.

    Memoised: the all-discrete architecture group alone enumerates 26k configs,
    and instance sampling asks about the same few subsets repeatedly.
    """

    cache_key = tuple(sorted(free))
    if cache_key in _SPACE_CACHE:
        return _SPACE_CACHE[cache_key]

    discrete: list[list[object]] = []
    for name in free:
        spec = schema[name]
        if spec.get("choices"):
            discrete.append(list(spec["choices"]))
        elif spec["type"] in ("int", "integer"):
            discrete.append(list(range(int(spec["min"]), int(spec["max"]) + 1)))
        else:
            _SPACE_CACHE[cache_key] = _INFINITE
            return _INFINITE

    import itertools

    legal = 0
    for combo in itertools.product(*discrete):
        candidate = knobs_mod.with_defaults(free, dict(zip(free, combo)))
        try:
            knobs_mod.check_hard_constraints(knobs_mod.validate(candidate))
        except knobs_mod.ProposalError:
            continue
        legal += 1
    _SPACE_CACHE[cache_key] = float(legal)
    return float(legal)


def _sample_free_knobs(
    rng: random.Random, mode: str, schema: dict, min_space: int
) -> tuple[str, list[str]]:
    """Pick the free-knob set for one instance, wide enough to search in.

    A subset whose legal space is smaller than the number of proposals an
    episode will make leaves the policy nothing to propose after a few rounds:
    the reservoir de-duplicates against what has already been evaluated, so it
    ends up empty and the episode terminates early on ``empty_reservoir_limit``
    rather than on its budget. ``batching`` alone is the real case -- 9 legal
    configs total -- so it gets merged with another group instead of being
    dropped, which keeps the batching knobs in the training distribution.
    """

    groups = sorted(KNOB_GROUPS)
    if mode == "all":
        return "all", list(knobs_mod.knob_names())
    if mode == "groups":
        count = rng.choice((1, 2))
    else:  # "mixed"
        roll = rng.random()
        if roll < 0.25:
            return "all", list(knobs_mod.knob_names())
        count = 1 if roll < 0.7 else 2

    chosen = rng.sample(groups, count)
    while True:
        free: list[str] = []
        for name in chosen:
            free.extend(KNOB_GROUPS[name])
        if legal_space_size(free, schema) >= min_space:
            return "+".join(sorted(chosen)), free
        remaining = [name for name in groups if name not in chosen]
        if not remaining:
            return "all", list(knobs_mod.knob_names())
        chosen.append(rng.choice(remaining))


def _randomised_pinned(
    rng: random.Random, free: list[str], schema: dict
) -> dict[str, object]:
    """Pinned values sampled from the schema, kept jointly legal.

    Only the batching pair interacts, so retry until the pinned part cannot by
    itself make every proposal illegal.
    """

    held = [name for name in knobs_mod.knob_names() if name not in free]
    for _ in range(200):
        pinned: dict[str, object] = {}
        for name in held:
            spec = schema[name]
            if spec.get("choices"):
                pinned[name] = rng.choice(list(spec["choices"]))
            elif spec["type"] in ("int", "integer"):
                pinned[name] = rng.randint(int(spec["min"]), int(spec["max"]))
            else:
                pinned[name] = round(rng.uniform(spec["min"], spec["max"]), 6)
        candidate = knobs_mod.with_defaults(free, {}, pinned)
        try:
            knobs_mod.check_hard_constraints(knobs_mod.validate(candidate))
        except knobs_mod.ProposalError:
            continue
        return pinned
    raise RuntimeError("could not sample a legal pinned assignment")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument(
        "--iterations",
        type=int,
        default=4,
        help=(
            "propose/measure rounds per episode. Each round costs one real "
            "training run, so this is the main cost dial: "
            "episodes x n_samples x iterations x budget of GPU time per step."
        ),
    )
    parser.add_argument("--reservoir-size", type=int, default=2)
    parser.add_argument(
        "--evaluations-per-round",
        type=int,
        default=1,
        help=(
            "how many of the proposed candidates get measured. Keeping this "
            "below --reservoir-size is what gives the GP something to do: it "
            "chooses which proposal is worth a full training budget."
        ),
    )
    parser.add_argument("--budgets", default="300")
    parser.add_argument(
        "--references",
        default="",
        help="JSON map from budget to measured reference val_bpb "
        "(from tasks.nanogpt.scripts.rl_reference)",
    )
    parser.add_argument(
        "--free-knob-mode", choices=("mixed", "groups", "all"), default="mixed"
    )
    parser.add_argument("--randomise-pinned", action="store_true")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--real-kwargs",
        default="",
        help="JSON object of evaluation plumbing merged into every episode",
    )
    parser.add_argument(
        "--allow-missing-reference",
        action="store_true",
        help=(
            "emit episodes with no reward_ref_point. NOT recommended: the "
            "improvement reward then measures against the episode's own first "
            "proposal, which pays the policy for opening badly."
        ),
    )
    args = parser.parse_args(argv)

    if args.count < 1:
        parser.error("--count must be positive")
    if args.evaluations_per_round > args.reservoir_size:
        parser.error("--evaluations-per-round cannot exceed --reservoir-size")

    budgets = [float(item) for item in args.budgets.split(",") if item.strip()]
    if not budgets:
        parser.error("--budgets must list at least one budget")

    base_real: dict[str, object] = {}
    if args.real_kwargs.strip():
        try:
            parsed = json.loads(args.real_kwargs)
        except json.JSONDecodeError as exc:
            parser.error(f"--real-kwargs is not valid JSON: {exc}")
        if not isinstance(parsed, dict):
            parser.error("--real-kwargs must decode to a JSON object")
        base_real = dict(parsed)

    references: dict[str, float] = {}
    if args.references:
        with open(args.references, encoding="utf-8") as handle:
            references = {str(k): float(v) for k, v in json.load(handle).items()}
    missing = [f"{b:.0f}" for b in budgets if f"{b:.0f}" not in references]
    if missing and not args.allow_missing_reference:
        parser.error(
            "no measured reference for budget(s) "
            + ", ".join(missing)
            + ". Run tasks.nanogpt.scripts.rl_reference first, or pass "
            "--allow-missing-reference if you accept a sandbaggable reward."
        )

    schema = knobs_mod.load_schema()
    rng = random.Random(args.seed_offset or 0)
    specs: list[EpisodeSpec] = []
    summary: dict[str, int] = {}
    # Every proposal an episode makes should be able to be distinct, plus slack
    # so the last rounds still have somewhere to go.
    min_space = args.iterations * args.reservoir_size + 4

    for index in range(args.count):
        label, free = _sample_free_knobs(rng, args.free_knob_mode, schema, min_space)
        budget = budgets[index % len(budgets)]
        pinned = (
            _randomised_pinned(rng, free, schema) if args.randomise_pinned else {}
        )
        real = dict(base_real)
        real.update(
            {
                "free_knobs": sorted(free),
                "time_budget": budget,
                "reservoir_size": args.reservoir_size,
            }
        )
        if pinned:
            real["pinned"] = pinned
        reference = references.get(f"{budget:.0f}")
        if reference is not None:
            real["reference_val_bpb"] = reference
        specs.append(
            EpisodeSpec(
                task="nanogpt",
                mode="real",
                iterations=args.iterations,
                reservoir_size=args.reservoir_size,
                evaluations_per_round=args.evaluations_per_round,
                reward="improvement",
                # Oriented space: val_bpb is minimised, so the maximisation
                # reference is its negation.
                reward_ref_point=(
                    (-reference,) if reference is not None else None
                ),
                seed=args.seed_offset + index,
                real=real,
            )
        )
        key = f"{label}@{budget:.0f}s"
        summary[key] = summary.get(key, 0) + 1

    rows = make_prompt_rows(specs)
    with open(args.output, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    runs_per_episode = args.iterations * args.evaluations_per_round
    print(f"wrote {len(rows)} episode(s) to {args.output}")
    print(f"  distinct instances: {len(summary)}")
    for key in sorted(summary):
        print(f"    {key:<34} x{summary[key]}")
    print(
        f"  cost: <= {runs_per_episode} real run(s) per episode "
        f"({args.iterations} rounds x {args.evaluations_per_round} eval), "
        f"~{sum(budgets) / len(budgets) + 40:.0f}s each. Repeats across "
        "episodes are served from the shared cache."
    )
    if missing:
        print(
            "  WARNING: no reward_ref_point for budget(s) "
            + ", ".join(missing)
            + " -- the reward is sandbaggable for those episodes."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
