"""Freeze every number the reward write-up uses into facts.json.

The write-up and the notebook both read this file rather than the logs, so the
prose cannot drift away from the measurement. Each entry carries where it came
from: a file:line for anything read out of source, a log path for anything read
out of a run. Numbers that are arithmetic rather than measurement are marked
`derivation` so a reader never has to guess which is which.

Run from anywhere; paths are absolute because the logs live outside the repo.
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

RL = Path("/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/LDM-rl/rl")
ENV_PY = RL / "ldm_rl" / "env.py"
PPO = RL / "slime" / "slime" / "utils" / "ppo_utils.py"
ARGS = RL / "slime" / "slime" / "utils" / "arguments.py"
DATA_PY = RL / "slime" / "slime" / "backends" / "megatron_utils" / "data.py"
RUNS = Path("/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_rl/runs")

# Two logs per run, and the difference matters. The top-level file is the srun
# orchestration log; the per-step metrics -- including the zero-variance counters
# -- are only in runs/<RUN>/train.log. Reading only the former is what made an
# earlier version of this file report the instrumentation as absent.
RUN_LOGS = {
    "9b-sft-a": (RUNS / "9b_sft_split_20260909T071112Z.log",
                 RUNS / "9b-sft_20260909T071143Z" / "train.log"),
    "9b-sft-b": (RUNS / "9b_sft_split_20260909T094237Z.log",
                 RUNS / "9b-sft-gbs2_20260909T094256Z" / "train.log"),
    "9b-base": (RUNS / "9b_base_split_20260909T073652Z.log",
                RUNS / "9b-base_20260909T073721Z" / "train.log"),
    # The K=8 cell. Same SFT arm, same 50 rollouts, same 100 groups as 9b-sft-a,
    # so the group counts compare without any adjustment for length. It also
    # carries UPDATES_PER_ROLLOUT=2 against the others' 1 -- see the note in
    # facts["k8_confound"] for why that does not touch this particular metric.
    "k8sft": (RUNS / "k8sft_20260909T200313Z" / "train.log",
              RUNS / "k8sft_20260909T200313Z" / "train.log"),
}


def line_of(path: Path, pattern: str) -> int | None:
    """1-indexed line number of the first regex hit, so citations stay checkable."""
    rx = re.compile(pattern)
    for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if rx.search(line):
            return i
    return None


def scalar_from_log(text: str, key: str) -> str | None:
    """Megatron prints resolved args as `name ....... value`; take the value."""
    m = re.search(rf"^\s*{re.escape(key)} \.+ (\S+)\s*$", text, re.M)
    return m.group(1) if m else None


def zero_std_series(text: str, key: str) -> list[float]:
    """slime emits these once per rollout, bucketed by reward value.

    `count_0.0` is a bucket key, so it is present only on rollouts where at
    least one group scored exactly zero; the other three are emitted every
    rollout. Summing them is therefore only meaningful against the number of
    groups, which is rollout_batch_size per step.
    """
    return [float(x) for x in
            re.findall(rf"zero_std/{re.escape(key)}'?\s*:\s*([-0-9.eE+]+)", text)]


def collect_run(name: str, paths: tuple[Path, Path]) -> dict:
    orch, train = paths
    text = orch.read_text(errors="replace")
    ttext = train.read_text(errors="replace") if train.exists() else ""
    rewards = [float(x) for x in re.findall(r"'rollout/raw_reward': ([-0-9.eE+]+)", text)]
    n = len(rewards)
    half = n // 2
    zeros = [i for i, r in enumerate(rewards) if r == 0.0]
    ng = zero_std_series(ttext, "count_no_gradient")
    le = zero_std_series(ttext, "count_lt_eps")
    c0 = zero_std_series(ttext, "count_0.0")
    sf = zero_std_series(ttext, "mode_is_scale_free")
    rb = scalar_from_log(text, "rollout_batch_size")
    groups = (int(rb) * len(ng)) if (rb and rb.isdigit() and ng) else None
    return {
        "log": str(orch),
        "train_log": str(train),
        "steps": n,
        # Measured directly, not inferred from the group mean.
        "groups_total": groups,
        "groups_no_gradient": int(sum(ng)) if ng else None,
        "groups_lt_eps": int(sum(le)) if le else None,
        "groups_exactly_zero": int(sum(c0)) if c0 else None,
        "no_gradient_frac": (sum(ng) / groups) if (ng and groups) else None,
        "mode_is_scale_free": sorted(set(sf)) if sf else None,
        # `rollout/raw_reward` is the MEAN over the rollout, not a per-sample value.
        "raw_reward_series_is_group_mean": True,
        "raw_reward": rewards,
        "n_zero": len(zeros),
        "zero_frac": len(zeros) / n if n else None,
        "zero_frac_first_half": sum(1 for i in zeros if i < half) / half if half else None,
        "zero_frac_second_half": (
            sum(1 for i in zeros if i >= half) / (n - half) if n - half else None
        ),
        "n_samples_per_prompt": scalar_from_log(text, "n_samples_per_prompt"),
        "rollout_batch_size": scalar_from_log(text, "rollout_batch_size"),
        "grpo_advantage_std_mode": scalar_from_log(text, "grpo_advantage_std_mode"),
        "advantage_estimator": scalar_from_log(text, "advantage_estimator"),
        "normalize_advantages": scalar_from_log(text, "normalize_advantages"),
        "resolved_batch_line": (
            m.group(0)
            if (m := re.search(r"resolved: rollout_batch=\S+ n_samples=\S+ "
                               r"updates_per_rollout=\S+ -> global_batch=\d+", text))
            else None
        ),
        # The keys that would separate "all zero" from "all tied" are emitted by
        # slime but do not appear in these logs; record the absence rather than
        # leaving a reader to assume they were checked.
        # The group std is not logged, so the floor cannot be measured directly.
        # The mean bounds it: a group whose mean is below 1e-5 cannot be spread
        # much wider. Counting those says how often the floor could matter.
        "n_mean_below_1e5": sum(1 for r in rewards if 0.0 < r < 1e-5),
        "n_mean_below_1e6": sum(1 for r in rewards if 0.0 < r < 1e-6),
        "n_nonzero": sum(1 for r in rewards if r > 0.0),
        # Present in train.log, absent from the orchestration log. The earlier
        # claim that they were absent everywhere came from checking only the latter.
        "zero_std_keys_in_orch_log": "zero_std/" in text,
        "zero_std_keys_in_train_log": "zero_std/" in ttext,
        "reward_kind_in_train_log": bool(re.search(r"'kind': '", ttext)),
    }


def git_rev(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip() or None
    except Exception:
        return None


facts = {
    "collected_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "ldm_rl_head": git_rev(RL.parent),
    "source": {
        "env_py": str(ENV_PY),
        "reward_policies_line": line_of(ENV_PY, r"^REWARD_POLICIES = "),
        "reward_policies": ["improvement", "raw", "binary", "acquisition", "hypervolume"],
        "reward_default": "improvement",
        "reward_default_line": line_of(ENV_PY, r'reward: str = "improvement"'),
        "reward_failure_line": line_of(ENV_PY, r"reward_failure: float"),
        "reward_invalid_line": line_of(ENV_PY, r"reward_invalid: float"),
        "reward_failure_value": 0.0,
        "reward_invalid_value": 0.0,
        "baseline_line": line_of(ENV_PY, r"baseline = self\._componentwise_best"),
        "improvement_clip_line": line_of(ENV_PY, r"improvements = tuple\(max\(0\.0"),
        "advantage_std_eps": 1e-6,
        "advantage_std_eps_line": line_of(PPO, r"^ADVANTAGE_STD_EPS = "),
        "std_mode_default": "legacy_eps",
        "std_mode_default_line": line_of(ARGS, r'"--grpo-advantage-std-mode"'),
        "normalize_advantages_default": False,
        "normalize_advantages_line": line_of(ARGS, r'"--normalize-advantages"'),
        "logged_value_is_mean_line": line_of(DATA_PY, r"log_dict\[key\] = \(val\.float\(\)\.mean\(\)"),
    },
    "runs": {k: collect_run(k, v) for k, v in RUN_LOGS.items() if v[0].exists()},
    "derivation": {
        "note": "Arithmetic, not measurement. Both follow from the definitions above.",
        "k2_legacy_eps": (
            "For a group of 2, centered = +/-(r1-r2)/2 and the population std is "
            "|r1-r2|/2, so (r-mean)/std is exactly +/-1 for any non-zero difference: "
            "the advantage carries the sign of which sample was better and nothing "
            "about by how much."
        ),
        "k2_scale_free": (
            "scale_free divides by the unbiased std, |r1-r2|/sqrt(2), giving exactly "
            "+/-1/sqrt(2). Its own bound |advantage| <= sqrt(n-1) is 1 at n=2. "
            "Changing the std mode therefore does not restore magnitude at K=2."
        ),
        "eps_attenuation": "legacy_eps scales by std/(std+1e-6): 2x at std 1e-6, 101x at 1e-8.",
    },
    # From PR #5's frozen facts (rl/ldm_skyrl/facts.json, figure F3) on this same
    # line. Carried here with its unequal step counts attached, because the two
    # 0.00 readings rest on 24 and 15 steps rather than on equal power.
    "k8_confound": (
        "The K=8 cell differs from the K=2 runs in two ways, not one: "
        "n_samples_per_prompt 2 -> 8 and UPDATES_PER_ROLLOUT 1 -> 2. For the "
        "zero-variance counts specifically the comparison is still one-factor, "
        "because _compute_zero_std_metrics runs over the rollout's samples "
        "before any optimizer update, and UPDATES_PER_ROLLOUT only sets how "
        "many inner updates a rollout gets. Every other quantity here is "
        "confounded and should not be read as a K effect."
    ),
    "zero_variance_vs_n_from_pr5": {
        "fraction_of_groups_with_zero_std": {"2": 0.87, "4": 0.28, "8": 0.00, "16": 0.00},
        "step_counts": {"2": 52, "4": 32, "8": 24, "16": 15},
        "source": "KangOxford/Large-Discovery-Models PR #5, rl/ldm_skyrl/facts.json",
    },
}

out = Path(__file__).with_name("facts.json")
out.write_text(json.dumps(facts, indent=2, sort_keys=False) + "\n")
print(f"wrote {out}")
for k, r in facts["runs"].items():
    print(f"  {k}: steps={r['steps']} zero={r['n_zero']} "
          f"({r['zero_frac']:.0%})  no-gradient groups "
          f"{r['groups_no_gradient']}/{r['groups_total']} "
          f"exactly-zero {r['groups_exactly_zero']}")
