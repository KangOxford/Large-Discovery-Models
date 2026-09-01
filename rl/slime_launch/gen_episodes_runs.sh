#!/bin/bash
# Generate the episode prompt-data for every run in the matrix (see RUNS.md).
# The reward policy + acquisition aggregation are baked into each jsonl here,
# so the launcher only picks the model. Reads config_real.json for sizes and
# the real-evaluation kwargs (vina/nn/target), and stamps gp_history_file into
# each episode's real kwargs so the shared warm GP is wired automatically.
set -eu
REPO_ROOT=${REPO_ROOT:-/mnt/data0/ys/LDM}
CONFIG=${CONFIG:-$REPO_ROOT/rl/slime_launch/config_real.json}
export PYTHONPATH=$REPO_ROOT/rl:$REPO_ROOT:${PYTHONPATH:-}
EP=rl.ldm_rl.episodes

read -r COUNT ITERS RES EVALS WARMUP WARMUP_ITERS < <(python3 -c "
import json;c=json.load(open('$CONFIG'));e=c['episodes'];w=c['warmup']
print(e['count'],e['iterations'],e['reservoir_size'],e['evaluations_per_round'],w['num_samples'],w['iterations'])")

# real kwargs = config.real_kwargs + gp_history_file
RK=$(python3 -c "
import json;c=json.load(open('$CONFIG'))
rk=dict(c['real_kwargs']);rk['gp_history_file']=c['gp_history_file']
print(json.dumps(rk))")

gen() {  # gen <out> <reward> <agg> <count> <iters>
  python3 -m $EP --output "$REPO_ROOT/$1" --task small_molecule --mode real \
    --count "$4" --iterations "$5" --reservoir-size "$RES" --evaluations-per-round "$EVALS" \
    --reward "$2" --acquisition-agg "$3" --real-kwargs "$RK"
}

# warm-up (rollout-only; reward unused, keep acquisition for consistency)
gen rl_episodes_sm_warmup.jsonl   acquisition  max  "$WARMUP" "$WARMUP_ITERS"
# R1/R2: acquisition-max
gen rl_episodes_sm_acqmax.jsonl   acquisition  max  "$COUNT"  "$ITERS"
# R4: acquisition-mean
gen rl_episodes_sm_acqmean.jsonl  acquisition  mean "$COUNT"  "$ITERS"
# R3: real-outcome reward (Pareto hypervolume improvement over the observed front)
gen rl_episodes_sm_hv.jsonl       hypervolume  max  "$COUNT"  "$ITERS"
echo "episodes written to $REPO_ROOT/rl_episodes_sm_{warmup,acqmax,acqmean,hv}.jsonl"
