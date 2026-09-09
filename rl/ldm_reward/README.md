# Why the 9B GRPO runs produce no gradient

`reward_zero_variance.ipynb` is the write-up. It reads `facts.json` and nothing
else, so it can be re-run anywhere without the cluster, the logs, or the network.

```
collect_facts.py   reads the run logs and the ldm_rl / slime sources -> facts.json
facts.json         every number the notebook and the PR body use
reward_zero_variance.ipynb   the figures and the argument
figs/              PNGs, 300 dpi, quantised to 256 colours
```

To refresh after new runs, edit `RUN_LOGS` in `collect_facts.py`, run it, then
execute the notebook in place:

```bash
python collect_facts.py
jupyter nbconvert --execute --inplace reward_zero_variance.ipynb
```

`collect_facts.py` records where each fact came from — a `file:line` for anything
read out of source, a log path for anything read out of a run — and marks the two
entries that are arithmetic rather than measurement as `derivation`.
