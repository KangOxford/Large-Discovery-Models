#!/usr/bin/env python3
"""Harvest the lr=0 control arm against the frozen rules in PRESPEC_lr0_control.md.

Reports difference, equivalence and power as three separate verdicts, never collapsing
them into one. Records why any run was excluded rather than silently dropping it.
"""
import json, glob, re, subprocess
from pathlib import Path
import numpy as np
from scipy import stats

T = Path("/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_rl")
K       = 400      # frozen
WARM    = 63       # frozen
DELTA   = 0.10     # frozen minimum effect of interest / equivalence margin
ALPHA   = 0.05
POWER   = 0.80

def vina(gp):
    o = []
    for ln in Path(gp).read_text(errors="replace").splitlines():
        ln = ln.strip()
        if not ln: continue
        try: d = json.loads(ln)
        except Exception: continue
        v = d.get("vina", d.get("score", d.get("docking")))
        if v is None: continue
        try: o.append(float(v))
        except Exception: pass
    return o[WARM:]

def argv_n(d):
    lg = Path(d)/"train.log"
    if not lg.exists(): return None
    m = re.search(r'--n-samples-per-prompt\s+(\d+)', lg.read_text(errors="replace")[:400000])
    return int(m.group(1)) if m else None

def slurm_state(run):
    """Why a run stopped, from sacct rather than from the absence of data."""
    try:
        out = subprocess.run(["sacct","--format=JobID,JobName%40,State,Elapsed","-Pn"],
                             capture_output=True, text=True, timeout=90).stdout
    except Exception:
        return "sacct unavailable"
    for line in out.splitlines():
        if run in line:
            f = line.split("|")
            return f"{f[2]} after {f[3]}"
    return "no sacct record"

def collect(patterns, arm):
    kept, dropped = [], []
    for pat in patterns:
        for ds in sorted(glob.glob(str(T/pat))):
            d = Path(ds); name = d.name.rsplit("_",1)[0]
            gp = d/"gp_history.jsonl"
            if not gp.exists():
                dropped.append((name, "no gp_history.jsonl", slurm_state(name))); continue
            n = argv_n(d)
            if n != 4:
                dropped.append((name, f"argv n_samples={n}, rule requires 4", slurm_state(name))); continue
            v = vina(gp)
            if len(v) < K:
                dropped.append((name, f"{len(v)} proposals < K={K}", slurm_state(name))); continue
            r, _ = stats.spearmanr(np.arange(K), v[:K])
            kept.append((name, float(r), len(v)))
    return kept, dropped

trained, tr_drop = collect(("runs/M-dhvn4-*","runs/X-dhvn4-*"), "trained")
control, ct_drop = collect(("runs/LR0-*","runs/LR0c*"), "control")

print("="*76)
print(f"lr=0 control arm, frozen rules: K={K}, warm-start {WARM} dropped, argv n_samples==4")
print("="*76)
for arm, kept, drop in (("TRAINED (lr=1e-6)", trained, tr_drop), ("CONTROL (lr=0)", control, ct_drop)):
    print(f"\n{arm}: {len(kept)} qualifying")
    for n_, r_, L in kept:
        print(f"    {n_:<40} rho {r_:+.4f}   ({L} proposals)")
    if drop:
        print(f"  excluded {len(drop)}, with reasons preserved:")
        for n_, why, st in drop:
            print(f"    {n_:<40} {why:<34} [{st}]")

if len(trained) < 2 or len(control) < 2:
    print("\nAn arm has fewer than 2 qualifying runs. No test; the control arm is still filling.")
    print("Verdicts: DIFFERENCE — not testable yet.  EQUIVALENCE — not testable yet.")
    print(f"          POWER — control arm at {len(control)}/{5} of the frozen target.")
else:
    a = np.array([r for _,r,_ in trained]); b = np.array([r for _,r,_ in control])
    t, df, diff = (lambda va,vb: ((a.mean()-b.mean())/np.sqrt(va+vb),
                                  (va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1)),
                                  a.mean()-b.mean()))(a.var(ddof=1)/len(a), b.var(ddof=1)/len(b))
    se = np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
    tc = stats.t.ppf(1-ALPHA/2, df)
    ci = (diff-tc*se, diff+tc*se)
    p2 = 2*(1-stats.t.cdf(abs(t), df))
    print(f"\n1. DIFFERENCE (frozen test: Welch t, alpha={ALPHA} two-sided)")
    print(f"   trained mean {a.mean():+.4f} (n={len(a)}, sd {a.std(ddof=1):.4f})")
    print(f"   control mean {b.mean():+.4f} (n={len(b)}, sd {b.std(ddof=1):.4f})")
    print(f"   diff {diff:+.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  t={t:+.2f} df={df:.1f} p={p2:.4f}")
    print(f"   -> {'DIFFERENT' if p2<ALPHA else 'no significant difference'}")
    tl = (diff-(-DELTA))/se; tu = (DELTA-diff)/se
    pl = 1-stats.t.cdf(tl, df); pu = 1-stats.t.cdf(tu, df)
    print(f"\n2. EQUIVALENCE (frozen TOST, margin |delta| < {DELTA})")
    print(f"   lower p={pl:.4f}   upper p={pu:.4f}   -> "
          f"{'EQUIVALENT within the frozen margin' if max(pl,pu)<ALPHA else 'NOT established'}")
    sd_pool = np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
    n_need = 2*((1.96+0.842)*sd_pool/DELTA)**2
    res = (1.96+0.842)*sd_pool*np.sqrt(2/min(len(a),len(b)))
    print(f"\n3. POWER, re-estimated from the completed arms (the sizing SD was a prior)")
    print(f"   pooled sd {sd_pool:.4f} (sizing prior was 0.0552)")
    print(f"   runs per arm needed for |delta|={DELTA}: {n_need:.1f}; smallest arm has {min(len(a),len(b))}")
    print(f"   resolvable |delta| achieved at 80% power: {res:.3f}")
    print(f"   -> {'adequately powered at the frozen margin' if n_need<=min(len(a),len(b)) else 'UNDERPOWERED at the frozen margin'}")
