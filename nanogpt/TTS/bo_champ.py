#!/usr/bin/env python3
"""Autonomous 10h+ Bayesian-Optimization orchestrator over a joint
architecture+recipe search space for opt_param.py on 8x H100 (single-GPU per run,
300s karpathy rules). Robust: per-run timeout, failure penalty, atomic checkpoint,
GP->random fallback, wall-clock deadline. Resumable from bo_state.json.

Run:  PY bo_orchestrator.py   (env BO_HOURS=10.5)
State/logs written next to this script.
"""
import os, sys, json, time, math, subprocess, traceback
import numpy as np

SOL = "/mnt/data0/user_logs/alice_scaling_laws/first-steps/nanochat_autoresearch/solutions"
PY = "/mnt/data0/user_logs/alice_scaling_laws/hbo/.venv/bin/python"
STATE = os.path.join(SOL, "bo_champ_state.json")
LOG = os.path.join(SOL, "bo_champ.log")
NGPU = 7   # GPU 0-6 only; GPU 7 reserved for user's other job
SEED = "42"
RUN_TIMEOUT = 620            # kill a training run that overruns (300s train + ~70s overhead, margin)
PENALTY = 1.05               # bpb for failed/OOM runs (worse than vanilla) so BO avoids the region
N_RANDOM = 2                 # of the 8 per round, this many are pure-random exploration
HOURS = float(os.environ.get("BO_HOURS", "10.5"))
DEADLINE = time.time() + HOURS * 3600
GP_MAX = 350                 # cap history points used per GP fit for speed

# FOCUSED exploit-BO tight around the confirmed champion best (d8/dim640/ng48/mlr0.05/wd0.90
# = val_bpb 0.9362 8-seed). DEPTH fixed at 8 (confirmed optimal). NGPU=7 (GPU 0-6).
# name, kind, a, b, scale  (choice: a=list)
SPACE = [
    ("MODEL_DIM",      "choice", [512, 640, 768], None, "choice"),
    ("NGRAM_MULT",     "int",    40,    56,    "lin"),
    ("MATRIX_LR",      "float",  0.040, 0.060, "log"),
    ("EMBEDDING_LR",   "float",  0.45,  0.75,  "log"),
    ("UNEMBEDDING_LR", "float",  0.003, 0.006, "log"),
    ("WARMDOWN_RATIO", "float",  0.85,  0.95,  "lin"),
    ("WEIGHT_DECAY",   "float",  0.06,  0.14,  "lin"),
    ("NS_STEPS",       "int",    5,     7,     "lin"),
]
D = len(SPACE)
NAMES = [s[0] for s in SPACE]

# champion defaults for ALL knobs (non-SPACE knobs stay at these)
BASE = dict(DEPTH=8, MODEL_DIM=640, NGRAM_MULT=48, MATRIX_LR=0.05, EMBEDDING_LR=0.6,
            UNEMBEDDING_LR=0.004, WARMDOWN_RATIO=0.90, WEIGHT_DECAY=0.1, ROTARY_BASE=1000000,
            MLP_TAU=0.5, X0_LAMBDA_INIT=0.1, MUON_MOMENTUM=0.95, NS_STEPS=5)

def _cfg(**kw):
    base = dict(BASE); base.update(kw); return base

# warm-start: our confirmed champion-neighborhood (config, seed42 bpb) points
SEED_POINTS = [
    (_cfg(), 0.9350),
    (_cfg(NGRAM_MULT=40), 0.9367),
    (_cfg(MATRIX_LR=0.052), 0.9357),
    (_cfg(WARMDOWN_RATIO=0.92), 0.9364),
    (_cfg(NS_STEPS=6), 0.9363),
    (_cfg(EMBEDDING_LR=0.7), 0.9371),
    (_cfg(MODEL_DIM=512), 0.9371),
    (_cfg(MODEL_DIM=768), 0.9430),
    (_cfg(NGRAM_MULT=56), 0.9364),
    (_cfg(WEIGHT_DECAY=0.08, WARMDOWN_RATIO=0.88, MATRIX_LR=0.051), 0.9379),
]


def logmsg(m):
    line = time.strftime("[%Y-%m-%d %H:%M:%S] ") + m
    try:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print(line, flush=True)


# ---------- encode / decode ----------
def enc(cfg):
    u = np.empty(D)
    for i, (name, kind, a, b, sc) in enumerate(SPACE):
        v = cfg[name]
        if kind == "choice":
            try:
                idx = a.index(int(v))
            except Exception:
                idx = int(np.argmin([abs(int(v) - x) for x in a]))
            u[i] = (idx + 0.5) / len(a)
        elif sc == "log":
            u[i] = (math.log10(max(v, 1e-12)) - math.log10(a)) / (math.log10(b) - math.log10(a))
        else:
            u[i] = (v - a) / (b - a)
    return np.clip(u, 0.0, 1.0)


def dec(u):
    u = np.clip(u, 0.0, 1.0)
    cfg = dict(BASE)
    for i, (name, kind, a, b, sc) in enumerate(SPACE):
        if kind == "choice":
            idx = min(len(a) - 1, int(u[i] * len(a)))
            cfg[name] = a[idx]
        elif sc == "log":
            val = 10 ** (math.log10(a) + u[i] * (math.log10(b) - math.log10(a)))
            cfg[name] = int(round(val)) if kind == "int" else float(val)
        else:
            val = a + u[i] * (b - a)
            cfg[name] = int(round(val)) if kind == "int" else float(val)
    return cfg


# ---------- GP (RBF + EI minimization) ----------
def _phi(z):
    return np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _Phi(z):
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2)))


def gp_propose(hist, k):
    """Return k unit-cube points via GP-EI with local penalization. Robust fallbacks."""
    pts = hist[-GP_MAX:] if len(hist) > GP_MAX else hist
    X = np.array([enc(h["params"]) for h in pts])
    y = np.array([h["bpb"] for h in pts])
    rng = np.random.default_rng(1234 + len(hist))
    # candidate pool: uniform + perturbations around current best handful
    C = rng.random((5000, D))
    order = np.argsort(y)[:6]
    for j in order:
        C = np.vstack([C, np.clip(X[j] + rng.normal(0, 0.06, (250, D)), 0, 1)])
    try:
        ymu, ysd = y.mean(), y.std() + 1e-9
        yz = (y - ymu) / ysd
        l, noise = 0.22, 1e-3
        def rbf(A, B):
            return np.exp(-0.5 * ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1) / (l * l))
        Kn = rbf(X, X) + noise * np.eye(len(X))
        L = np.linalg.cholesky(Kn + 1e-6 * np.eye(len(X)))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, yz))
        Ks = rbf(C, X)
        mu = Ks @ alpha
        V = np.linalg.solve(L, Ks.T)
        var = np.clip(1.0 - (V * V).sum(0), 1e-9, None)
        sd = np.sqrt(var)
        ybest = yz.min()
        xi = 0.001
        imp = ybest - mu - xi
        z = imp / sd
        ei = np.where(sd > 0, imp * _Phi(z) + sd * _phi(z), 0.0)
        picks, eiw = [], ei.copy()
        for _ in range(k):
            bi = int(np.argmax(eiw))
            picks.append(C[bi])
            eiw = eiw * (1.0 - np.exp(-((C - C[bi]) ** 2).sum(1) / (2 * 0.12 ** 2)))
        return [p.copy() for p in picks]
    except Exception as e:
        logmsg("GP failed (%s) -> random batch" % repr(e))
        idx = rng.choice(len(C), k, replace=False)
        return [C[i].copy() for i in idx]


def propose(hist, k):
    rng = np.random.default_rng(777 + len(hist))
    n_bo = max(1, k - N_RANDOM)
    us = gp_propose(hist, n_bo)
    us += [rng.random(D) for _ in range(k - len(us))]
    return [dec(u) for u in us[:k]]


# ---------- run a config ----------
def batch_for(dim, depth):
    s = dim * depth
    if s <= 768 * 8:
        return 72, 1
    if s <= 768 * 11:
        return 36, 2
    return 24, 3


def launch(cfg, gpu):
    env = dict(os.environ)
    db, ga = batch_for(cfg["MODEL_DIM"], cfg["DEPTH"])
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu), "SEED": SEED,
        "HF_ENDPOINT": "https://hf-mirror.com", "OMP_NUM_THREADS": "8",
        "COMPILE_MODE": "max-autotune-no-cudagraphs",
        "DEVICE_BATCH_SIZE": str(db), "GRAD_ACCUM": str(ga),
        "TRITON_CACHE_DIR": "/tmp/triton_bo_%d" % gpu,
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor_bo_%d" % gpu,
    })
    for name in cfg:
        env[name] = str(cfg[name])
    logf = os.path.join(SOL, "bo_champ_g%d.log" % gpu)
    fh = open(logf, "w")
    p = subprocess.Popen([PY, "opt_param.py"], cwd=SOL, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return p, fh, logf


def parse_bpb(logf):
    v = None
    try:
        for line in open(logf):
            if line.startswith("val_bpb:"):
                try:
                    v = float(line.split()[1])
                except Exception:
                    pass
    except Exception:
        pass
    return v


# ---------- state ----------
def load_state():
    try:
        if os.path.exists(STATE):
            d = json.load(open(STATE))
            if isinstance(d, list) and d:
                return d
    except Exception as e:
        logmsg("load_state failed: %s" % repr(e))
    return [{"params": p, "bpb": b, "round": 0, "seed_pt": True} for p, b in SEED_POINTS]


def save_state(hist):
    try:
        tmp = STATE + ".tmp"
        json.dump(hist, open(tmp, "w"))
        os.replace(tmp, STATE)
    except Exception as e:
        logmsg("save_state failed: %s" % repr(e))


def compact(p):
    return ("d%d dim%d ng%d mlr%.3f elr%.2f uemb%.4f wd%.2f wdk%.2f rope%.0e tau%.2f x0%.3f mom%.3f" % (
        p["DEPTH"], p["MODEL_DIM"], p["NGRAM_MULT"], p["MATRIX_LR"], p["EMBEDDING_LR"],
        p["UNEMBEDDING_LR"], p["WARMDOWN_RATIO"], p["WEIGHT_DECAY"], p["ROTARY_BASE"],
        p["MLP_TAU"], p["X0_LAMBDA_INIT"], p["MUON_MOMENTUM"]))


# ---------- main loop ----------
def main():
    logmsg("=== BO orchestrator start | hours=%.2f | deadline=%s ===" % (
        HOURS, time.strftime("%H:%M:%S", time.localtime(DEADLINE))))
    hist = load_state()
    logmsg("history loaded: %d points (best=%.5f)" % (len(hist), min(h["bpb"] for h in hist)))
    rnd = max([h.get("round", 0) for h in hist] + [0])
    while time.time() < DEADLINE:
        rnd += 1
        try:
            cfgs = propose(hist, NGPU)
        except Exception as e:
            logmsg("propose crashed: %s\n%s" % (repr(e), traceback.format_exc()))
            rng = np.random.default_rng(rnd)
            cfgs = [dec(rng.random(D)) for _ in range(NGPU)]
        # launch all
        procs = []
        for i, cfg in enumerate(cfgs):
            try:
                p, fh, logf = launch(cfg, i)
                procs.append([cfg, p, fh, logf])
            except Exception as e:
                logmsg("launch fail gpu%d: %s" % (i, repr(e)))
        # wait with global timeout
        t0 = time.time()
        for rec in procs:
            cfg, p, fh, logf = rec
            rem = max(1.0, RUN_TIMEOUT - (time.time() - t0))
            try:
                p.wait(timeout=rem)
            except subprocess.TimeoutExpired:
                logmsg("timeout -> kill: %s" % compact(cfg))
                try:
                    p.kill()
                except Exception:
                    pass
            except Exception as e:
                logmsg("wait err: %s" % repr(e))
            try:
                fh.close()
            except Exception:
                pass
        # collect
        added = 0
        for cfg, p, fh, logf in procs:
            bpb = parse_bpb(logf)
            if bpb is None or not (0.5 < bpb < 2.0):
                bpb = PENALTY
            hist.append({"params": cfg, "bpb": float(bpb), "round": rnd})
            added += 1
        save_state(hist)
        valid = [h for h in hist if h["bpb"] < PENALTY - 1e-6]
        best = min(valid, key=lambda h: h["bpb"]) if valid else min(hist, key=lambda h: h["bpb"])
        hrs = (DEADLINE - time.time()) / 3600.0
        rbest = min([h["bpb"] for h in hist[-added:]]) if added else float("nan")
        logmsg("round %d done | n=%d (+%d) | round_best=%.5f | GLOBAL_BEST=%.5f | %.1fh left | %s" % (
            rnd, len(hist), added, rbest, best["bpb"], hrs, compact(best["params"])))
    logmsg("=== DEADLINE reached. rounds=%d n=%d ===" % (rnd, len(hist)))
    valid = [h for h in hist if h["bpb"] < PENALTY - 1e-6]
    best = min(valid, key=lambda h: h["bpb"])
    logmsg("FINAL BEST bpb=%.6f | %s" % (best["bpb"], compact(best["params"])))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logmsg("FATAL: %s\n%s" % (repr(e), traceback.format_exc()))
        sys.exit(1)
