#!/usr/bin/env python
"""Build rl/ldm_skyrl/skyrl_port_plan.ipynb.

Conventions carried over from the other notebook builders in this workspace:
figures before tables before prose; one reading sentence per figure; exact
numbers pushed into the trailing tables; 300 dpi with a 256-colour palette so
the executed notebook stays inside GitHub's inline-render budget; the notebook
reads committed JSON rather than recomputing or calling the network, so it and
the PR body it produces cannot drift apart.

All prose is English: the markdown cells are lifted verbatim into the PR body.
"""
from pathlib import Path

import nbformat as nbf

OUT = Path("/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/"
           "_wt_skyrl/rl/ldm_skyrl/skyrl_port_plan.ipynb")

NB = nbf.v4.new_notebook()


def md(s: str) -> None:
    NB.cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s: str) -> None:
    NB.cells.append(nbf.v4.new_code_cell(s.strip("\n")))


SETUP = r'''
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, display
from PIL import Image as PILImage

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 7.5,
    "axes.grid": True, "grid.alpha": .22, "grid.linewidth": .5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 8.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
})

INK, HL, CTRL, WARN, MUTE = "#22303f", "#c1442e", "#5b7c99", "#d99b28", "#9aa7b1"

HERE = Path.cwd()
FACTS = json.loads((HERE / "facts.json").read_text())
PHASES = json.loads((HERE / "phases.json").read_text())
PA = json.loads((HERE / "phase_a_results.json").read_text())
FIGDIR = HERE / "figs"; FIGDIR.mkdir(exist_ok=True)

def show(fig, name):
    """Save at 300 dpi, quantise to a 256-colour palette, display inline.

    The quantise step happens here rather than as a post-process on the .ipynb
    because `nbconvert --execute --inplace` would re-run the cell and silently
    undo anything applied to the file afterwards.
    """
    p = FIGDIR / f"{name}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    im = PILImage.open(p).convert("RGB").quantize(colors=256, method=PILImage.MEDIANCUT)
    im.save(p, optimize=True)
    print(f"{name}.png  {p.stat().st_size/1024:.0f} KB")
    display(Image(filename=str(p)))

print(f"facts collected {FACTS['collected_utc']} | ldm_rl read at {FACTS['ldm_rl_ref'][:8]}")
'''

F1 = r'''
ps = sorted(FACTS["port_surface"], key=lambda r: r["lines"])
names = [r["file"] for r in ps]
lines = [r["lines"] for r in ps]
coupled = [r["coupled_lines"] for r in ps]
y = np.arange(len(names))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True,
                             gridspec_kw={"width_ratios": [2.1, 1]})
a1.barh(y, lines, color=CTRL, height=.68, edgecolor="none")
for yi, ln in zip(y, lines):
    a1.text(ln + 14, yi, str(ln), va="center", fontsize=6.6, color=INK)
a1.set_xlim(0, max(lines) * 1.18)
a1.set_yticks(y); a1.set_yticklabels(names)
a1.set_xlabel("lines in the module")
a1.set_title("how big each module is", loc="left")

a2.barh(y, coupled, color=[HL if c >= 10 else (CTRL if c else MUTE) for c in coupled],
        height=.68, edgecolor="none")
for yi, c in zip(y, coupled):
    a2.text(c + .35, yi, str(c), va="center", fontsize=6.6, color=INK)
a2.set_xlim(0, max(coupled) * 1.28)
a2.set_xlabel("framework-naming lines")
a2.set_title("how many of them name slime /\nsglang / megatron", loc="left")

fig.suptitle("F1  Where the framework coupling actually is in rl/ldm_rl",
             x=0.005, ha="left", fontsize=8.5)
fig.tight_layout(rect=(0, 0, 1, .93))
show(fig, "f1_port_surface")

tot_l = sum(lines); tot_c = sum(coupled)
bridge = next(r for r in ps if r["file"] == "bridge.py")
env = next(r for r in ps if r["file"] == "env.py")
print(f"{tot_l} lines total, {tot_c} of them name a framework.")
print(f"bridge.py: {bridge['lines']} lines carrying {bridge['coupled_lines']} of those {tot_c}.")
print(f"env.py:    {env['lines']} lines carrying {env['coupled_lines']}.")
'''

F2 = r'''
cc = FACTS["cuda_constraint"]
pts = [
    ("LDM slime stack (today)", 12.8, True),
    ("SkyRL 0.3.0 (b8a5caaa)", 12.9, True),
    ("SkyRL main", 13.0, False),
]
fig, ax = plt.subplots(figsize=(6.4, 2.1))
ax.axvspan(12.4, 12.95, color=CTRL, alpha=.13)
ax.axvspan(12.95, 13.35, color=HL, alpha=.13)
for k, (label, ver, ok) in enumerate(pts):
    ax.scatter([ver], [k], s=95, color=(CTRL if ok else HL), zorder=3,
               edgecolor="white", lw=.8)
    dx, ha = ((.03, "left") if not ok else (-.03, "right"))
    ax.text(ver + dx, k, label, ha=ha, va="center", fontsize=7.4, color=INK)
ax.axvline(12.95, color=INK, lw=1.0, ls="--")
ax.text(12.965, -.55, "CUDA 13 needs driver r580;\nthis cluster has 565",
        fontsize=6.8, color=INK, va="bottom")
ax.text(12.425, -.55, "CUDA 12.x runs on a 12.7 driver\n(minor-version compatibility)",
        fontsize=6.8, color=INK, va="bottom")
ax.set_yticks([]); ax.set_ylim(-.95, 2.6)
ax.set_xlim(12.4, 13.35); ax.set_xticks([12.5, 12.7, 12.8, 12.9, 13.0, 13.2])
ax.set_xlabel("CUDA toolkit version the wheels are built against")
ax.grid(axis="y", alpha=0)
ax.set_title(f"F2  Driver {cc['isambard_driver']} puts the boundary between 12.x and 13.x",
             loc="left")
show(fig, "f2_cuda_red_line")

for r in cc["rows"]:
    print(("FEASIBLE  " if r["feasible"] else "BLOCKED   ") + r["target"]
          + f"   (needs CUDA {r['needs_cuda']}, driver {r['needs_driver']})")
'''

F3 = r'''
zv = FACTS["zero_variance"]["rows"]
n = [r["n_samples"] for r in zv]
g = [r["zero_var_groups_per_step"] for r in zv]
steps = [r["steps"] for r in zv]
cost = [x / n[0] for x in n]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.5))

a1.plot(n, g, "o-", color=HL, lw=1.4, ms=5)
for k, (xi, yi, si) in enumerate(zip(n, g, steps)):
    dx, ha = (10, "left") if k == 0 else (0, "center")
    a1.annotate(f"{yi:.2f}\n{si} steps", (xi, yi), textcoords="offset points",
                xytext=(dx, 9), ha=ha, fontsize=6.4, color=INK)
a1.set_xscale("log", base=2); a1.set_xticks(n); a1.set_xticklabels(n)
a1.set_ylim(-.12, 1.30)
a1.set_xlabel("n_samples_per_prompt"); a1.set_ylabel("zero-variance groups per step")
a1.set_title("measured on the slime line (PR #2)", loc="left")

a2.plot(n, cost, "s-", color=CTRL, lw=1.4, ms=5, label="oversampling: rollout cost")
a2.axhline(1.0, color=WARN, lw=1.4, ls="--", label="grpo_norm_by_std=false: no extra rollouts")
a2.set_xscale("log", base=2); a2.set_xticks(n); a2.set_xticklabels(n)
a2.set_xlabel("n_samples_per_prompt"); a2.set_ylabel("rollout cost, relative to n=2")
a2.set_title("what each remedy costs", loc="left")
a2.legend(frameon=False, loc="upper left")
fig.suptitle("F3  Zero-variance groups: the slime line bought this with rollouts",
             x=0.005, ha="left", fontsize=8.5)
fig.tight_layout(rect=(0, 0, 1, .94))
show(fig, "f3_zero_variance")

print("arm step counts are unequal:", dict(zip(n, steps)))
print("the two 0.00 readings rest on 24 and 15 steps respectively, not on equal power.")
'''

F4 = r'''
gp = FACTS["gp_cost"]
obs = gp["step_seconds_observed"]
xs = np.array([0, 1, 2]); ys = np.array([obs["0_evals"], obs["1_eval"], obs["2_evals"]], float)
slope, intercept = np.polyfit(xs, ys, 1)
full = gp["gp_calls_per_step_all_success"]
grid = np.linspace(0, full, 200)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.5))

a1.plot(grid, intercept + slope * grid, color=MUTE, lw=1.2, ls="--",
        label=f"extrapolated: {intercept:.0f} + {slope:.0f}s per evaluation")
a1.plot(xs, ys, "o", color=HL, ms=6, label="measured (3 points)")
a1.axvline(full, color=WARN, lw=1.1, ls=":")
a1.annotate(f"a fully successful step issues {full}\n"
            f"-> {(intercept + slope*full)/3600:.1f} h per step, unoverlapped",
            xy=(full, intercept + slope * full), xycoords="data",
            xytext=(0.97, 0.06), textcoords="axes fraction",
            ha="right", va="bottom", fontsize=6.6, color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTE, lw=.6))
a1.annotate("19 / 85 / 156 s", (1, ys[1]), textcoords="offset points",
            xytext=(10, -2), fontsize=6.6, color=HL)
a1.set_xlabel("successful evaluations in the step"); a1.set_ylabel("step wall-clock, seconds")
a1.set_title("the measured cost, and where it goes", loc="left")
a1.legend(frameon=False, loc="upper left")

conc = np.array([1, 2, 4, 8, 16])
hours = (intercept + slope * full / conc) / 3600
a2.plot(conc, hours, "o-", color=CTRL, lw=1.4, ms=5)
a2.axvline(8, color=WARN, lw=1.1, ls=":")
a2.annotate("docking saturates\nat 8 workers", (8, hours[3]), textcoords="offset points",
            xytext=(6, 12), fontsize=6.6, color=INK)
a2.set_xscale("log", base=2); a2.set_xticks(conc); a2.set_xticklabels(conc)
a2.set_xlabel("concurrent evaluations"); a2.set_ylabel("modelled hours per step")
a2.set_title("arithmetic, not a measurement", loc="left")
fig.suptitle("F4  The reward is the expensive half of the step",
             x=0.005, ha="left", fontsize=8.5)
fig.tight_layout(rect=(0, 0, 1, .94))
show(fig, "f4_gp_wall")

print(f"fit: {intercept:.1f}s fixed + {slope:.1f}s per evaluation "
      f"(the line's own estimate of a single GP call is {gp['seconds_per_gp_call_sk']:.0f}s)")
print(f"at {full} evaluations and 8-way concurrency: {(intercept + slope*full/8)/60:.0f} min per step")
'''

F5 = r'''
ab = FACTS["ablation"]["rows"]
labels = [r["run"] for r in ab][::-1]
vals = np.array([r["mean_reward"] for r in ab][::-1])
sds = np.array([r["sd"] for r in ab][::-1])
adopt = [r["adopt"] for r in ab][::-1]
base = next(r["mean_reward"] for r in ab if r["run"].startswith("Untrained"))

fig, ax = plt.subplots(figsize=(6.4, 3.2))
y = np.arange(len(labels))
colors = [HL if a else (MUTE if l.startswith("Untrained") else CTRL)
          for a, l in zip(adopt, labels)]
ax.errorbar(vals, y, xerr=sds, fmt="none", ecolor=INK, elinewidth=.8, capsize=2)
ax.scatter(vals, y, s=44, c=colors, zorder=3, edgecolor="white", lw=.7)
ax.axvline(base, color=WARN, lw=1.1, ls="--")
ax.text(base + .18, 1.1, "untrained model", fontsize=6.6, color=INK,
        ha="left", va="center")
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.4)
ax.set_ylim(-.8, len(labels) - .2)
ax.set_xlabel("mean reward, % (3 passes over 480 held-out tasks; bars are 1 sd)")
ax.set_title("F5  The recipe's own ablation, on the recipe's own benchmark\n"
             "red = the settings this port adopts as phase C candidates", loc="left")
show(fig, "f5_ablation")

trained = [v for v, l in zip(vals, labels) if not l.startswith("Untrained")]
print(f"top row beats the untrained model by {vals[-1]-base:.2f} points; "
      f"the spread across the trained rows is {max(trained)-min(trained):.2f}, "
      f"against a typical 1-sd bar of {np.median(sds):.2f}.")
print("This is Qwen3.6-35B-A3B on knowledge-work tasks, not molecules: it ranks candidates, it does not transfer a result.")
'''

F6 = r'''
items = {i["id"]: i for i in PHASES["items"]}
level = {}
def lvl(i):
    if i in level: return level[i]
    d = items[i]["deps"]
    level[i] = 1 + max((lvl(x) for x in d), default=0)
    return level[i]
for i in items: lvl(i)

phases = ["A", "B", "C", "D"]
prow = {p: k for k, p in enumerate(phases)}
pcol = {"A": HL, "B": CTRL, "C": WARN, "D": INK}
maxlvl = max(level.values())

fig, ax = plt.subplots(figsize=(6.6, 2.6))
pos = {}
for p in phases:
    ids = sorted((i for i in items if items[i]["phase"] == p), key=lambda i: (level[i], i))
    bylvl = {}
    for i in ids:
        bylvl.setdefault(level[i], []).append(i)
    for L, group in bylvl.items():
        for k, i in enumerate(group):
            pos[i] = (L, prow[p] + (k - (len(group) - 1) / 2) * 0.19)
for i, it in items.items():
    for d in it["deps"]:
        (x0, y0), (x1, y1) = pos[d], pos[i]
        ax.annotate("", xy=(x1 - .045, y1), xytext=(x0 + .045, y0),
                    arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=.55,
                                    shrinkA=0, shrinkB=0, alpha=.85))
for i, (x, y) in pos.items():
    ax.scatter([x], [y], s=170, color=pcol[items[i]["phase"]], zorder=3, edgecolor="white", lw=.8)
    ax.text(x, y, i, ha="center", va="center", color="white", fontsize=5.9,
            fontweight="bold", zorder=4)
ax.set_yticks(list(prow.values()))
ax.set_yticklabels([f"{p}  {PHASES['items'][0]['phase'] and ''}" for p in phases])
ax.set_yticklabels(["A  stack", "B  wiring", "C  algorithm", "D  objective"])
ax.invert_yaxis()
ax.set_xticks(range(1, maxlvl + 1))
ax.set_xlabel("dependency level (everything on one level can run at the same time)")
ax.set_xlim(.55, maxlvl + .45); ax.grid(axis="y", alpha=0)
ax.set_title("F6  Nineteen items, and what actually blocks what", loc="left")
show(fig, "f6_phase_dag")

width = {}
for i in items: width.setdefault(level[i], []).append(i)
for L in sorted(width):
    print(f"level {L}: {len(width[L])} items in parallel -> {' '.join(sorted(width[L]))}")
b_deps_on_a = [i for i in items if items[i]["phase"] == "B"
               and any(items[d]["phase"] == "A" for d in items[i]["deps"])]
print(f"\nphase B items that wait on phase A: {b_deps_on_a or 'none'}")
'''

T1 = r'''
import pandas as pd
rows = [
 ("zero-variance groups make GRPO's (r-mean)/std into 0/0",
  "raise n_samples to 8, costing 4x the rollouts",
  "grpo_norm_by_std=false removes the division; zero_variance_filter masks the group",
  "C1, C2"),
 ("a GP call takes 67 s and a full step issues up to 80 of them",
  "none; the cost sits on the critical path of every step",
  "FullyAsyncRayPPOTrainer overlaps rollout with training; the rate limiter bounds concurrency",
  "C4, B3"),
 ("9B gradients for linear_attn.A_log and dt_bias are all non-finite",
  "seven hypotheses excluded by measurement; unresolved",
  "the recipe documents a same-family GDN backward failure with a named cause (the tilelang JIT missing CUDA headers)",
  "C5"),
 ("50 of 111 runs died to GPU memory taken by neighbours",
  "verify each card is empty before launching",
  "nothing: this is a scheduler-level problem and both lines have it equally",
  "unchanged"),
]
T = pd.DataFrame(rows, columns=["measured on the slime line", "what the slime line does",
                                "what SkyRL 0.3.0 offers", "phase items"])
display(T.style.hide(axis="index").set_properties(**{"text-align": "left", "font-size": "11px"}))
'''

T2 = r'''
reg = pd.DataFrame([{"id": i["id"], "phase": i["phase"], "item": i["title"],
                     "criterion": i["criterion"], "budget": i["budget"],
                     "depends on": ", ".join(i["deps"]) or "-"}
                    for i in PHASES["items"]])
display(reg.style.hide(axis="index").set_properties(**{"text-align": "left", "font-size": "10px"}))
'''

F7 = r'''
dl = PA["delivery"]
built = {k.replace("_", "-"): v for k, v in PA["already_built_here"]["versions"].items() if v}

def bucket(d):
    if "prebuilt" in d["delivery"]: return 0
    if d["package"] in built:       return 1
    return 2

rows = sorted(dl, key=lambda d: (bucket(d), d["package"]))
cols = [CTRL, WARN, HL]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 3.1), gridspec_kw={"width_ratios": [1.45, 1]})

y = np.arange(len(rows))[::-1]
for yi, d in zip(y, rows):
    a1.barh(yi, 1, color=cols[bucket(d)], height=.7, edgecolor="none")
    a1.text(0.015, yi, d["package"], va="center", ha="left", color="white",
            fontsize=6.6, fontweight="bold")
    a1.text(0.985, yi, d["version"], va="center", ha="right", color="white", fontsize=6.2)
a1.set_yticks([]); a1.set_xticks([]); a1.grid(False)
for sp in a1.spines.values(): sp.set_visible(False)
a1.set_title("how each package reaches an aarch64 machine", loc="left")
handles = [plt.Rectangle((0,0),1,1,color=c) for c in cols]
a1.legend(handles, ["prebuilt aarch64 wheel", "compiles; already built here", "compiles; not built yet"],
          loc="upper center", bbox_to_anchor=(.5, -.02), ncol=1, frameon=False, fontsize=6.2)

STAGES = ["create venv", "install torch", "install vllm", "import + cuda", "emit a token"]
smap = {"create venv": 0, "install torch": 1, "install vllm": 2, "generate one token": 4, "passed": 5}
att = PA["a2"]["attempts"]
ya = np.arange(len(att))[::-1]
for yi, a in zip(ya, att):
    reached = smap[a["stopped_at"]]
    ok = a["stopped_at"] == "passed"
    a2.barh(yi, reached, color=(CTRL if ok else MUTE), height=.6, edgecolor="none")
    if not ok:
        a2.scatter([reached], [yi], marker="x", s=34, color=HL, zorder=3, linewidths=1.3)
a2.set_yticks(ya); a2.set_yticklabels([f"job {a['job']}" for a in att], fontsize=6.2)
a2.set_xticks(range(len(STAGES) + 1)); a2.set_xticklabels([""] + STAGES, rotation=38, ha="right", fontsize=6.0)
a2.set_xlim(0, 5.15)
a2.set_title("A2, attempt by attempt (x marks where it stopped)", loc="left")
fig.suptitle("F7  What phase A measured", x=0.005, ha="left", fontsize=8.5)
fig.tight_layout(rect=(0, 0, 1, .93))
show(fig, "f7_phase_a")

n = [sum(1 for d in dl if bucket(d) == k) for k in range(3)]
print(f"{len(dl)} heavy packages: {n[0]} arrive as aarch64 wheels, {n[1]} would compile but are "
      f"already built in {PA['already_built_here']['env']}, {n[2]} are neither.")
print("still to build:", [d["package"] for d in dl if bucket(d) == 2])
print()
for a in att:
    print(f"  job {a['job']}  " + ("PASS" if a["stopped_at"] == "passed" else f"stopped at {a['stopped_at']}"))
    if a["cause"]: print(f"      {a['cause']}")
'''

T3 = r'''
a1r, a2r = PA["a1"], PA["a2"]
reg = pd.DataFrame([
 {"item": "A1", "criterion": a1r["criterion"], "result": a1r["status"],
  "evidence": f"{a1r['packages_resolved']} packages resolved; torch {a1r['torch_version']} "
              f"({a1r['torch_delivery']}); 'cu130' appears {a1r['cu130_occurrences']} times in the lock"},
 {"item": "A2", "criterion": "import vllm succeeds and a model emits one token on one GH200",
  "result": a2r["status"],
  "evidence": f"job {a2r['job_id']} on {a2r['node']}: torch {a2r['torch']}, vllm {a2r['vllm']}, "
              f"cuda available {a2r['cuda_available']}, {a2r['device']}; "
              f"model load {a2r['model_load']}; token emitted {a2r['token_text']}"},
 {"item": "A6", "criterion": "name the attention backend vLLM selects on aarch64", "result": "PASS",
  "evidence": f"{a2r['attention_backend']}, chosen out of {a2r['backend_candidates']}; "
              f"flashinfer {a2r['flashinfer']} -- it is not absent on aarch64"},
 {"item": "A3", "criterion": "trainer.strategy=megatron reaches the first forward", "result": "not run",
  "evidence": "next; should reuse the already-built stack rather than install from the lock"},
 {"item": "A4", "criterion": "engine output changes after one NCCL weight sync", "result": "not run", "evidence": ""},
 {"item": "A5", "criterion": "two training steps and a checkpoint, constant-reward generator", "result": "not run", "evidence": ""},
])
display(reg.style.hide(axis="index").set_properties(**{"text-align": "left", "font-size": "10px"}))
'''

F8 = r'''
B = PA["blocked"]
sch, cap = B["scheduling"], B["capacity"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.7), gridspec_kw={"width_ratios": [1.25, 1]})

# Left: the predicted start does not care how small the request is.
labels = [r["request"] for r in sch["rows"]][::-1]
import datetime as _dt
now = _dt.datetime(2026, 9, 9, 18, 24)
dead = _dt.datetime(2026, 9, 10, 5, 0)
hrs = [( _dt.datetime.strptime(r["predicted_start"], "%Y-%m-%dT%H:%M") - now).total_seconds()/3600
       for r in sch["rows"]][::-1]
y = np.arange(len(labels))
a1.barh(y, hrs, color=HL, height=.62, edgecolor="none")
a1.axvline((dead - now).total_seconds()/3600, color=INK, lw=1.3, ls="--")
a1.text((dead - now).total_seconds()/3600 + 0.7, -0.42, "deadline 05:00Z",
        fontsize=6.6, color=INK, ha="left", va="center")
for yi, h in zip(y, hrs):
    a1.text(h - 1.2, yi, f"{h:.0f} h", va="center", ha="right", color="white", fontsize=6.6)
a1.set_yticks(y); a1.set_yticklabels(labels, fontsize=6.4)
a1.set_xlabel("hours from submission to predicted start (sbatch --test-only)")
a1.set_title("resizing the request does not move it", loc="left")

# Right: what Slurm's step view claims free, against what the cards report.
vals = [cap["nodes_with_no_slurm_step"], cap["nodes_probed_with_nvidia_smi"],
        cap["nodes_found_with_a_free_card"]]
names = ["no Slurm step\n(looks free)", "probed with\nnvidia-smi", "actually had\na free card"]
a2.bar(range(3), vals, color=[MUTE, CTRL, HL], width=.62, edgecolor="none")
for i, v in enumerate(vals):
    a2.text(i, v + 2, str(v), ha="center", fontsize=7.2, color=INK)
a2.set_xticks(range(3)); a2.set_xticklabels(names, fontsize=6.2)
a2.set_ylim(0, max(vals) * 1.22)
a2.set_ylabel("nodes")
a2.set_title(f"median {cap['median_used_mib_on_stepless_nodes']/1024:.0f} GiB per card\non the stepless ones", loc="left")

fig.suptitle("F8  Why A3, A4 and A5 have not run", x=0.005, ha="left", fontsize=8.5)
fig.tight_layout(rect=(0, 0, 1, .93))
show(fig, "f8_capacity")

print(f"account: {sch['account_running_jobs']} running jobs, {sch['account_pending_jobs']} pending, "
      f"{sch['account_nodes_held']} nodes held; a newly submitted job gets priority {sch['submitted_job_priority']}.")
print(f"{cap['nodes_with_no_slurm_step']} of {cap['account_nodes_allocated']} allocated nodes carry no step; "
      f"{cap['nodes_probed_with_nvidia_smi']} were probed and {cap['nodes_found_with_a_free_card']} had a free card.")
print(cap["reason"])
'''

T4 = r'''
st = pd.DataFrame([{"what the staging turned up": r["finding"],
                    "consequence": r["consequence"],
                    "how it presents": r["misleading_symptom"] or "-"}
                   for r in PA["blocked"]["staging"]])
display(st.style.hide(axis="index").set_properties(**{"text-align": "left", "font-size": "10px"}))
'''

T5 = r'''
a5, a4, a3 = PA["a5"], PA["a4"], PA["a3"]
res = pd.DataFrame([
 {"item": "A5", "result": a5["status"], "criterion": a5["criterion"],
  "evidence": f"{a5['when_utc']}, {a5['job']} on {a5['node']}, {a5['gpus']} GPU. {a5['path']}. "
              f"{a5['batch']}. Checkpoints {', '.join(a5['checkpoints'])} containing "
              f"{a5['checkpoint_contents']}. "
              + ", ".join(f"{k} {v}" for k, v in a5["metrics"].items())},
 {"item": "A4", "result": a4["status"], "criterion": a4["criterion"],
  "evidence": f"OBSERVED: {a4['observed']}. NOT OBSERVED: {a4['not_observed']}"},
 {"item": "A3", "result": a3["status"], "criterion": a3["criterion"],
  "evidence": a3["root_cause"] + " LAYERS: " + " | ".join(a3["layers"])
              + " CLEARED AND VERIFIED IN RUN: " + "; ".join(a3["cleared_and_verified_in_run"])},
])
display(res.style.hide(axis="index").set_properties(**{"text-align": "left", "font-size": "10px"}))
print(f"A5 passed on attempt {a5['attempts_before_passing'] + 1}.")
'''


# The reader and the data live in two files and only one of them gets edited.
import json as _json
_PA_CHECK = _json.loads((OUT.parent / "phase_a_results.json").read_text())
_REQUIRED = {"a5": ("status","criterion","when_utc","job","node","gpus","path","batch",
                    "checkpoints","checkpoint_contents","metrics","attempts_before_passing"),
             "a4": ("status","criterion","observed","not_observed"),
             "a3": ("status","criterion","root_cause","layers","cleared_and_verified_in_run")}
for _k, _fields in _REQUIRED.items():
    _missing = [f for f in _fields if f not in _PA_CHECK.get(_k, {})]
    if _missing:
        raise SystemExit(f"phase_a_results.json['{_k}'] is missing {_missing} -- "
                         f"the notebook cell that reads it would fail inside nbconvert instead")

# ----------------------------------------------------------------- assembly
md(r"""
# A second RL line for LDM, on SkyRL

The small-molecule acquisition RL line has one number it exists to produce:
Pareto hypervolume at budget 80, on G12D and on G12C, for SFT+RL against
SFT-only. That number does not exist yet. What stands between the line and it is
not the evaluation stack but the training: across seven 9B runs with
checkpoints, every one of 238 recorded gradient norms was non-finite, and 37 of
269 optimizer steps took effect.

This is a plan to stand a second training line beside the existing one, built on
[SkyRL](https://github.com/NovaSky-AI/SkyRL) and shaped by
[ApexAgents-SkyRL-Recipe](https://github.com/Mercor-Intelligence/ApexAgents-SkyRL-Recipe),
which trains knowledge-work agents up to 397B on released SkyRL with no forks.
slime keeps running. The two lines are judged against each other on the same
number, through the same evaluation stack.

Everything below is generated by
[`skyrl_port_plan.ipynb`](rl/ldm_skyrl/skyrl_port_plan.ipynb), which reads
frozen JSON committed alongside it rather than recomputing or calling the
network, so this text and the notebook cannot drift apart.
""")

code(SETUP)

md(r"""
## 1. The port is one file wide

`rl/ldm_rl` was written against contracts rather than against slime: its own
module docstring says the environment "depends only on `ldm_tts` contracts,
never on Slime or a specific task". That claim is checkable, and it holds.
""")
code(F1)
md("**F1.** The environment is 803 lines and names a framework on one of them; "
   "`bridge.py` carries 16 of the 30 coupled lines in the package, so replacing "
   "the framework means replacing one adapter rather than rewriting the loop.")

md(r"""
## 2. Only one SkyRL version can run on this cluster

This is the constraint that decides the pin, and it is worth stating before
anything else, because it invalidates the obvious choice. SkyRL `main` requires
CUDA 13.0 and an r580 driver. This cluster runs driver 565, which is CUDA 12.7,
and the line already knows what a cu130 build does here: it installs, and then
`torch.cuda.is_available()` returns False.

The recipe pins SkyRL 0.3.0 at commit `b8a5caaa`, whose wheels are cu128 and
cu129. That is inside the red line.
""")
code(F2)
md("**F2.** The pin is not a preference: SkyRL 0.3.0 is the only version of the "
   "two that this driver can run at all.")

md(r"""
SkyRL 0.3.0's inference backend is vLLM and nothing else
(`config.py:743`, `backend: str = \"vllm\"`), so the existing sglang deployment
does not carry over and the port depends on a vLLM aarch64 wheel existing at the
pinned version. It does.
""")
code(r'''
v = FACTS["vllm_aarch64"]
print("index:", v["recipe_index"])
for w in v["wheels"]:
    print("  ", w)
print("\naarch64 wheel present on the pinned index:", v["aarch64_present"])
print("\nrecent vLLM releases on PyPI:")
for r in v["pypi_recent"]:
    print(f"   {r['version']:<8} aarch64={r['aarch64']}  x86_64={r['x86_64']}")
''')
md("The single largest technical risk in this port was whether SkyRL's only "
   "inference backend has a build for this architecture. It does, at the exact "
   "pinned version, on the exact index the recipe uses. That retires the risk at "
   "the wheel level only: whether it *runs* on driver 565 is phase item A2.")

md(r"""
## 3. Two failures the slime line measured, and what SkyRL does about them

Two distinct non-finite-gradient stories on this line, which should not be run
together. The 1.5B one came from zero-variance groups turning `(r - mean) / std`
into `0/0`; its root cause was a prompt that showed a JSON Schema while asking
for an instance, and it is fixed. The 9B one is different: the gradients of
`linear_attn.A_log` and `dt_bias` are all non-finite while a later layer's are
finite, so the values are generated inside particular layers rather than
propagated back into them. Seven hypotheses have been excluded by measurement.
""")
code(F3)
md("**F3.** Raising `n_samples` from 2 to 8 does remove the zero-variance groups, "
   "and it costs four times the rollouts to do it. `grpo_norm_by_std=false` "
   "removes the division instead, which costs nothing. `zero_variance_filter` is "
   "not free in the same way: it masks the group rather than manufacturing "
   "signal, so refilling the mini-batch still takes extra rollouts.")

md(r"""
## 4. The reward, not the model, is what makes a step slow
""")
code(F4)
md("**F4.** The three measured points fit a line at roughly 67 seconds per "
   "evaluation, which is the line's own independent estimate of a single GP "
   "call; everything to the right of them is arithmetic, not measurement. Under "
   "the synchronous trainer this sits on the critical path, which is the whole "
   "case for the fully async one, and phase item C4 is what turns that case into "
   "a number.")

md(r"""
## 5. Which algorithm settings are worth trying first
""")
code(F5)
md("**F5.** These are the recipe's results on the recipe's own benchmark: a 35B "
   "mixture-of-experts model on knowledge-work tasks, not a 9B model proposing "
   "molecules. The table is useful for ordering what to try, and it is not "
   "evidence about this domain. Phase C is where it becomes evidence or stops "
   "being interesting.")

md(r"""
## 6. What can be done at the same time

Grouping work into phases is only useful if the grouping says something true
about what blocks what. The register below is a dependency graph, and the levels
are computed from it rather than asserted.
""")
code(F6)
md("**F6.** The result worth acting on: no phase-B item depends on any phase-A "
   "item. The entire domain-wiring phase is CPU-only and can be written and "
   "tested on a login node while phase A is still deciding whether the stack "
   "installs. Phase A is a chain rather than a fan-out, which is a property of "
   "the stack, not a scheduling choice.")

md(r"""
## 7. Failure to mechanism to phase item
""")
code(T1)

md(r"""
## 8. The register

Every item carries a criterion written down before the measurement, the artifact
it must leave behind, its budget, and its dependencies.
""")
code(T2)

md(r"""
## 9. Phase A, measured

A1, A2 and A6 have been run on this cluster. A3, A4 and A5 have not.
""")
code(F7)
md("**F7.** The left panel is what moves phase A: of the packages with no "
   "prebuilt aarch64 wheel, most are already compiled here, because the slime "
   "line built them. The route is to add vLLM and SkyRL to that stack rather "
   "than install this lock from nothing. The right panel is A2's five attempts; "
   "all four failures were in the probe or the install command and none in the "
   "stack, and each stopped within minutes because the install forbids source "
   "builds.")
code(T3)
md(r"""
Two of the four A2 failures are worth keeping rather than tidying away, because
both are traps any later phase item can fall into.

`uv`'s index strategy is first-match: it stops at the first index carrying a
package *name*. `torch` exists on PyPI, so the pytorch index named one line
above was never consulted, and the error read "no version of
torch==2.11.0+cu128" while that wheel sat on that index. A name being present is
not a version being present.

flashinfer JIT-compiles its sampling kernel on first use, and the only `nvcc`
inside a fresh venv is `site-packages/nvidia/cu13/bin/nvcc` -- a CUDA 13
compiler, arriving as an ordinary dependency, on a cluster whose driver is CUDA
12.7. The recipe records the same shape of trap for tilelang's gated-delta-net
kernel. `CUDA_HOME` now points at the 12.9 toolkit the slime line already built
against. This one reaches past A2: a runtime JIT that quietly picks the cu13
compiler is the CUDA red line being crossed from inside an environment that
otherwise looks fine.
""")

md(r"""
## 9b. What blocked A3, A4 and A5 for most of the night

They are implemented, staged and import-checked. They have not run, and the
reason is cluster capacity rather than anything about the port.
""")
code(F8)
md("**F8.** Left: the same script at four sizes, from one GPU for an hour to a "
   "whole node for eight, all predicted to start about two days out; shrinking "
   "the request does not help because a newly submitted job gets priority 1 "
   "behind this account's own queue. Right: Slurm's step view lists 101 "
   "allocated nodes with nothing running on them, and of the 61 probed with "
   "nvidia-smi exactly one had a free card -- the rest hold about 90 GiB per "
   "card. A step that has vanished from `squeue` is not an idle node; it is "
   "usually a live process whose srun client died with its session.")

md(r"""
Four things the staging turned up on the way. None of them is a phase item, and
each cost a wrong turn, so they are recorded rather than smoothed over.
""")
code(T4)
md("Two of these share a shape worth naming: **the error message points at a "
   "layer that is not the one that broke.** A full filesystem reports itself as "
   "a corrupt package archive, and a requirement line that installed nothing "
   "reports itself, fifteen minutes later, as a missing module inside the "
   "trainer. Both invite a fix to the wrong thing.")

md(r"""
## 10. A5 passed; A4 is half-measured; A3 failed with a named cause

The gate that matters is closed. A5 runs SkyRL's own `examples/train/gsm8k` path
with `strategy=fsdp`, colocated, on a local Qwen2.5-1.5B-Instruct, with **no LDM
code anywhere on it** -- so it is a statement about the stack rather than about
the port. Two GRPO steps completed and two checkpoints were written on
GH200 / aarch64 / driver 565.
""")
code(T5)
md(r"""
**A4 is recorded as partial on purpose.** Its criterion has two halves -- that a
sync happens, and that the engine's output changes because of it -- and only the
first was measured. The payload had originally asserted A4 from the fact that
colocated training necessarily syncs weights, which is an inference standing in
where an observation belongs; the `timing/sync_weights` lines replaced that
inference, and the second half stays open.

**Seven landings were needed, and six of the failures were in the harness rather
than the stack.** They are worth listing because they were six different shapes,
not one problem hit six times:

| # | died on | what kind of mistake |
|---|---|---|
| 1 | `ModuleNotFoundError: jaxtyping` | hand-written dependency list, missing an entry |
| 2 | `ModuleNotFoundError: peft` | the same list, a second entry |
| 3 | `ModuleNotFoundError: vllm_router` | a *second* hand-written list -- the "heavy, skip it" exclusions |
| 4 | guard aborted: every card taken | the guard working; a probe from 20 minutes earlier had gone stale |
| 5 | `42 not divisible by 4` | making the GPU count a variable broke the batch arithmetic |
| 6 | `RouterArgs has no attribute pd_disaggregation` | installed by name, bypassing the version the A1 lock had already resolved |

Failures 1 to 3 are one defect wearing three faces: **a hand-curated list is a
negative assertion -- "nothing else is needed" -- with no declared coverage.**
Replacing the dependency list with SkyRL's own `[project.dependencies]` fixed one
of the two lists and left the other sitting beside it.

What replaced both: walk every module in the package with `pkgutil`, collect
every missing top-level name in one pass, install, repeat until empty. That found
`flax`, `jax`, `sqlalchemy`, `sqlmodel` and `vllm_router` together -- four of
which had never appeared on any of my lists.

Then failure 6 happened twice, and the second time is the instructive one. After
the first, the fix was added *to the sweep's name mapping* -- and the sweep only
fires on a module that is **missing**. `vllm-router` 0.1.15 imports perfectly and
fails at run time, so the sweep found nothing to do and never reached the fix.
**A probe that detects absence cannot detect wrongness, and a clean sweep over an
installed-but-wrong package looks exactly like a correct environment.** The pin
is now unconditional, runs before the sweep, and asserts that the attribute
exists rather than comparing a version string.

## 10b. A3 is five layers deep, and each one hid the next

A3 failed. The verdict is the least useful part; the structure is the product.

```
1  the python packages install and import -- only the CUDA extensions are absent,
   and transformers' lazy importer rewrites that into
     "Could not import module 'Qwen3NextForCausalLM'.
      Are this object's requirements defined correctly?"
   which reads as a megatron-bridge / transformers version conflict and is not one
2  uv's build cache is keyed on (package, version), not on the toolchain, so every
   "rebuild" reissued a kernel-less wheel in 4 s while printing Building and Built
3  the default compiler is GCC 7.5.0; torch 2.11's headers reject it outright
4  the megatron-bridge install lacked --no-build-isolation -- the third site of a
   cause whose first two sites had already been fixed
5  nvidia-resiliency-ext 0.6.0 ships manylinux_2_39 wheels and no sdist at all;
   these nodes run glibc 2.38 and accept manylinux_2_38_aarch64 at best
```

**Layers 1 to 4 are cleared and verified inside a real run:**

```
host compiler for the CUDA extensions: g++ (SUSE Linux) 12.3.0
GATE S1b: causal-conv1d built AND causal_conv1d_cuda imports
Built mamba-ssm==2.3.2.post1  (2m46s)
both CUDA extensions load, and transformers qwen3_next imports
```

That last line is layer 1's disguise disappearing, **because a kernel got built and
not because any version changed**.

**Layer 5 is a platform fact, one glibc minor version wide.** No flag reaches it:
a forced source build returns in 0 s because there is no source distribution to
build. Routes that were not taken: an older megatron-bridge whose dependencies
have 2.38 wheels, building nvidia-resiliency-ext from its git repository rather
than PyPI, or a container with glibc 2.39.

### The recipe that does build the kernels

`CC`/`CXX` set by absolute path to `/opt/cray/pe/gcc-native/12/bin` with
`NVCC_PREPEND_FLAGS=-ccbin` on the same `g++`; `--refresh-package` to defeat the
build cache; `--no-build-isolation`; and every extension check importing `torch`
first, because a bare `import causal_conv1d_cuda` raises `ImportError: libc10.so`
even when the kernel is present and correct.

### What separated the layers was never a log line

| layer | what a log said | what actually settled it |
|---|---|---|
| 2 | `Building` then `Built` | **4 s versus 361 s** of wall time |
| 3 | `module load gcc-native/12.3` printed nothing | `$CXX --version` reporting 12.3.0 |
| 1 | a version-conflict `ModuleNotFoundError` | importing the module directly |
| 5 | "requirements are unsatisfiable" | the accepted platform tag: `manylinux_2_38_aarch64` |

Four gates in this harness reported success on environments that could not run
the code: `pip` exiting 0 while the extension was absent; the word `Built` while
nothing compiled; a silent `module load` that did not change the compiler; and a
bare extension import that fails on *correct* environments because torch was not
loaded first. **Each time the directly measurable quantity was one command away
and a cheaper proxy stood in front of it.**

**A3 does not say the megatron path is broken.** Nothing here tests it -- the run
never reaches it.

## 11. What this plan does not claim

The recipe records a gated-delta-net backward failure whose cause is named: the
TileLang JIT compiling the GDN kernels without a full CUDA toolkit, dying on a
missing `cuda/atomic` header. The 9B failure here is in the same family of
layers. That is a lead and not a diagnosis, and phase item C5 is written to ask
only whether the failure reproduces on the SkyRL stack. It is worth running
whichever way it lands: reproduction points at the model or the checkpoint,
while non-reproduction points at the training stack and gives the line back its
9B runs.

Nothing here proposes retiring slime. A1, A2, A5 and A6 passed and are reported
above with their evidence. A4 is half-measured and says so. A3 ran and
failed, with a single named cause and six eliminated hypotheses in section 10b;
what it does not establish is anything about the megatron path itself, which the
run never reaches. And the neighbour-contention failure that killed 50
of 111 runs is untouched by any of this, because it is a property of the
scheduler rather than of the framework.
""")

NB.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(NB, str(OUT))
print(f"-> {OUT}  ({len(NB.cells)} cells)")
