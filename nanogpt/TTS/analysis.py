import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

INPUT_TSV = Path("TTS/ablation_runs/iters20_seed123_02/iteration_feedback.tsv")
FAILURE_SCORE_CUTOFF = 1.0e8


def normalize_feedback_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Support both the old hand-curated TSV and new iteration_feedback.tsv."""
    df = raw_df.copy()
    df.columns = [str(column).strip() for column in df.columns]

    if "val_bpb" in df.columns:
        df["val_bpb"] = pd.to_numeric(df["val_bpb"], errors="coerce")
    elif {"score_key", "score"}.issubset(df.columns):
        metric_rows = df["score_key"].astype(str).str.strip().eq("val_bpb")
        df = df[metric_rows].copy()
        df["val_bpb"] = pd.to_numeric(df["score"], errors="coerce")
    elif "score" in df.columns:
        df["val_bpb"] = pd.to_numeric(df["score"], errors="coerce")
    else:
        available = ", ".join(df.columns)
        raise KeyError(
            "Could not find val_bpb. Expected either a 'val_bpb' column or "
            f"new feedback columns 'score_key'/'score'. Available columns: {available}"
        )

    if "memory_gb" in df.columns:
        df["memory_gb"] = pd.to_numeric(df["memory_gb"], errors="coerce")
    else:
        df["memory_gb"] = np.nan

    if "commit" not in df.columns:
        if "state_id" in df.columns:
            df["commit"] = df["state_id"]
        elif "iteration" in df.columns:
            df["commit"] = df["iteration"].map(lambda value: f"iter_{value}")
        else:
            df["commit"] = df.index.map(lambda value: f"row_{value + 1}")

    if "description" not in df.columns:
        if "action" in df.columns:
            df["description"] = df["action"]
        elif "path" in df.columns:
            df["description"] = df["path"]
        else:
            df["description"] = ""

    if "status" in df.columns:
        status = df["status"].astype(str).str.strip().str.upper()
    else:
        status = pd.Series("", index=df.index)

    status = status.replace(
        {
            "EVALUATED": "KEEP",
            "SUCCESS": "KEEP",
            "VALID": "KEEP",
            "OK": "KEEP",
            "FAILED": "CRASH",
            "FAILURE": "CRASH",
            "ERROR": "CRASH",
            "TIMEOUT": "CRASH",
        }
    )
    status = status.where(status.isin(["KEEP", "DISCARD", "CRASH"]), "")

    invalid_score = df["val_bpb"].isna() | (df["val_bpb"].abs() >= FAILURE_SCORE_CUTOFF)
    if "score_valid" in df.columns:
        score_valid = pd.to_numeric(df["score_valid"], errors="coerce")
        invalid_score |= score_valid.eq(0)

    status = status.mask((status == "") & ~invalid_score, "KEEP")
    status = status.mask(invalid_score, "CRASH")
    df["status"] = status
    df.loc[df["status"] == "CRASH", "val_bpb"] = np.nan
    return df


# Load the TSV. The normalizer accepts either:
# - old format: commit, val_bpb, memory_gb, status, description
# - new format: iteration_feedback.tsv with score_key/score/action/state_id
df = normalize_feedback_dataframe(pd.read_csv(INPUT_TSV, sep="\t"))

print(f"Total experiments: {len(df)}")
print(f"Columns: {list(df.columns)}")
df.head(10)

counts = df["status"].value_counts()
print("Experiment outcomes:")
print(counts.to_string())

n_keep = counts.get("KEEP", 0)
n_discard = counts.get("DISCARD", 0)
n_crash = counts.get("CRASH", 0)
n_decided = n_keep + n_discard
if n_decided > 0:
    print(f"\nKeep rate: {n_keep}/{n_decided} = {n_keep / n_decided:.1%}")


fig, ax = plt.subplots(figsize=(16, 8))

# Filter out crashes for plotting
valid = df[df["status"] != "CRASH"].copy()
valid = valid.dropna(subset=["val_bpb"])
valid = valid.reset_index(drop=True)
if valid.empty:
    raise SystemExit("No valid non-crash val_bpb rows found to plot.")

baseline_bpb = valid.loc[0, "val_bpb"]

# Only plot points at or below baseline (the interesting region)
below = valid[valid["val_bpb"] <= baseline_bpb + 0.0005]

# --- Identify the highlights to get boundaries for the background colors ---
highlight_commit_x = "109de35"
highlight_label = "Self-Reflection Happens"
highlight_commit_y = "63b1eba"

round2_highlight_commit_x = "f891adc"
round2_highlight_commit_y = "f144e50"

round3_highlight_commit_x = "e5be1b4"
round3_highlight_commit_y = "0ffc20f"

round4_highlight_commit_x = "d09d3cf"
round4_highlight_commit_y = "d09d3cf"



star1_idx = None
star2_idx = None
star3_idx = None
star4_idx = None

highlight_x = valid[valid["commit"].astype(str).str.strip() == highlight_commit_x]
highlight_y = valid[valid["commit"].astype(str).str.strip() == highlight_commit_y]
if not highlight_x.empty and not highlight_y.empty:
    star1_idx = highlight_x.index[0]
    star1_bpb = highlight_y.loc[highlight_y.index[0], "val_bpb"]

round2_highlight_x = valid[valid["commit"].astype(str).str.strip() == round2_highlight_commit_x]
round2_highlight_y = valid[valid["commit"].astype(str).str.strip() == round2_highlight_commit_y]
if not round2_highlight_x.empty and not round2_highlight_y.empty:
    star2_idx = round2_highlight_x.index[0]
    star2_bpb = round2_highlight_y.loc[round2_highlight_y.index[0], "val_bpb"]

round3_highlight_x = valid[valid["commit"].astype(str).str.strip() == round3_highlight_commit_x]
round3_highlight_y = valid[valid["commit"].astype(str).str.strip() == round3_highlight_commit_y]
if not round3_highlight_x.empty and not round3_highlight_y.empty:
    star3_idx = round3_highlight_x.index[0]
    star3_bpb = round3_highlight_y.loc[round3_highlight_y.index[0], "val_bpb"]


round4_highlight_x = valid[valid["commit"].astype(str).str.strip() == round4_highlight_commit_x]
round4_highlight_y = valid[valid["commit"].astype(str).str.strip() == round4_highlight_commit_y]
if not round4_highlight_x.empty and not round4_highlight_y.empty:
    star4_idx = round4_highlight_x.index[0]
    star4_bpb = round4_highlight_y.loc[round4_highlight_y.index[0], "val_bpb"]


# --- Draw Background Colors & Add Round Descriptions ---
max_idx = len(valid) - 1
idx1 = star1_idx if star1_idx is not None else max_idx
idx2 = star2_idx if star2_idx is not None else max_idx
idx3 = star3_idx if star3_idx is not None else max_idx
idx4 = star4_idx if star4_idx is not None else max_idx

# Round 1 (Start to Star 1)
ax.axvspan(-0.5, idx1, facecolor='#e6f2ff', alpha=0.5, label='Round 1', zorder=1)
ax.text((-0.5 + idx1) / 2, 0.96, "1st\nDense Scale",
        transform=ax.get_xaxis_transform(), ha='center', va='top',
        fontsize=11, fontweight='bold', color='#1b4f72',
        linespacing=1.25)


# Round 2 (Star 1 to Star 2)
if idx1 < max_idx:
    ax.axvspan(idx1, idx2, facecolor='#fff4e6', alpha=0.5, label='Round 2', zorder=1)
    ax.text((idx1 + idx2) / 2, 0.96, "2nd\nTry Sparse Memory", 
            transform=ax.get_xaxis_transform(), ha='center', va='top', 
            fontsize=11, fontweight='bold', color='#7e5109', linespacing=1.25)

# Round 3 (Star 2 to Star 3)
if idx2 < max_idx:
    ax.axvspan(idx2, idx3, facecolor='#e6ffe6', alpha=0.5, label='Round 3', zorder=1)
    ax.text((idx2 + idx3) / 2, 0.86, "3rd\nRebalancing Dense Compute\nAround Sparse Memory", 
            transform=ax.get_xaxis_transform(), ha='center', va='top', 
            fontsize=11, fontweight='bold', color='#1e8449', linespacing=1.25)

# Round 4 (Star 3 to Star 4)
if idx3 < max_idx:
    ax.axvspan(idx3, idx4, facecolor='#f3e8ff', alpha=0.5, label='Round 4', zorder=1)
    ax.text((idx3 + idx4) / 2, 0.96, "4th\nConfirming the 512 Sparse Frontier\nwith Ablation", 
            transform=ax.get_xaxis_transform(), ha='center', va='top', 
            fontsize=11, fontweight='bold', color='#6c3483', linespacing=1.25)

# Round 5 (Star 4 to the end)
if idx4 < max_idx:
    round5_end = max_idx + 5
    ax.axvspan(idx4, round5_end, facecolor='#ffe6f0', alpha=0.5, label='Round 5', zorder=1)
    ax.text((idx4 + round5_end) / 2, 0.96, "5th\nGrid Search: Optimizing the 512 Sparse Frontier\nWith VRAM-Aware Batch Scaling", 
            transform=ax.get_xaxis_transform(), ha='center', va='top', 
            fontsize=11, fontweight='bold', color='#922b21', linespacing=1.25)


# --- Plotting actual data ---
# Plot discarded as faint background dots
disc = below[below["status"] == "DISCARD"]
ax.scatter(disc.index, disc["val_bpb"],
           c="#cccccc", s=12, alpha=0.5, zorder=2)

# Plot kept experiments as prominent green dots
kept_v = below[below["status"] == "KEEP"]
# ax.scatter(kept_v.index, kept_v["val_bpb"],
        #    c="#2ecc71", s=50, zorder=4, edgecolors="black", linewidths=0.5)

# Best-so-far trace at each iteration, rather than a line connecting keep dots.
kept_mask = valid["status"] == "KEEP"
kept_idx = valid.index[kept_mask]
kept_bpb = valid.loc[kept_mask, "val_bpb"]
best_so_far = valid["val_bpb"].where(kept_mask).cummin().ffill()
ax.step(valid.index, best_so_far, where="post", color="#27ae60",
        linewidth=2, alpha=0.7, zorder=3, label="Best so far")


print_commit_list1 = ["d6307a7", "92c752a", "03451cd", "cd933d1", "cd9f56d", 
"ca0bf2e", "4aad2de", "63b1eba", "e12c850"]
print_commit_list2=["c7b9254", "99b0dc8", "40b137b", "021a2c1", "85fc2c2",
"a8813d4", "2d83009", "18b151d", "ef8fb21", "73a6bc1", "4bd5b50", "38460f5",
"d51c3cf", "c7feb78", "73e3730", "7f8c4d5", "de18ab9", "f144e50"]

print_commit_list3=["5c45636", "4bdd7dc", "36d9d25", "36d9d25", "c29ae95", "22f47f4", "eb523e5",
"ed387d1", "dee0d2f", "0ffc20f"]

print_commit_list4=["c47ea06", "63836d5", "bebf288", "b9196e4", "75197dc", "28f0220",
""]

print_commit_list5=["ee54b97", ]

print_commit_list = print_commit_list1 + print_commit_list2 + print_commit_list3 + print_commit_list4 + print_commit_list5
# Label each kept experiment with its description
for idx, bpb in zip(kept_idx, kept_bpb):
    commit = str(valid.loc[idx, "commit"]).strip()
    if commit in print_commit_list:
    
        desc = str(valid.loc[idx, "description"]).strip()
        if len(desc) > 45:
            desc = desc[:42] + "..."

        if commit in print_commit_list5:
            ax.scatter(idx, bpb,
                c="#3498db", s=50, zorder=4, edgecolors="black", linewidths=0.5)

        else:
            ax.scatter(idx, bpb,
            c="#2ecc71", s=50, zorder=4, edgecolors="black", linewidths=0.5)

            ax.annotate(desc, (idx, bpb),
                        textcoords="offset points",
                        xytext=(6, 6), fontsize=8.0,
                        color="#1a7a3a", alpha=0.9,
                        rotation=30, ha="left", va="bottom")

if len(valid) > 0:
    ax.annotate("Searched: batch100 warmdown 0.45625 adam 0.625", (max(len(valid) - 150, 0), valid["val_bpb"].iloc[-1]),
                textcoords="offset points",
                xytext=(6, 6), fontsize=8.0,
                color="#1a7a3a", alpha=0.9,
                rotation=30, ha="left", va="bottom")

# Draw the star highlights
if star1_idx is not None:
    ax.scatter([star1_idx], [star1_bpb],
               c="#f39c12", s=200, zorder=6, marker="*",
               edgecolors="black", linewidths=0.6, label="Self-Reflection happens here")
    # ax.annotate(highlight_label, (star1_idx, star1_bpb),
    #             textcoords="offset points",
    #             xytext=(-40, 48), fontsize=10.0, fontweight="bold",
    #             color="#d35400", alpha=0.95,
    #             arrowprops=dict(arrowstyle="->", color="#d35400", lw=1.2),
    #             ha="left", va="top")

if star2_idx is not None:
    ax.scatter([star2_idx], [star2_bpb],
               c="#f39c12", s=200, zorder=6, marker="*",
               edgecolors="black", linewidths=0.6)
    # ax.annotate(highlight_label, (star2_idx, star2_bpb),
    #             textcoords="offset points",
    #             xytext=(-40, 48), fontsize=10.0, fontweight="bold",
    #             color="#d35400", alpha=0.95,
    #             arrowprops=dict(arrowstyle="->", color="#d35400", lw=1.2),
    #             ha="left", va="top")

if star3_idx is not None:
    ax.scatter([star3_idx], [star3_bpb],
               c="#f39c12", s=200, zorder=6, marker="*",
               edgecolors="black", linewidths=0.6)
    # ax.annotate(highlight_label, (star3_idx, star3_bpb),
    #             textcoords="offset points",
    #             xytext=(-40, 48), fontsize=10.0, fontweight="bold",
    #             color="#d35400", alpha=0.95,
    #             arrowprops=dict(arrowstyle="->", color="#d35400", lw=1.2),
    #             ha="left", va="top")

if star4_idx is not None:
    ax.scatter([star4_idx], [star4_bpb],
               c="#f39c12", s=200, zorder=6, marker="*",
               edgecolors="black", linewidths=0.6)
    # ax.annotate(highlight_label, (star4_idx, star4_bpb),
    #             textcoords="offset points",
    #             xytext=(-40, 48), fontsize=10.0, fontweight="bold",
    #             color="#d35400", alpha=0.95,
    #             arrowprops=dict(arrowstyle="->", color="#d35400", lw=1.2),
    #             ha="left", va="top")

n_total = len(df)
n_kept = len(df[df["status"] == "KEEP"])
ax.set_xlabel("Experiment #", fontsize=12)
ax.set_ylabel("Validation BPB (lower is better)", fontsize=12)
ax.set_title(f"CodeX under ZERO Prior: Autoresearch with {n_total} Experiments, {n_kept} Kept Improvements", fontsize=14, pad=20)

# Create legend, placing Round backgrounds first
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, zorder=0)

# Y-axis: from just below best to just above baseline
best_bpb = kept_bpb.min()
margin = (baseline_bpb - best_bpb) * 0.15
# ax.set_ylim(best_bpb - margin, baseline_bpb + margin)
ax.set_xlim(-0.5, max_idx + 5)

plt.tight_layout()
plt.savefig("TTS/progress.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to progress.png")
