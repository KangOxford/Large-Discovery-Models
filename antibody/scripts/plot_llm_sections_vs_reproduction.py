from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


ANTIGENS = ["1ADQ_A", "1FBI_X", "1H0D_C", "1NSN_S", "1OB1_C"]
SEED = 42

REPRO_CSV = Path("outputs/reproduction/BO_transformed_overlap_optim_res.csv")

METHODS = {
    "Reproduction no-LLM": {
        "type": "reproduction",
        "path": REPRO_CSV,
    },
    "Section 1 LLM": {
        "type": "llm",
        "path": Path("outputs/section_ablation/section1_ranked_init_only"),
    },
    "Section 2 LLM": {
        "type": "llm",
        "path": Path("outputs/section_ablation/section2_trust_region_only"),
    },
    "Section 3 LLM": {
        "type": "llm",
        "path": Path("outputs/section_ablation/section3_antigen_context_effect"),
    },
    "Full LLM": {
        "type": "llm",
        "path": Path("outputs/backup/LLM_AntBO_1to3_seed42_5antigens"),
    },
}

OUT_DIR = Path("outputs/comparisons/plots/llm_sections_vs_reproduction")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_reproduction_curve(rep_df, antigen):
    sub = rep_df[(rep_df["Antigen"] == antigen) & (rep_df["Seed"] == SEED)].copy()

    if len(sub) == 0:
        print(f"[Missing reproduction] {antigen}")
        return None

    sub = sub.sort_values("Num BB Evals")

    curve = pd.DataFrame({
        "x": sub["Num BB Evals"].astype(float),
        "y": sub["Best Binding Energy"].astype(float),
    })

    return curve


def find_llm_results_csv(root, antigen):
    matches = list(root.glob(f"antigen_{antigen}_*/results.csv"))

    if len(matches) == 0:
        print(f"[Missing LLM result] {antigen} under {root}")
        return None

    if len(matches) > 1:
        print(f"[Warning] multiple results found for {antigen} under {root}; using first:")
        for m in matches:
            print("  ", m)

    return matches[0]


def load_llm_curve(root, antigen):
    csv_path = find_llm_results_csv(root, antigen)

    if csv_path is None:
        return None

    df = pd.read_csv(csv_path)

    if len(df) == 0:
        print(f"[Empty result] {csv_path}")
        return None

    curve = pd.DataFrame({
        "x": df["Index"].astype(float) + 1,
        "y": df["BestValue"].astype(float),
    })

    return curve


rep_df = pd.read_csv(REPRO_CSV)

summary_rows = []

for antigen in ANTIGENS:
    plt.figure(figsize=(9, 6))

    final_values = {}

    for method_name, info in METHODS.items():
        if info["type"] == "reproduction":
            curve = load_reproduction_curve(rep_df, antigen)
        else:
            curve = load_llm_curve(info["path"], antigen)

        if curve is None:
            continue

        plt.plot(curve["x"], curve["y"], label=method_name, linewidth=2)

        final_values[method_name] = float(curve["y"].iloc[-1])

    if len(final_values) == 0:
        plt.close()
        continue

    best_method = min(final_values, key=final_values.get)
    best_energy = final_values[best_method]

    plt.title(f"{antigen}: LLM sections vs reproduction")
    plt.xlabel("Number of black-box evaluations")
    plt.ylabel("Best-so-far binding energy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    note = f"Best: {best_method} ({best_energy:.3f})"
    plt.text(
        0.02,
        0.02,
        note,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    out_path = OUT_DIR / f"{antigen}_llm_sections_vs_reproduction.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("[Saved]", out_path)

    row = {
        "Antigen": antigen,
        "Best_method": best_method,
        "Best_energy": best_energy,
    }

    for method_name, value in final_values.items():
        row[method_name] = value

    summary_rows.append(row)


summary_df = pd.DataFrame(summary_rows)
summary_path = OUT_DIR / "plot_final_energy_summary.csv"
summary_df.to_csv(summary_path, index=False)

print()
print("[Saved]", summary_path)
print()
print(summary_df.to_string(index=False))
