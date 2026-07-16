from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Reproduction aggregate results, seed 42
reproduction_csv = Path("outputs/reproduction/BO_transformed_overlap_optim_res.csv")

# LLM-AntBO results, seed 42
llm_root = Path("outputs/backup/LLM_AntBO_1to3_seed42_5antigens")

antigens = [
    "1ADQ_A",
    "1FBI_X",
    "1H0D_C",
    "1NSN_S",
    "1OB1_C",
]

target_seed = 42

output_dir = Path("outputs/comparisons/reproduction_vs_llm_5antigens")
output_dir.mkdir(exist_ok=True, parents=True)


def load_reproduction_curve(rep_df, antigen, seed):
    sub = rep_df[(rep_df["Antigen"] == antigen) & (rep_df["Seed"] == seed)].copy()

    if len(sub) == 0:
        raise ValueError(f"No reproduction result found for {antigen}, seed {seed}")

    sub = sub.sort_values("Num BB Evals")

    x = pd.to_numeric(sub["Num BB Evals"], errors="coerce")
    y = pd.to_numeric(sub["Best Binding Energy"], errors="coerce")

    valid = x.notna() & y.notna()

    return x[valid], y[valid], sub[valid]


def load_llm_curve(llm_root, antigen):
    matches = list(llm_root.glob(f"antigen_{antigen}_*/results.csv"))

    if len(matches) == 0:
        raise ValueError(f"No LLM result found for {antigen}")

    csv_path = matches[0]
    df = pd.read_csv(csv_path)

    x = pd.to_numeric(df["Index"], errors="coerce") + 1
    y = pd.to_numeric(df["BestValue"], errors="coerce")

    valid = x.notna() & y.notna()

    return x[valid], y[valid], df[valid], csv_path


rep_df = pd.read_csv(reproduction_csv)

summary_rows = []

for antigen in antigens:
    rep_x, rep_y, rep_sub = load_reproduction_curve(rep_df, antigen, target_seed)
    llm_x, llm_y, llm_df, llm_file = load_llm_curve(llm_root, antigen)

    plt.figure(figsize=(7.2, 4.8))

    plt.plot(rep_x, rep_y, label="Reproduction")
    plt.plot(llm_x, llm_y, label="LLM-AntBO")

    plt.xlabel("Evaluation")
    plt.ylabel("Best-so-far energy")
    plt.title(f"Best-so-far convergence: {antigen}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / f"{antigen}_reproduction_vs_llm.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    rep_final_best = rep_y.iloc[-1]
    llm_final_best = llm_y.iloc[-1]

    rep_final_protein = rep_sub["Best Protein"].iloc[-1]
    llm_final_protein = llm_df["BestProtein"].iloc[-1]

    summary_rows.append({
        "Antigen": antigen,
        "Reproduction_final_best": rep_final_best,
        "LLM_final_best": llm_final_best,
        "LLM_minus_Reproduction": llm_final_best - rep_final_best,
        "Reproduction_final_protein": rep_final_protein,
        "LLM_final_protein": llm_final_protein,
        "Reproduction_num_evals": len(rep_y),
        "LLM_num_evals": len(llm_y),
        "LLM_file": str(llm_file),
    })

    print(f"[Saved] {output_path}")


summary = pd.DataFrame(summary_rows)
summary_path = output_dir / "final_best_summary.csv"
summary.to_csv(summary_path, index=False)

print()
print("[Saved]", summary_path)
print()
print(summary[[
    "Antigen",
    "Reproduction_final_best",
    "LLM_final_best",
    "LLM_minus_Reproduction",
]])
