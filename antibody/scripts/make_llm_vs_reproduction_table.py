from pathlib import Path
import pandas as pd


ANTIGENS = [
    "1ADQ_A",
    "1FBI_X",
    "1H0D_C",
    "1NSN_S",
    "1OB1_C",
]

SEED = 42

REPRO_CSV = Path("outputs/reproduction/BO_transformed_overlap_optim_res.csv")

LLM_METHODS = {
    "Section1_LLM_ranked_init": Path("outputs/section_ablation/section1_ranked_init_only"),
    "Section2_LLM_trust_region": Path("outputs/section_ablation/section2_trust_region_only"),
    "Section3_LLM_antigen_context": Path("outputs/section_ablation/section3_antigen_context_effect"),
    "Full_LLM_all_sections": Path("outputs/backup/LLM_AntBO_1to3_seed42_5antigens"),
}

OUT_DIR = Path("outputs/comparisons/tables")
OUT_DIR.mkdir(exist_ok=True, parents=True)


def get_reproduction_result(rep_df, antigen):
    sub = rep_df[(rep_df["Antigen"] == antigen) & (rep_df["Seed"] == SEED)].copy()

    if len(sub) == 0:
        return None, None, 0

    sub = sub.sort_values("Num BB Evals")

    final_best = float(sub["Best Binding Energy"].iloc[-1])
    best_protein = str(sub["Best Protein"].iloc[-1])
    n_eval = int(sub["Num BB Evals"].iloc[-1])

    return final_best, best_protein, n_eval


def find_llm_result_csv(root, antigen):
    matches = list(root.glob(f"antigen_{antigen}_*/results.csv"))

    if len(matches) == 0:
        return None

    if len(matches) > 1:
        print(f"[Warning] multiple results found for {antigen} under {root}; using first:")
        for m in matches:
            print("  ", m)

    return matches[0]


def get_llm_result(root, antigen):
    csv_path = find_llm_result_csv(root, antigen)

    if csv_path is None:
        return None, None, 0, "missing"

    df = pd.read_csv(csv_path)

    if len(df) == 0:
        return None, None, 0, "empty"

    final_best = float(df["BestValue"].iloc[-1])
    best_protein = str(df["BestProtein"].iloc[-1])
    n_eval = len(df)

    return final_best, best_protein, n_eval, "ok"


rep_df = pd.read_csv(REPRO_CSV)

rows = []

for antigen in ANTIGENS:
    repro_energy, repro_protein, repro_n_eval = get_reproduction_result(rep_df, antigen)

    row = {
        "Antigen": antigen,
        "Reproduction_no_LLM_final_best": repro_energy,
        "Reproduction_no_LLM_best_protein": repro_protein,
        "Reproduction_no_LLM_n_eval": repro_n_eval,
    }

    energies = {
        "Reproduction_no_LLM": repro_energy,
    }

    for method_name, root in LLM_METHODS.items():
        llm_energy, llm_protein, llm_n_eval, status = get_llm_result(root, antigen)

        row[f"{method_name}_final_best"] = llm_energy
        row[f"{method_name}_best_protein"] = llm_protein
        row[f"{method_name}_n_eval"] = llm_n_eval
        row[f"{method_name}_status"] = status

        if repro_energy is not None and llm_energy is not None:
            delta = llm_energy - repro_energy
            improvement = repro_energy - llm_energy
        else:
            delta = None
            improvement = None

        # delta < 0 means LLM is better, because lower energy is better
        row[f"{method_name}_minus_reproduction"] = delta

        # improvement > 0 means LLM is better
        row[f"{method_name}_improvement_over_reproduction"] = improvement

        energies[method_name] = llm_energy

    valid_energies = {
        k: v for k, v in energies.items()
        if v is not None
    }

    if len(valid_energies) > 0:
        best_method = min(valid_energies, key=valid_energies.get)
        best_energy = valid_energies[best_method]
    else:
        best_method = None
        best_energy = None

    row["Best_method_overall"] = best_method
    row["Best_energy_overall"] = best_energy

    rows.append(row)


df = pd.DataFrame(rows)

wide_path = OUT_DIR / "llm_sections_vs_reproduction_wide.csv"
xlsx_path = OUT_DIR / "llm_sections_vs_reproduction.xlsx"

df.to_csv(wide_path, index=False)

with pd.ExcelWriter(xlsx_path) as writer:
    df.to_excel(writer, sheet_name="LLM_vs_reproduction", index=False)

print("[Saved]", wide_path)
print("[Saved]", xlsx_path)

print()
print("Compact comparison:")
compact_cols = [
    "Antigen",
    "Reproduction_no_LLM_final_best",
    "Section1_LLM_ranked_init_final_best",
    "Section1_LLM_ranked_init_minus_reproduction",
    "Section2_LLM_trust_region_final_best",
    "Section2_LLM_trust_region_minus_reproduction",
    "Section3_LLM_antigen_context_final_best",
    "Section3_LLM_antigen_context_minus_reproduction",
    "Full_LLM_all_sections_final_best",
    "Full_LLM_all_sections_minus_reproduction",
    "Best_method_overall",
]

print(df[compact_cols].to_string(index=False))

print()
print("Note:")
print("For *_minus_reproduction columns, negative value means the LLM method is better than reproduction.")
print("For energy, lower is better.")
