import pandas as pd
from pathlib import Path

wide_path = Path("outputs/comparisons/tables/llm_sections_vs_reproduction_wide.csv")
out_path = Path("outputs/comparisons/tables/llm_sections_vs_reproduction_vertical.csv")

df = pd.read_csv(wide_path)

methods = [
    {
        "Method": "Reproduction_no_LLM",
        "Energy_col": "Reproduction_no_LLM_final_best",
        "Delta_col": None,
    },
    {
        "Method": "Section1_LLM_ranked_init",
        "Energy_col": "Section1_LLM_ranked_init_final_best",
        "Delta_col": "Section1_LLM_ranked_init_minus_reproduction",
    },
    {
        "Method": "Section2_LLM_trust_region",
        "Energy_col": "Section2_LLM_trust_region_final_best",
        "Delta_col": "Section2_LLM_trust_region_minus_reproduction",
    },
    {
        "Method": "Section3_LLM_antigen_context",
        "Energy_col": "Section3_LLM_antigen_context_final_best",
        "Delta_col": "Section3_LLM_antigen_context_minus_reproduction",
    },
    {
        "Method": "Full_LLM_all_sections",
        "Energy_col": "Full_LLM_all_sections_final_best",
        "Delta_col": "Full_LLM_all_sections_minus_reproduction",
    },
]

rows = []

for _, row in df.iterrows():
    antigen = row["Antigen"]

    for item in methods:
        method = item["Method"]
        energy = row[item["Energy_col"]]

        if item["Delta_col"] is None:
            delta = 0.0
            result = "baseline"
        else:
            delta = row[item["Delta_col"]]

            if pd.isna(delta):
                result = "missing"
            elif delta < 0:
                result = "better_than_reproduction"
            elif delta > 0:
                result = "worse_than_reproduction"
            else:
                result = "same_as_reproduction"

        rows.append({
            "Antigen": antigen,
            "Method": method,
            "Final_best_energy": energy,
            "Delta_vs_reproduction": delta,
            "Result": result,
            "Best_method_overall": row["Best_method_overall"],
        })

vertical_df = pd.DataFrame(rows)
vertical_df.to_csv(out_path, index=False)

print("[Saved]", out_path)
print()
print(vertical_df.to_string(index=False))
