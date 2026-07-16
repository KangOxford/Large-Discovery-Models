import pandas as pd

path = "outputs/comparisons/tables/llm_sections_vs_reproduction_wide.csv"
df = pd.read_csv(path)

methods = [
    (
        "Reproduction no-LLM",
        "Reproduction_no_LLM_final_best",
        None,
    ),
    (
        "Section 1 LLM: ranked initialization",
        "Section1_LLM_ranked_init_final_best",
        "Section1_LLM_ranked_init_minus_reproduction",
    ),
    (
        "Section 2 LLM: trust region",
        "Section2_LLM_trust_region_final_best",
        "Section2_LLM_trust_region_minus_reproduction",
    ),
    (
        "Section 3 LLM: antigen context",
        "Section3_LLM_antigen_context_final_best",
        "Section3_LLM_antigen_context_minus_reproduction",
    ),
    (
        "Full LLM: all sections",
        "Full_LLM_all_sections_final_best",
        "Full_LLM_all_sections_minus_reproduction",
    ),
]

for _, row in df.iterrows():
    print("=" * 90)
    print(f"Antigen: {row['Antigen']}")
    print("-" * 90)

    for i, (name, energy_col, delta_col) in enumerate(methods, start=1):
        energy = row.get(energy_col)

        if delta_col is None:
            print(f"{i}. {name}")
            print(f"   final best energy: {energy}")
            print(f"   delta vs reproduction: baseline")
        else:
            delta = row.get(delta_col)

            if pd.isna(delta):
                verdict = "missing"
            elif delta < 0:
                verdict = "better than reproduction"
            elif delta > 0:
                verdict = "worse than reproduction"
            else:
                verdict = "same as reproduction"

            print(f"{i}. {name}")
            print(f"   final best energy: {energy}")
            print(f"   delta vs reproduction: {delta}")
            print(f"   result: {verdict}")

    print("-" * 90)
    print(f"Best method overall: {row['Best_method_overall']}")
    print()
