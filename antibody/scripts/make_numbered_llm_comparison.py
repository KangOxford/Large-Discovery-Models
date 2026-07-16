import pandas as pd
from pathlib import Path

wide_path = Path("outputs/comparisons/tables/llm_sections_vs_reproduction_wide.csv")
out_txt = Path("outputs/comparisons/tables/llm_sections_vs_reproduction_numbered.txt")
out_csv = Path("outputs/comparisons/tables/llm_sections_vs_reproduction_clean.csv")

df = pd.read_csv(wide_path)

# 重新整理成短列名，避免横着看崩掉
clean_rows = []

for _, r in df.iterrows():
    clean_rows.append({
        "Antigen": r["Antigen"],

        "Repro_energy": r["Reproduction_no_LLM_final_best"],

        "S1_energy": r["Section1_LLM_ranked_init_final_best"],
        "S1_delta": r["Section1_LLM_ranked_init_minus_reproduction"],

        "S2_energy": r["Section2_LLM_trust_region_final_best"],
        "S2_delta": r["Section2_LLM_trust_region_minus_reproduction"],

        "S3_energy": r["Section3_LLM_antigen_context_final_best"],
        "S3_delta": r["Section3_LLM_antigen_context_minus_reproduction"],

        "Full_LLM_energy": r["Full_LLM_all_sections_final_best"],
        "Full_LLM_delta": r["Full_LLM_all_sections_minus_reproduction"],

        "Best_method": r["Best_method_overall"],
        "Best_energy": r["Best_energy_overall"],
    })

clean_df = pd.DataFrame(clean_rows)
clean_df.to_csv(out_csv, index=False)

lines = []

lines.append("Column meaning:")
lines.append("1. Antigen")
lines.append("2. Repro_energy = reproduction / no-LLM final best energy")
lines.append("3. S1_energy = Section 1 LLM final best energy")
lines.append("4. S1_delta = Section 1 LLM energy - reproduction energy")
lines.append("5. S2_energy = Section 2 LLM final best energy")
lines.append("6. S2_delta = Section 2 LLM energy - reproduction energy")
lines.append("7. S3_energy = Section 3 LLM final best energy")
lines.append("8. S3_delta = Section 3 LLM energy - reproduction energy")
lines.append("9. Full_LLM_energy = all LLM sections final best energy")
lines.append("10. Full_LLM_delta = full LLM energy - reproduction energy")
lines.append("11. Best_method = method with lowest energy")
lines.append("12. Best_energy = lowest energy among all methods")
lines.append("")
lines.append("Important:")
lines.append("delta < 0 means better than reproduction, because lower binding energy is better.")
lines.append("delta > 0 means worse than reproduction.")
lines.append("")
lines.append("=" * 100)
lines.append("")

for _, r in clean_df.iterrows():
    lines.append(f"Antigen: {r['Antigen']}")
    lines.append("-" * 100)
    lines.append(f"1. Antigen: {r['Antigen']}")
    lines.append(f"2. Repro_energy: {r['Repro_energy']}")

    lines.append(f"3. S1_energy: {r['S1_energy']}")
    lines.append(f"4. S1_delta: {r['S1_delta']}")

    lines.append(f"5. S2_energy: {r['S2_energy']}")
    lines.append(f"6. S2_delta: {r['S2_delta']}")

    lines.append(f"7. S3_energy: {r['S3_energy']}")
    lines.append(f"8. S3_delta: {r['S3_delta']}")

    lines.append(f"9. Full_LLM_energy: {r['Full_LLM_energy']}")
    lines.append(f"10. Full_LLM_delta: {r['Full_LLM_delta']}")

    lines.append(f"11. Best_method: {r['Best_method']}")
    lines.append(f"12. Best_energy: {r['Best_energy']}")
    lines.append("")

out_txt.write_text("\n".join(lines))

print("[Saved]", out_txt)
print("[Saved]", out_csv)
print()
print(out_txt.read_text())
