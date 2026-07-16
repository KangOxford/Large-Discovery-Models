from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
config_path = ROOT / "bo" / "config.yaml"

if len(sys.argv) != 4:
    raise SystemExit(
        "Usage: python scripts/set_llm_sections.py "
        "<llm_ranked_init:true/false> "
        "<llm_trust_region:true/false> "
        "<llm_antigen_context:true/false>"
    )

ranked_init = sys.argv[1].lower()
trust_region = sys.argv[2].lower()
antigen_context = sys.argv[3].lower()

for name, value in [
    ("llm_ranked_init", ranked_init),
    ("llm_trust_region", trust_region),
    ("llm_antigen_context", antigen_context),
]:
    if value not in ["true", "false"]:
        raise ValueError(f"{name} must be true or false")

text = config_path.read_text()
new_lines = []

for line in text.splitlines():
    stripped = line.strip()

    if stripped.startswith("llm_ranked_init:"):
        new_lines.append(f"llm_ranked_init: {ranked_init}")
    elif stripped.startswith("llm_trust_region:"):
        new_lines.append(f"llm_trust_region: {trust_region}")
    elif stripped.startswith("llm_antigen_context:"):
        new_lines.append(f"llm_antigen_context: {antigen_context}")
    else:
        new_lines.append(line)

config_path.write_text("\n".join(new_lines) + "\n")

print(f"Updated {config_path}")
print(f"llm_ranked_init: {ranked_init}")
print(f"llm_trust_region: {trust_region}")
print(f"llm_antigen_context: {antigen_context}")
