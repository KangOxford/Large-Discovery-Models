from pathlib import Path
import os
import shutil

ROOT = Path("/mnt/data0/shared/AntBO/HEBO/AntBO")
OUT = ROOT / "outputs"
APPLY = os.getenv("APPLY") == "1"

def ensure_dir(rel):
    path = OUT / rel
    print("MKDIR", path.relative_to(ROOT))
    if APPLY:
        path.mkdir(parents=True, exist_ok=True)

def unique_dst(dst):
    if not dst.exists():
        return dst
    for i in range(1, 1000):
        candidate = dst.with_name(f"{dst.name}_{i}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Too many destination collisions for {dst}")

def move(src_rel, dst_rel):
    src = OUT / src_rel
    dst = OUT / dst_rel

    if not src.exists():
        print("SKIP missing", src.relative_to(ROOT))
        return

    dst = unique_dst(dst)
    action = "MOVE" if APPLY else "DRY-RUN MOVE"
    print(action, src.relative_to(ROOT), "->", dst.relative_to(ROOT))

    if APPLY:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

def move_children(src_rel, dst_rel):
    src = OUT / src_rel
    dst = OUT / dst_rel

    if not src.exists():
        print("SKIP missing", src.relative_to(ROOT))
        return

    for child in sorted(src.iterdir()):
        move(str(child.relative_to(OUT)), str((dst / child.name).relative_to(OUT)))

def remove_empty_dirs():
    for path in sorted(OUT.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and path != OUT:
            try:
                if not any(path.iterdir()):
                    action = "RMDIR" if APPLY else "DRY-RUN RMDIR"
                    print(action, path.relative_to(ROOT))
                    if APPLY:
                        path.rmdir()
            except FileNotFoundError:
                pass

# Create target folders
for rel in [
    "archive/incomplete",
    "archive/debug",
    "archive/smoke",
    "archive/old_layout",
    "baselines",
    "comparison",
    "experiments/ablation",
    "figures/formal_5ag5seed200",
    "tables/formal_5ag5seed200",
]:
    ensure_dir(rel)

# Ablation
move("ldm/section_ablation", "experiments/ablation/ldm_section_ablation")

# Traditional baselines
move("reproduction", "baselines/reproduction")

# Merge comparisons into comparison
move_children("comparisons", "comparison")

# Old layout / duplicate / plotlink / old figures and tables
move("manifest.yaml", "archive/old_layout/root_manifest.yaml")
move("figures_tables", "archive/old_layout/figures_tables")
move("llm_acq_5antigen_5seed_200eval_plotlink", "archive/old_layout/llm_acq_5antigen_5seed_200eval_plotlink")
move("llm_direct_formal_plotlink", "archive/old_layout/llm_direct_formal_plotlink")
move("ldm_reservoir_5antigen_5seed_200eval", "archive/old_layout/ldm_reservoir_5antigen_5seed_200eval")
move("ldm_reservoir_nchoices_5antigen_5seed_200eval", "archive/old_layout/ldm_reservoir_nchoices_5antigen_5seed_200eval")

# Smoke / sanity / manual test
for rel in [
    "ldm_reservoir_independent_sanity_llm_absolut",
    "ldm_reservoir_nchoices_sanity_llm_absolut",
    "ldm_reservoir_sanity_absolut",
    "ldm_reservoir_sanity_llm_absolut",
    "ldm_reservoir_smoke",
    "llm_direct_smoke_manual",
]:
    move(rel, f"archive/smoke/{rel}")

# Debug / logs / configs / sandboxes / decision traces
for rel in [
    # "logs",
    "ldm_reservoir_logs",
    "ldm/llm_decisions",
    "ldm_parallel/logs",
    "ldm_parallel/configs",
    "ldm_parallel/absolut_sandboxes",
    "ldm_parallel_argmax/logs",
    "ldm_parallel_argmax/configs",
    "ldm_parallel_argmax/absolut_sandboxes",
    "llm_direct_formal/logs",
]:
    move(rel, f"archive/debug/{rel}")

remove_empty_dirs()

print()
print("Mode:", "APPLY" if APPLY else "DRY-RUN")
print("After cleanup, desired top-level outputs dirs are:")
print("  archive")
print("  baselines")
print("  comparison")
print("  experiments")
print("  figures")
print("  tables")
