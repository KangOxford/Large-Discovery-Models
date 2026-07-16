from pathlib import Path
import os
import shutil

ROOT = Path("/mnt/data0/shared/AntBO/HEBO/AntBO")
OUT = ROOT / "outputs"
APPLY = os.getenv("APPLY") == "1"

PROTECTED = {
    "archive",
    "baselines",
    "comparison",
    "experiments",
    "figures",
    "tables",
}

def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))

def mkdir(rel_path: str):
    path = OUT / rel_path
    print("MKDIR", rel(path))
    if APPLY:
        path.mkdir(parents=True, exist_ok=True)

def unique_dst(dst: Path) -> Path:
    if not dst.exists():
        return dst
    for i in range(1, 1000):
        candidate = dst.with_name(f"{dst.name}_{i}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Too many destination collisions for {dst}")

def move_rel(src_rel: str, dst_rel: str):
    src = OUT / src_rel
    dst = OUT / dst_rel

    if not src.exists():
        print("SKIP missing", rel(src))
        return

    dst = unique_dst(dst)
    print(("MOVE" if APPLY else "DRY-RUN MOVE"), rel(src), "->", rel(dst))

    if APPLY:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

def move_children(src_rel: str, dst_rel: str):
    src = OUT / src_rel
    dst = OUT / dst_rel

    if not src.exists():
        print("SKIP missing", rel(src))
        return

    for child in sorted(src.iterdir()):
        move_rel(str(child.relative_to(OUT)), str((dst / child.name).relative_to(OUT)))

def remove_empty_dirs():
    for path in sorted(OUT.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if not path.is_dir():
            continue

        if path == OUT:
            continue

        top_name = path.relative_to(OUT).parts[0]
        if top_name in {"experiments", "tables", "figures", "archive", "baselines", "comparison"}:
            # Keep these structure directories safe.
            continue

        try:
            if not any(path.iterdir()):
                print(("RMDIR" if APPLY else "DRY-RUN RMDIR"), rel(path))
                if APPLY:
                    path.rmdir()
        except FileNotFoundError:
            pass

# Create final target folders.
for d in [
    "archive/debug",
    "archive/smoke",
    "archive/old_layout",
    "baselines",
    "comparison",
    "experiments/ablation",
    "figures/comparison",
    "figures/formal_5ag5seed200",
    "tables/comparison",
    "tables/formal_5ag5seed200",
]:
    mkdir(d)

# Ablation.
move_rel("ldm/section_ablation", "experiments/ablation/ldm_section_ablation")

# Traditional baselines.
move_rel("reproduction", "baselines/reproduction")

# Merge comparisons into comparison.
move_children("comparisons", "comparison")

# Old layout / plotlink / duplicate mirrors.
for src, dst in [
    ("manifest.yaml", "archive/old_layout/root_manifest.yaml"),
    ("README.md", "archive/old_layout/README.md"),
    ("figures_tables", "archive/old_layout/figures_tables"),
    ("llm_acq_5antigen_5seed_200eval_plotlink", "archive/old_layout/llm_acq_5antigen_5seed_200eval_plotlink"),
    ("llm_direct_formal_plotlink", "archive/old_layout/llm_direct_formal_plotlink"),
    ("ldm_reservoir_5antigen_5seed_200eval", "archive/old_layout/ldm_reservoir_5antigen_5seed_200eval"),
    ("ldm_reservoir_nchoices_5antigen_5seed_200eval", "archive/old_layout/ldm_reservoir_nchoices_5antigen_5seed_200eval"),
    ("llm_baseline", "archive/old_layout/llm_baseline"),
    ("llm_direct_formal", "archive/old_layout/llm_direct_formal"),
    ("ldm", "archive/old_layout/ldm"),
    ("ldm_parallel", "archive/old_layout/ldm_parallel"),
    ("ldm_parallel_argmax", "archive/old_layout/ldm_parallel_argmax"),
]:
    move_rel(src, dst)

# Smoke / sanity / manual tests.
for src in [
    "ldm_reservoir_independent_sanity_llm_absolut",
    "ldm_reservoir_nchoices_sanity_llm_absolut",
    "ldm_reservoir_sanity_absolut",
    "ldm_reservoir_sanity_llm_absolut",
    "ldm_reservoir_smoke",
    "llm_direct_smoke_manual",
]:
    move_rel(src, f"archive/smoke/{src}")

# Debug material.
# Note: outputs/logs was deleted or permission-problematic, so it is intentionally skipped.
for src in [
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
    move_rel(src, f"archive/debug/{src}")

# Loose top-level files.
for src in sorted(OUT.iterdir()):
    if not src.is_file():
        continue

    if src.name == ".DS_Store":
        move_rel(src.name, f"archive/debug/ds_store/{src.name}")
    elif src.suffix.lower() == ".log":
        move_rel(src.name, f"archive/debug/logs/{src.name}")
    else:
        move_rel(src.name, f"archive/old_layout/top_level_files/{src.name}")

remove_empty_dirs()

print()
print("Mode:", "APPLY" if APPLY else "DRY-RUN")
print("Expected top-level dirs: archive, baselines, comparison, experiments, figures, tables")
