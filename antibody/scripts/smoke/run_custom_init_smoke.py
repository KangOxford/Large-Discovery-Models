"""Custom-init cache bootstrap smoke test.

Validates that `bo.custom_init._ensure_init_dataset()` resolves the initial
dataset under the new `./cache/` location, with three scenarios:

  1. `./cache/init_dataset/` already exists  → used directly.
  2. Only `./cache/init_dataset.zip` exists   → auto-extracted.
  3. Neither exists                            → clear `FileNotFoundError`.

This test mutates `./cache/` temporarily; always restores original state.

Usage (from repo root):
    python scripts/smoke/run_custom_init_smoke.py
"""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "cache"
ZIP = CACHE / "init_dataset.zip"
DIR = CACHE / "init_dataset"


def _drop_module(name: str) -> None:
    sys.modules.pop(name, None)


def scenario_extracted_dir() -> None:
    """Scenario 1: extracted directory present → used directly."""
    assert DIR.is_dir(), f"precondition: {DIR} must exist for this scenario"
    _drop_module("bo.custom_init")
    from bo.custom_init import INIT_DATA_PATH
    assert INIT_DATA_PATH == str(DIR), f"INIT_DATA_PATH mismatch: {INIT_DATA_PATH}"
    print("SCENARIO 1 OK: extracted directory used directly")


def scenario_zip_only() -> None:
    """Scenario 2: zip only, directory missing → auto-extract on import."""
    if not ZIP.is_file():
        print("SCENARIO 2 SKIPPED: zip missing")
        return
    saved = CACHE / "init_dataset.bak"
    if DIR.exists():
        if saved.exists():
            shutil.rmtree(saved)
        shutil.move(str(DIR), str(saved))
    try:
        _drop_module("bo.custom_init")
        from bo.custom_init import INIT_DATA_PATH  # noqa: F401
        assert DIR.is_dir(), "auto-extract did not produce init_dataset/"
        print("SCENARIO 2 OK: zip auto-extracted into cache/")
    finally:
        if saved.exists():
            if DIR.exists():
                shutil.rmtree(DIR)
            shutil.move(str(saved), str(DIR))


def scenario_missing() -> None:
    """Scenario 3: neither dir nor zip → FileNotFoundError mentions ./cache."""
    saved_zip = CACHE / "init_dataset.zip.bak"
    saved_dir = CACHE / "init_dataset.bak"
    if ZIP.exists():
        shutil.move(str(ZIP), str(saved_zip))
    if DIR.exists():
        shutil.move(str(DIR), str(saved_dir))
    try:
        _drop_module("bo.custom_init")
        try:
            import bo.custom_init  # noqa: F401
        except FileNotFoundError as e:
            msg = str(e)
            assert "cache" in msg, f"Error message should mention cache: {msg}"
            print("SCENARIO 3 OK: clear FileNotFoundError")
        else:
            raise AssertionError("Expected FileNotFoundError but import succeeded")
    finally:
        if saved_zip.exists():
            shutil.move(str(saved_zip), str(ZIP))
        if saved_dir.exists():
            shutil.move(str(saved_dir), str(DIR))


def main() -> None:
    assert CACHE.is_dir(), f"missing cache dir: {CACHE}"
    scenario_extracted_dir()
    scenario_zip_only()
    scenario_missing()
    print("ALL CUSTOM_INIT SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()