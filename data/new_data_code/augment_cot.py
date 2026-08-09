#!/usr/bin/env python3
"""Deprecated wrapper for the integrated expert-justification CLI."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.augment_ldm_data import main


if __name__ == "__main__":
    print(
        "warning: data/new_data_code/augment_cot.py is deprecated; "
        "use scripts/augment_ldm_data.py",
        file=sys.stderr,
    )
    raise SystemExit(main())
