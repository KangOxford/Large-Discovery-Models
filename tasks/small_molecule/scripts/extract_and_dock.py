#!/usr/bin/env python3
"""CLI adapter for the small-molecule docking implementation."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tasks.small_molecule.core.docking import main


if __name__ == "__main__":
    raise SystemExit(main())
