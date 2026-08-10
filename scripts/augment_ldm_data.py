#!/usr/bin/env python3
"""Compatibility launcher for the consolidated data augmentation command."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    command = Path(__file__).resolve().parents[1] / "data" / "augment.py"
    runpy.run_path(str(command), run_name="__main__")
