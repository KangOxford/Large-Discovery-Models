#!/usr/bin/env python3
"""Compatibility launcher for the canonical nanoGPT workflow.

New runs should use ``scripts/run_ldm_tts.py`` with a config under
``config/nanogpt``. This launcher remains for existing direct CLI invocations.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tasks.nanogpt.core.workflow import main


if __name__ == "__main__":
    raise SystemExit(main())
