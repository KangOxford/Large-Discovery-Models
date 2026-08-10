# Empty package marker for pytest to discover tests/ as a package.
# Also sets up sys.path to allow ``from tasks.antibody.core.ldm.dsl...`` imports.
import sys
from pathlib import Path

ANTBO_ROOT = Path(__file__).resolve().parent.parent
if str(ANTBO_ROOT) not in sys.path:
    sys.path.insert(0, str(ANTBO_ROOT))