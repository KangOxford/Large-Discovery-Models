"""bo/ root conftest — adds AntBO root to sys.path for pytest collection."""
import sys
from pathlib import Path

ANTBO_ROOT = Path(__file__).resolve().parent.parent
if str(ANTBO_ROOT) not in sys.path:
    sys.path.insert(0, str(ANTBO_ROOT))