"""Central path configuration for the project.

All notebooks and scripts resolve data locations through this module so the
project can be moved or re-rooted without touching analysis code.
"""

from pathlib import Path

# Repository root = two levels above this file (src/causalpdm/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"

CAUSRCA_RAW = DATA_DIR / "causrca" / "raw"
CMAPSS_RAW = DATA_DIR / "cmapss" / "raw"
SCANIA_RAW = DATA_DIR / "scania" / "raw"

REPORTS_DIR = REPO_ROOT / "reports"
