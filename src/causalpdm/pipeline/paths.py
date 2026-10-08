"""Where everything lives on disk."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
CAUSRCA_DIR = DATA_DIR / "causrca" / "raw"

NORMAL_RECORDINGS_DIR = CAUSRCA_DIR / "real_op"
FAULT_RECORDINGS_DIR = CAUSRCA_DIR / "dig_twin"
EXPERT_GRAPH_DIR = CAUSRCA_DIR / "expert_graph"

RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"
REPORTS_DIR = PROJECT_ROOT / "reports"
EDA_DIR = REPORTS_DIR / "eda"


def subsystem_dir(subsystem: str) -> Path:
    """Folder holding one subsystem's fault experiments and expert subgraph."""
    if subsystem == "full":
        return EXPERT_GRAPH_DIR
    return FAULT_RECORDINGS_DIR / f"exp_{subsystem}"


def make_output_folders() -> None:
    """Create the folders we write into, if they do not exist yet."""
    for folder in (RESULTS_DIR, CACHE_DIR, REPORTS_DIR, EDA_DIR):
        folder.mkdir(parents=True, exist_ok=True)
