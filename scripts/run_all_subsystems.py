#!/usr/bin/env python3
"""
Run the pipeline on every subsystem and write one combined results table.

    python scripts/run_all_subsystems.py

Every subsystem uses the same settings from settings.py. They were tuned on
coolant only, so hydraulics and probe are effectively held out.

Writes results/all_subsystems.csv, plus each subsystem's own results folder.
Hydraulics is by far the slowest (about half an hour), so it runs last.
"""

import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

import pandas as pd

import settings
from causalpdm.pipeline import paths, runner

SUBSYSTEMS = ["coolant", "probe", "hydraulics"]


def main() -> int:
    rows = []
    for subsystem in SUBSYSTEMS:
        print(f"\n=== {subsystem} ===")
        result = runner.run(
            subsystem=subsystem,
            normal_recordings=settings.NORMAL_RECORDINGS,
            seconds_per_step=settings.RESAMPLE_SECONDS,
            max_lag=settings.MAX_LAG,
            significance=settings.SIGNIFICANCE,
            independence_test=settings.INDEPENDENCE_TEST,
            multiple_test_correction=settings.MULTIPLE_TEST_CORRECTION,
            min_rows=settings.MIN_RECORDING_ROWS,
            top_k=settings.TOP_K_ROOT_CAUSES,
            alarms_only_as_effects=settings.ALARMS_ONLY_AS_EFFECTS,
            same_moment_partners=settings.SAME_MOMENT_PARTNERS,
            use_cache=settings.USE_CACHE,
        )
        runner.save(result)
        rows.append({**result.headline(), "graph_shd": result.graph_score["shd"]})

        # Written after every subsystem, so a long hydraulics run that gets
        # interrupted still leaves the finished ones on disk.
        table = pd.DataFrame(rows)
        table.to_csv(paths.RESULTS_DIR / "all_subsystems.csv", index=False)

    print("\n" + table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
