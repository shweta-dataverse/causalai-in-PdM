#!/usr/bin/env python3
"""
Find the best settings by scoring them against the engineers' causal graph.

    python scripts/tune_settings.py

Most causal discovery work cannot do this, because there is no correct answer
to compare against. causRCA ships one, so we can simply try each combination of
settings and keep whichever recovers the expert graph best.

Writes results/tuning_results.csv and results/best_settings.json. Copy the
winning values into settings.py to use them.
"""

import json
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

import settings
from causalpdm.pipeline import data, paths, tuning


def main() -> int:
    dataset = data.load(settings.SUBSYSTEM, normal_recordings=settings.NORMAL_RECORDINGS)
    total = len(settings.TUNING_GRIDS) * len(settings.TUNING_LAGS) * len(settings.TUNING_SIGNIFICANCE)

    print(f"Trying {total} combinations of settings on the {settings.SUBSYSTEM} subsystem.")
    print(f"Each one is scored against the engineers' graph "
          f"({len(dataset.expert_graph.edges)} known links).\n")

    result = tuning.search(
        dataset,
        grids=settings.TUNING_GRIDS,
        lags=settings.TUNING_LAGS,
        significances=settings.TUNING_SIGNIFICANCE,
        objective=settings.TUNING_OBJECTIVE,
        independence_test=settings.INDEPENDENCE_TEST,
        min_rows=settings.MIN_RECORDING_ROWS,
    )

    print("\n" + "=" * 78)
    print(result.table.to_string(index=False))
    print("=" * 78)
    print("\n" + tuning.explain_result(result))

    paths.make_output_folders()
    result.table.to_csv(paths.RESULTS_DIR / "tuning_results.csv", index=False)
    (paths.RESULTS_DIR / "best_settings.json").write_text(
        json.dumps(result.best_settings, indent=2))

    print("\nTo use these, set them in settings.py:")
    for key, value in result.best_settings.items():
        print(f"  {key} = {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
