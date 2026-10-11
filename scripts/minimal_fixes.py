#!/usr/bin/env python3
"""
For every fault: the smallest fix that would have prevented it.

    python scripts/minimal_fixes.py

Run after scripts/run_all_subsystems.py - it reads the saved results. For each
fault it searches fixes of one, two and three signals among the eight highest
ranked suspects, and keeps the smallest one that would have prevented the
fault with probability at least 0.9 (see explain.minimal_fix).

Reported per subsystem:

  found      share of faults where a fix of at most three signals works
  size       how many signals the fix needed, on average
  exact      share where the fix is exactly the signals the test rig forced
  only_true  share where every signal in the fix was really forced

Writes results/minimal_fixes.csv and results/minimal_fixes_by_fault.csv.
"""

import pickle
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

warnings.filterwarnings("ignore")

import pandas as pd

from causalpdm.pipeline import data, explain, paths

SAVED = {"coolant": "pipeline_result.pkl",
         "hydraulics": "pipeline_result_hydraulics.pkl",
         "probe": "pipeline_result_probe.pkl"}
CANDIDATES = 8


def main() -> int:
    by_fault = []
    for subsystem, name in SAVED.items():
        result = pickle.loads((paths.CACHE_DIR / name).read_bytes())
        model = result.causal_model
        alarms = data.load_expert_graph(subsystem).alarm_names
        tables = {label.split("/", 1)[1]: table.reset_index(drop=True)
                  for label, table in zip(result.table_labels, result.tables)
                  if label.startswith("fault")}
        for run, ranking in result.root_cause_results:
            start = max(ranking.fault_window[0] - 2, model.max_lag)
            found = explain.minimal_fix(
                model, tables[run.name], ranking.ranked_variables[:CANDIDATES],
                result.healthy_values, alarms, from_step=start)
            fix, truth = set(found["fix"]), set(run.caused_by)
            by_fault.append({
                "subsystem": subsystem, "fault": run.name, "true_causes": sorted(truth),
                "fix": sorted(fix), "probability": round(found["probability"], 3),
                "size": found["size"], "found": found["reached_target"],
                "exact": found["reached_target"] and fix == truth,
                "only_true": found["reached_target"] and fix <= truth,
                "sets_tried": found["sets_tried"],
            })
        done = pd.DataFrame([row for row in by_fault if row["subsystem"] == subsystem])
        print(f"  {subsystem:<11} faults {len(done):>3}  found {done['found'].mean():.2f}  "
              f"size {done.loc[done['found'], 'size'].mean():.2f}  "
              f"exact {done['exact'].mean():.2f}  only true causes {done['only_true'].mean():.2f}",
              flush=True)

    table = pd.DataFrame(by_fault)
    table.to_csv(paths.RESULTS_DIR / "minimal_fixes_by_fault.csv", index=False)
    summary = table.groupby("subsystem").agg(
        faults=("fault", "count"), found=("found", "mean"),
        size=("size", "mean"), exact=("exact", "mean"), only_true=("only_true", "mean"))
    summary.round(3).to_csv(paths.RESULTS_DIR / "minimal_fixes.csv")
    print("\n" + summary.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
