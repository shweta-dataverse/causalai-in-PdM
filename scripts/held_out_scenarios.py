#!/usr/bin/env python3
"""
Can the system handle a fault type it has never seen?

    python scripts/held_out_scenarios.py

For every fault scenario in turn, its recordings are removed, the causal graph
and the equations are learned from everything else, and then only that
scenario is diagnosed and replayed. This is the situation a plant is in: the
next fault is rarely one that has already happened in exactly this way.

The in-sample numbers from run_all_subsystems.py learn from the very faults
they then diagnose. The gap between the two tables is what that costs.

One small thing is still shared across folds: the healthy value of each signal
and the cut points for analog signals are learned once from all recordings.
Both come almost entirely from healthy periods.

Writes results/held_out_scenarios.csv (one row per subsystem) and
results/held_out_by_scenario.csv (one row per fold).
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
from causalpdm.pipeline import data, evaluate, paths, runner

SUBSYSTEMS = ["coolant", "probe", "hydraulics"]


def main() -> int:
    summary, folds = [], []
    for subsystem in SUBSYSTEMS:
        scenarios = sorted({run.experiment for run in data.load_fault_runs(subsystem)})
        pairs, checks, counterfactuals = [], [], []
        for scenario in scenarios:
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
                hold_out_scenario=scenario,
                use_cache=settings.USE_CACHE,
                verbose=False,
            )
            fold_pairs = [(r.ranked_variables, run.caused_by) for run, r in result.root_cause_results]
            pairs += fold_pairs
            checks += result.intervention_checks
            counterfactuals += result.counterfactuals
            fold_rca = evaluate.score_rankings(fold_pairs, k=settings.TOP_K_ROOT_CAUSES)
            folds.append({
                "subsystem": subsystem, "held_out": scenario, "runs": len(fold_pairs),
                "graph_f1": result.graph_score["f1"],
                "map@3": fold_rca.get("map@3"), "top1": fold_rca.get("found_in_top_1"),
                "intervention_f1": result.intervention_score.get("mean_f1"),
                "suspects_prevented": result.counterfactual_score.get("suspects_success_rate"),
            })
            print(f"  {subsystem:<11} without {scenario:<8} graph F1 {result.graph_score['f1']:.2f}  "
                  f"MAP@3 {fold_rca.get('map@3')}  intervention F1 "
                  f"{result.intervention_score.get('mean_f1')}", flush=True)

        rca = evaluate.score_rankings(pairs, k=settings.TOP_K_ROOT_CAUSES)
        by_fold = pd.DataFrame([f for f in folds if f["subsystem"] == subsystem])
        examined = len(counterfactuals)
        summary.append({
            "subsystem": subsystem,
            "scenarios": len(scenarios),
            "graph_f1_mean": round(by_fold["graph_f1"].mean(), 3),
            "graph_f1_sd": round(by_fold["graph_f1"].std(), 3),
            "root_cause_map@3": rca["map@3"],
            "root_cause_top1": rca["found_in_top_1"],
            "intervention_f1": round(sum(c["f1"] for c in checks) / len(checks), 3),
            "counterfactual_success": round(
                sum(1 for _, _, several, _ in counterfactuals if several.prevented_everything)
                / examined, 3),
            "counterfactual_upper_bound": round(
                sum(1 for _, _, _, truth in counterfactuals if truth.prevented_everything)
                / examined, 3),
        })

        pd.DataFrame(summary).to_csv(paths.RESULTS_DIR / "held_out_scenarios.csv", index=False)
        pd.DataFrame(folds).to_csv(paths.RESULTS_DIR / "held_out_by_scenario.csv", index=False)

    print("\n" + pd.DataFrame(summary).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
