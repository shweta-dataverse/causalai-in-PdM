#!/usr/bin/env python3
"""
Where does the discovery result come from: tuning, pooling, or the method?

    python scripts/protocol_comparison.py

Settings were tuned on coolant against its expert graph. A fair question is how
much of the result is that tuning. And a plant has no expert graph to tune on
at all, so the more useful question is what it would get without one.

Rows, all with the current method (alarms only as effects, screening,
conditioning cap):

  untuned        1 s grid, lag 1, significance 0.05 - generic choices
  label-free     2.5 s grid and lag 3 from the machine's documented 5 s alarm
                 rule; significance chosen by how well the learned model
                 predicts recordings it has not seen - no answer used
  tuned          2.5 s grid, lag 3, significance 0.001 - chosen on coolant

each run pooled and as replicate votes, plus the authors' protocol (every
recording votes alone) at the tuned settings.

Writes results/protocol_comparison.csv.
"""

import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

import pandas as pd

import settings
from causalpdm.pipeline import data, discovery, evaluate, paths, preprocess, tuning

PREDICTION_CANDIDATES = [0.0001, 0.001, 0.01, 0.05]

SUBSYSTEMS = ["coolant", "hydraulics", "probe"]

SETTINGS = {
    "untuned":    {"grid": 1.0, "max_lag": 1, "significance": 0.05},
    "label-free": {"grid": 2.5, "max_lag": 3, "significance": "by prediction"},
    "tuned":      {"grid": 2.5, "max_lag": 3, "significance": 0.001},
}

PROTOCOLS = [("pooled", None), ("replicate vote", "replicates")]


def main() -> int:
    rows = []
    for subsystem in SUBSYSTEMS:
        dataset = data.load(subsystem, normal_recordings=settings.NORMAL_RECORDINGS)
        variables = dataset.expert_graph.variable_names
        method = {"alarms": dataset.expert_graph.alarm_names,
                  "same_moment_partners": settings.SAME_MOMENT_PARTNERS,
                  "max_parents_conditioned": settings.MAX_PARENTS_CONDITIONED}

        runs = [(name, protocol) for name in SETTINGS for protocol in PROTOCOLS]
        runs.append(("tuned", ("authors' protocol", "recording")))
        label_free_choice = None
        for name, (protocol, grouping) in runs:
            chosen = SETTINGS[name]
            tables, labels, _, _ = preprocess.build_tables(
                dataset.fault_runs, dataset.normal_recordings, variables,
                chosen["grid"], settings.MIN_RECORDING_ROWS)
            significance = chosen["significance"]
            if significance == "by prediction":
                if label_free_choice is None:
                    scores = tuning.choose_significance_by_prediction(
                        tables, variables, PREDICTION_CANDIDATES, chosen["max_lag"], **method)
                    label_free_choice = float(scores.loc[
                        scores["held_out_log_likelihood"].idxmax(), "significance"])
                significance = label_free_choice
            options = {"max_lag": chosen["max_lag"], "significance": significance, **method}
            started = time.time()
            if grouping is None:
                graph = discovery.discover(tables, variables, **options)
            else:
                graph = discovery.discover_by_vote(tables, labels, variables,
                                                   grouping=grouping, **options)
            score = evaluate.score_graph(graph.edge_pairs, dataset.expert_graph.edge_pairs)
            rows.append({"subsystem": subsystem, "settings": name, "protocol": protocol,
                         "significance": significance,
                         "f1": score["f1"], "precision": score["precision"],
                         "recall": score["recall"], "shd": score["shd"],
                         "seconds": round(time.time() - started, 1)})
            print(f"  {subsystem:<11} {name:<11} {protocol:<18} F1 {score['f1']:.3f}  "
                  f"P {score['precision']:.2f}  R {score['recall']:.2f}", flush=True)
            pd.DataFrame(rows).to_csv(paths.RESULTS_DIR / "protocol_comparison.csv", index=False)

    table = pd.DataFrame(rows).pivot_table(index=["settings", "protocol"],
                                           columns="subsystem", values="f1")
    print("\nGraph F1\n" + table.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
