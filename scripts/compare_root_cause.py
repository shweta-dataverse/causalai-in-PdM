#!/usr/bin/env python3
"""
Root cause analysis compared with the published methods, run here.

    python scripts/compare_root_cause.py

Run after scripts/run_all_subsystems.py - it reads our saved results. Every
method is scored with the same code on the same faults, in the "Sub" setting
(each subsystem's own signals), as in Mehling et al. (2026), Table 6.

  TimeRank, Baro          no causal graph
  PageRank, CausTR        given the engineers' expert graph
  CausTR (learned graph)  the same method on OUR learned graph - separates
                          what our graph is worth from what our scoring adds
  ours                    learned graph, no labels at all

The published methods diagnose at the labelled diagnosis time; ours uses only
the first alarm seen in the data. Their published values are printed for
reference; the ported versions reproduce them (see validate_baseline.py),
except PageRank on hydraulics (1.00 here vs 0.98), from using networkx's
PageRank where the authors used scikit-network.

Writes results/root_cause_comparison.csv.
"""

import pickle
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

warnings.filterwarnings("ignore")

import pandas as pd

from causalpdm.pipeline import baselines, data, evaluate, paths

SAVED = {"coolant": "pipeline_result.pkl",
         "hydraulics": "pipeline_result_hydraulics.pkl",
         "probe": "pipeline_result_probe.pkl"}
PUBLISHED = pd.read_csv(PROJECT_ROOT / "references" / "causrca_published" /
                        "unsupervised_results.csv").set_index("model")
PUBLISHED_NAME = {"TimeRank": "TimeRecency_BaselineUnsupervisedRCA", "Baro": "Baro",
                  "PageRank (expert graph)": "PageRankRCA",
                  "CausTR (expert graph)": "CausalPrioTimeRecencyRCA"}


def main() -> int:
    rows = []
    for subsystem, saved in SAVED.items():
        result = pickle.loads((paths.CACHE_DIR / saved).read_bytes())
        runs = data.load_fault_runs(subsystem)
        expert = data.load_expert_graph(subsystem)
        learned = result.causal_graph.edge_pairs
        methods = {
            "TimeRank": (baselines.time_recency, {}),
            "Baro": (baselines.baro, {}),
            "PageRank (expert graph)": (baselines.pagerank, {"graph": expert.edge_pairs}),
            "CausTR (expert graph)": (baselines.caustr, {"graph": expert.edge_pairs}),
            "CausTR (our learned graph)": (baselines.caustr, {"graph": learned}),
        }
        for name, (method, options) in methods.items():
            pairs = baselines.rank_all_faults(method, runs, expert.variable_names, **options)
            rows.append(_row(subsystem, name, pairs))
        ours = [(ranking.ranked_variables, run.caused_by) for run, ranking in result.root_cause_results]
        rows.append(_row(subsystem, "ours (learned graph, no labels)", ours))
        for row in rows[-6:]:
            print(f"  {subsystem:<11} {row['method']:<33} MAP@3 {row['map@3']:.3f}  "
                  f"top-1 {row['top1']:.2f}   published {row['published_map@3']}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(paths.RESULTS_DIR / "root_cause_comparison.csv", index=False)
    print("\nMAP@3\n" + table.pivot(index="method", columns="subsystem", values="map@3")
          .reindex(columns=list(SAVED)).to_string())
    return 0


def _row(subsystem: str, method: str, pairs) -> dict:
    score = evaluate.score_rankings(pairs, k=3)
    published = PUBLISHED_NAME.get(method)
    return {"subsystem": subsystem, "method": method, "faults": score["faults_scored"],
            "map@3": score["map@3"], "top1": score["found_in_top_1"],
            "published_map@3": round(float(PUBLISHED.loc[published, f"exp_{subsystem}-sub_MAP@3"]), 3)
            if published else None}


if __name__ == "__main__":
    raise SystemExit(main())
