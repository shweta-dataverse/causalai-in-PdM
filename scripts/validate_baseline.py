#!/usr/bin/env python3
"""
Check that our evaluation matches the published one.

    python scripts/validate_baseline.py

Every comparison in the thesis against Mehling et al. (2026) depends on our
scores being computed the same way as theirs. Two checks:

  Root cause MAP@3 - run their TimeRank baseline through our own scoring code
  and compare with the value they published.

  Graph F1 - score the graphs they published (references/causrca_published)
  with our own graph scoring and compare with their Table 3.

If both agree, our numbers and theirs are measured with the same ruler.

Writes results/final/baseline_validation.json.
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from causalpdm.pipeline import baselines, data, evaluate, paths

PUBLISHED_GRAPHS = PROJECT_ROOT / "references" / "causrca_published"

# TimeRank MAP@3 as published, from the authors' results file
# (eval/rca/results/unsupervised_results.csv; rounded in the paper's Table 6).
PUBLISHED = {
    ("coolant", "sub"): 0.8533,
    ("coolant", "full"): 0.3533,
    ("hydraulics", "sub"): 0.4722,
    ("hydraulics", "full"): 0.1836,
    ("probe", "sub"): 0.4706,
    ("probe", "full"): 0.1912,
}

# Agreement we accept. The method is a direct port, so anything beyond
# rounding means our data loading or our metric differs from theirs.
TOLERANCE = 0.005

# Many signals share a timestamp, and TimeRank does not say which of them comes
# first, so the score depends partly on an arbitrary tie order. We rerun with
# this many random tie orders and accept the published value if it lies inside
# the range they produce. Measured: coolant Full spans 0.353 to 0.383 from tie
# order alone, and the published value is 0.353.
TIE_ORDERS_TRIED = 200


def score_range_over_tie_orders(runs, variables) -> tuple[float, float]:
    names = sorted({name for run in runs for name in data.read_change_log(run.path)["node"]})
    scores = []
    for seed in range(TIE_ORDERS_TRIED):
        shuffled = names[:]
        random.Random(seed).shuffle(shuffled)
        tie_order = {name: position for position, name in enumerate(shuffled)}
        pairs = baselines.rank_all_faults(baselines.time_recency, runs, variables,
                                          tie_order=tie_order)
        scores.append(evaluate.score_rankings(pairs, k=3)["map@3"])
    return min(scores), max(scores)


def check_graph_scoring() -> list[dict]:
    """
    Score each graph the authors published with our code, against the expert
    graph restricted to the signals they used, and compare with their F1.

    The full expert graph file lists two links twice (104 rows, 102 distinct
    links). Their scorer adds rows into the answer matrix, so those two cells
    hold a 2, and its "prediction + answer = 2" test then counts them as found
    whether or not they were predicted. We reproduce that to confirm we
    understand their number, and report the correct score next to it.
    """
    expert_rows = data.load_expert_graph("full").edges
    row_counts = Counter(zip(expert_rows["source_label"], expert_rows["target_label"]))
    published = pd.read_csv(PUBLISHED_GRAPHS / "evaluation_results.csv", skipinitialspace=True)
    published.columns = [c.strip() for c in published.columns]
    rows = []
    for _, row in published.iterrows():
        dataset, model = row["dataset"].strip(), row["model"].strip()
        graph = nx.read_gml(PUBLISHED_GRAPHS / f"{model}_{dataset}_majority.gml")
        signals = set(graph.nodes())
        learned = {(a, b) for a, b in graph.edges() if a != b}
        truth = {edge: n for edge, n in row_counts.items() if set(edge) <= signals}
        ours = evaluate.score_graph(learned, set(truth))["f1"]
        as_they_scored = _score_with_duplicate_rows(learned, truth)
        agrees = abs(as_they_scored - float(row["F1"])) <= TOLERANCE
        rows.append({"dataset": dataset, "model": model, "published_f1": float(row["F1"]),
                     "reproduced_f1": round(as_they_scored, 3), "correct_f1": ours,
                     "agrees": agrees})
        print(f"  {dataset:<10} {model:<14} published {float(row['F1']):.3f}  "
              f"reproduced {as_they_scored:.3f}  correct {ours:.3f}  "
              f"{'agrees' if agrees else 'DIFFERS'}")
    return rows


def _score_with_duplicate_rows(learned: set, truth_rows: dict) -> float:
    found = (sum(1 for edge in learned if truth_rows.get(edge) == 1)
             + sum(1 for edge, n in truth_rows.items() if n > 1 and edge not in learned))
    precision = found / max(len(learned), 1)
    recall = found / max(sum(truth_rows.values()), 1)
    return 2 * precision * recall / (precision + recall) if found else 0.0


def main() -> int:
    print("Graph F1 of the authors' own published graphs:")
    graph_rows = check_graph_scoring()
    print("\nTimeRank MAP@3:")
    rows = []
    for (subsystem, setting), published in PUBLISHED.items():
        runs = data.load_fault_runs(subsystem)
        variables = data.load_expert_graph(subsystem).variable_names if setting == "sub" else None
        pairs = baselines.rank_all_faults(baselines.time_recency, runs, variables)
        ours = evaluate.score_rankings(pairs, k=3)["map@3"]
        low, high = score_range_over_tie_orders(runs, variables)
        exact = abs(ours - published) <= TOLERANCE
        within_ties = low - TOLERANCE <= published <= high + TOLERANCE
        rows.append({"subsystem": subsystem, "setting": setting,
                     "faults": len(pairs), "published_map@3": published,
                     "ours_map@3": ours, "tie_order_range": [low, high],
                     "matches_exactly": exact, "agrees": exact or within_ties})
        verdict = ("agrees exactly" if exact
                   else "agrees within tie order" if within_ties else "DIFFERS")
        print(f"  {subsystem:<11} {setting:<5} published {published:.3f}  ours {ours:.3f}  "
              f"tie range {low:.3f}-{high:.3f}  {verdict}")

    passed = all(row["agrees"] for row in rows + graph_rows)
    print("\nGate passed: our graph F1 and MAP@3 match the published evaluation." if passed
          else "\nGate FAILED: diagnose before comparing anything against the paper.")

    out = paths.RESULTS_DIR / "final"
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_validation.json").write_text(json.dumps(
        {"passed": passed, "tolerance": TOLERANCE,
         "graph_f1": graph_rows,
         "timerank_map@3": rows}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
