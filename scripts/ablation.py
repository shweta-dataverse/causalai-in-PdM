#!/usr/bin/env python3
"""
Which design decision earns which part of the result.

    python scripts/ablation.py

The full method is run once, then again with one decision switched off at a
time, everything else held fixed. On every subsystem.

  no alarm rule          alarms may be causes too
  no FDR correction      no correction for testing thousands of links
  continuous test        partial correlation on True/False data
  untuned settings       1 s grid, lag 1, significance 0.05
  no suspect restriction every signal considered equally in root cause analysis
  single model           the honest model also decides whether a variable
                         switches on in simulation (no class-balanced model)

Same-moment screening is not listed: it changes nothing on any subsystem
(measured, see settings.py). Midpoint vs sampled hidden randomness is in every
row already - the pipeline reports both.

Writes results/ablation.csv.
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

SUBSYSTEMS = ["coolant", "hydraulics", "probe"]

FULL_METHOD = dict(
    normal_recordings=settings.NORMAL_RECORDINGS, seconds_per_step=settings.RESAMPLE_SECONDS,
    max_lag=settings.MAX_LAG, significance=settings.SIGNIFICANCE,
    independence_test=settings.INDEPENDENCE_TEST,
    multiple_test_correction=settings.MULTIPLE_TEST_CORRECTION,
    min_rows=settings.MIN_RECORDING_ROWS, top_k=settings.TOP_K_ROOT_CAUSES,
    alarms_only_as_effects=settings.ALARMS_ONLY_AS_EFFECTS,
    same_moment_partners=settings.SAME_MOMENT_PARTNERS,
    max_parents_conditioned=settings.MAX_PARENTS_CONDITIONED,
    suspects=settings.ROOT_CAUSE_SUSPECTS, use_cache=settings.USE_CACHE)

ROWS = {
    "full method": {},
    "no alarm rule": {"alarms_only_as_effects": False},
    "no FDR correction": {"multiple_test_correction": "none"},
    "continuous test": {"independence_test": "continuous"},
    "untuned settings": {"seconds_per_step": 1.0, "max_lag": 1, "significance": 0.05},
    "no suspect restriction": {"suspects": "all"},
    "single model": {"balanced_decisions": False},
}


def main() -> int:
    rows = []
    for subsystem in SUBSYSTEMS:
        for name, change in ROWS.items():
            result = runner.run(subsystem=subsystem, verbose=False, **{**FULL_METHOD, **change})
            headline = result.headline()
            rows.append({
                "subsystem": subsystem, "change": name,
                "graph_f1": headline["graph_f1"],
                "root_cause_map@3": headline["root_cause_map@3"],
                "root_cause_top1": headline["root_cause_top1"],
                "intervention_f1": headline["intervention_f1"],
                "prevented_by_our_fix": headline["counterfactual_success"],
            })
            print(f"  {subsystem:<11} {name:<23} graph F1 {headline['graph_f1']:.3f}  "
                  f"MAP@3 {headline['root_cause_map@3']:.3f}  intervention F1 "
                  f"{headline['intervention_f1']:.3f}  prevented {headline['counterfactual_success']:.3f}",
                  flush=True)
            pd.DataFrame(rows).to_csv(paths.RESULTS_DIR / "ablation.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
