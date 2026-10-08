#!/usr/bin/env python3
"""
Run the whole causal root cause analysis pipeline.

    python scripts/run_pipeline.py

Settings come from settings.py in the project root. There are no command-line
options to remember - edit that file instead.

Results are written to results/ and picked up by the dashboard.
"""

import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

import settings
from causalpdm.pipeline import runner


def main() -> int:
    print(f"Causal root cause analysis - {settings.SUBSYSTEM} subsystem of the CNC lathe\n")

    result = runner.run(
        subsystem=settings.SUBSYSTEM,
        normal_recordings=settings.NORMAL_RECORDINGS,
        seconds_per_step=settings.RESAMPLE_SECONDS,
        max_lag=settings.MAX_LAG,
        significance=settings.SIGNIFICANCE,
        independence_test=settings.INDEPENDENCE_TEST,
        multiple_test_correction=settings.MULTIPLE_TEST_CORRECTION,
        min_rows=settings.MIN_RECORDING_ROWS,
        top_k=settings.TOP_K_ROOT_CAUSES,
        use_cache=settings.USE_CACHE,
    )
    runner.save(result)

    print("\n" + "=" * 70)
    print("HOW WELL DID IT DO")
    print("=" * 70)

    graph = result.graph_score
    print(f"\nFinding the causal graph (checked against the engineers' own graph)")
    print(f"  links the engineers drew      {graph['edges_expected']}")
    print(f"  links we found                {graph['edges_found']}")
    print(f"  of those, correct             {graph['correct']}")
    print(f"  precision / recall / F1       {graph['precision']} / {graph['recall']} / {graph['f1']}")

    rca = result.root_cause_score
    print(f"\nNaming the cause of each fault (checked against what was really forced)")
    print(f"  faults analysed               {rca['faults_scored']}")
    print(f"  right cause ranked first      {rca['found_in_top_1']:.0%} of the time")
    print(f"  MAP@3                         {rca['map@3']}")

    action = result.intervention_score
    print(f"\nPredicting what a fault would do (checked against the real injected faults)")
    print(f"  fault scenarios replayed      {action['scenarios_checked']}")
    print(f"  alarms predicted correctly    F1 {action['mean_f1']}")

    what_if = result.counterfactual_score
    print(f"\nSaying what would have prevented each fault")
    print(f"  faults examined               {what_if['faults_examined']}")
    print(f"  fixing top suspect helped     {what_if['top_suspect_silenced_an_alarm']}")
    print(f"  fixing the real causes worked {what_if['known_causes_prevented_fault']} "
          f"({what_if['known_causes_success_rate']:.0%})")

    print(f"\nFinished in {result.seconds_taken:.1f} seconds.")
    print("Results written to results/. To see them:  streamlit run dashboard/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
