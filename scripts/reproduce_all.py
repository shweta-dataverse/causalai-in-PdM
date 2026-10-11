#!/usr/bin/env python3
"""
Regenerate every number in the thesis, in order.

    python scripts/reproduce_all.py

Runs each script below, then copies the final tables into results/final/,
which is kept in git. Takes about an hour; causal discovery is cached, so a
second run is much faster.

Not included: tune_settings.py (its table is already in results/final/ and only
documents how the settings were chosen), and the full 92-signal machine, which
does not finish (see results/final/full_graph_scaling.log).
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
FINAL = RESULTS / "final"

STEPS = [
    ("validate_baseline.py", "our scoring reproduces the published numbers"),
    ("run_all_subsystems.py", "main results, every subsystem"),
    ("held_out_scenarios.py", "fault types never seen in training"),
    ("protocol_comparison.py", "tuned vs untuned vs label-free, pooled vs voted"),
    ("compare_root_cause.py", "root cause analysis vs the published methods"),
    ("minimal_fixes.py", "smallest fix that would have prevented each fault"),
    ("ablation.py", "what each design decision is worth"),
]

TABLES = [
    "all_subsystems.csv", "held_out_scenarios.csv", "held_out_by_scenario.csv",
    "protocol_comparison.csv", "root_cause_comparison.csv", "minimal_fixes.csv",
    "minimal_fixes_by_fault.csv", "ablation.csv",
]


def main() -> int:
    for script, purpose in STEPS:
        print(f"\n=== {script}: {purpose}", flush=True)
        started = time.time()
        finished = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / script)])
        print(f"=== {script} finished in {time.time() - started:.0f}s", flush=True)
        if finished.returncode != 0:
            print(f"Stopped: {script} failed.")
            return finished.returncode

    FINAL.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        shutil.copy(RESULTS / name, FINAL / name)
    for subsystem in ("coolant", "hydraulics", "probe"):
        target = FINAL / subsystem
        target.mkdir(exist_ok=True)
        for path in (RESULTS / subsystem).glob("*"):
            shutil.copy(path, target / path.name)
    print(f"\nFinal tables copied to {FINAL.relative_to(PROJECT_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
