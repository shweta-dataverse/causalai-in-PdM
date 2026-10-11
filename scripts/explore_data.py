#!/usr/bin/env python3
"""
A first look at the causRCA dataset.

Run this before anything else. It answers the basic questions - how much data
is there, what shape is it in, what varies and what does not - and writes a
short report to reports/eda/.

    python scripts/explore_data.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

import settings
from causalpdm.pipeline import data, paths, preprocess


def section(title: str) -> str:
    line = "=" * 78
    return f"\n{line}\n{title}\n{line}"


def main() -> int:
    paths.make_output_folders()
    report: list[str] = []

    def out(text: str = "") -> None:
        print(text)
        report.append(text)

    out(section("causRCA dataset: a first look"))
    out("Industrial CNC vertical lathe. Recordings of normal running, plus faults")
    out("injected deliberately into a digital twin so the true cause is known.")
    out("Source: Mehling, Pieper and Lueke (2025), doi 10.5281/zenodo.15876410")

    out(section("1. What is on disk"))
    normal_files = data.load_normal_recordings()
    out(f"Normal-operation recordings : {len(normal_files)}")
    total_fault = 0
    for name in ("coolant", "hydraulics", "probe"):
        runs = data.load_fault_runs(name)
        total_fault += len(runs)
        graph = data.load_expert_graph(name)
        out(f"  {name:<12} {len(runs):>3} fault runs | "
            f"{len(graph.variable_names):>3} variables | "
            f"{len(graph.edges):>3} expert links | "
            f"{len({r.experiment for r in runs})} scenarios")
    full = data.load_expert_graph("full")
    out(f"  {'full machine':<12} {total_fault:>3} fault runs | "
        f"{len(full.variable_names):>3} variables | {len(full.edges):>3} expert links")

    subsystem = settings.SUBSYSTEM
    dataset = data.load(subsystem, normal_recordings=settings.NORMAL_RECORDINGS)
    out(section(f"2. The {subsystem} subsystem in detail"))
    for key, value in dataset.summary().items():
        out(f"  {key:<22} {value}")

    out("\n  Variables, and what the engineers say they are:")
    for _, row in dataset.expert_graph.variables.iterrows():
        kind = "alarm " if row["type"] == "Alarm" else "signal"
        out(f"    {kind} {row['label']:<20} {str(row['comment'])[:52]}")

    out("\n  The expert causal graph (our ground truth):")
    for _, row in dataset.expert_graph.edges.iterrows():
        out(f"    {row['source_label']:<18} -> {row['target_label']:<18} "
            f"{str(row['comment'])[:44]}")

    out(section("3. How the recordings are stored"))
    out("Each file is a change log, not a regular time series: a row appears only")
    out("at the moment a variable changes value. A variable that never changed")
    out("during a recording does not appear in that file at all.")

    sample = data.read_change_log(dataset.fault_runs[0].path)
    out(f"\n  Example file: {dataset.fault_runs[0].name}")
    out(f"    rows (changes)   {len(sample):,}")
    out(f"    length           {sample['time_s'].max():.0f} seconds")
    out(f"    variables in it  {sample['node'].nunique()}")
    out(f"    columns          {', '.join(sample.columns)}")
    out("\n  First few rows:")
    for line in sample.head(5).to_string(index=False).splitlines():
        out(f"    {line}")

    lengths = []
    for run in dataset.fault_runs:
        log = data.read_change_log(run.path)
        lengths.append({"recording": run.name, "changes": len(log),
                        "seconds": round(float(log["time_s"].max()), 1),
                        "variables": log["node"].nunique()})
    length_table = pd.DataFrame(lengths)
    out("\n  Across all fault recordings:")
    for column in ("changes", "seconds", "variables"):
        values = length_table[column]
        out(f"    {column:<10} smallest {values.min():>8.0f} | "
            f"median {values.median():>8.0f} | largest {values.max():>8.0f}")

    out(section("4. Which variables actually appear"))
    out("Alarms never fire while the machine is healthy, so they are absent from")
    out("every normal recording. This is why the pipeline uses healthy and fault")
    out("recordings together: on their own, the healthy ones carry no information")
    out("about alarms at all.")

    variables = dataset.expert_graph.variable_names
    appears_in_faults = {name: 0 for name in variables}
    for run in dataset.fault_runs:
        present = set(data.read_change_log(run.path)["node"])
        for name in variables:
            appears_in_faults[name] += name in present

    appears_in_normal = {name: 0 for name in variables}
    for path in dataset.normal_recordings[:20]:
        present = set(data.read_change_log(path)["node"])
        for name in variables:
            appears_in_normal[name] += name in present

    out(f"\n    {'variable':<20} {'in fault runs':>14} {'in normal runs':>15}")
    for name in variables:
        out(f"    {name:<20} {appears_in_faults[name]:>8}/{len(dataset.fault_runs):<5} "
            f"{appears_in_normal[name]:>9}/20")

    out(section("5. After turning the change logs into tables"))
    tables, labels, healthy, _ = preprocess.build_tables(
        dataset.fault_runs, dataset.normal_recordings, variables,
        settings.RESAMPLE_SECONDS, settings.MIN_RECORDING_ROWS)
    fault_count = sum(1 for label in labels if label.startswith("fault"))
    out(f"  grid                 {settings.RESAMPLE_SECONDS} seconds per row")
    out(f"  tables built         {len(tables)} "
        f"({len(tables) - fault_count} normal + {fault_count} fault)")
    out(f"  rows per table       median {int(pd.Series([len(t) for t in tables]).median())}")
    out(f"  total rows           {sum(len(t) for t in tables):,}")
    out(f"  columns per table    {len(variables)}")

    out("\n  What each variable looks like when the machine is healthy:")
    for name, value in healthy.items():
        out(f"    {name:<20} {'on' if value > 0.5 else 'off'}")

    out("\n  How much each variable varies once all recordings are pooled:")
    variation = preprocess.describe_tables(tables, variables)
    for line in variation.to_string(index=False).splitlines():
        out(f"    {line}")

    never_varies = variation.loc[~variation["varies"], "variable"].tolist()
    if never_varies:
        out(f"\n  WARNING: these never change and cannot be used: {never_varies}")
    else:
        out("\n  Every variable changes at least once, so all of them are usable.")

    out(section("6. The fault scenarios, and their known causes"))
    scenarios = {}
    for run in dataset.fault_runs:
        scenarios.setdefault(run.experiment, {
            "runs": 0, "caused_by": run.caused_by,
            "alarms": run.expected_alarms, "diagnosis": run.diagnosis})
        scenarios[run.experiment]["runs"] += 1
    for name, info in sorted(scenarios.items()):
        out(f"\n  {name}  ({info['runs']} repeats)")
        out(f"    what was forced  {', '.join(info['caused_by'])}")
        out(f"    alarms expected  {', '.join(info['alarms'])}")
        out(f"    diagnosis        {info['diagnosis']}")

    out(section("Summary"))
    out(f"The {subsystem} subsystem gives us {len(tables)} usable recordings over")
    out(f"{len(variables)} variables, {len(dataset.expert_graph.edges)} known causal links to check our")
    out(f"answers against, and {len(dataset.fault_runs)} faults whose true cause is recorded.")
    out("That is enough to measure whether the pipeline is right, not just whether")
    out("it runs.")

    report_file = paths.EDA_DIR / f"{subsystem}_overview.txt"
    report_file.write_text("\n".join(report))
    print(f"\nreport written to {report_file.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
