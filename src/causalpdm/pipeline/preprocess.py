"""
Turning change logs into tables.

causRCA stores each recording as a list of value changes: a row appears only at
the moment a variable changed. Causal discovery needs the opposite shape - a
regular table with one row per time step and one column per variable.

Two things make this harder than a plain resample:

1. A variable that never changed during a recording does not appear in the file
   at all. We cannot tell from the file whether it was missing or simply quiet.
   We fill those gaps using the variable's normal healthy value, learned from
   the healthy recordings.

2. Values are written only when they change, so between two rows the variable
   holds its last value. Forward-filling is therefore the correct thing to do -
   it matches how the machine controller actually publishes data.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .data import read_change_log

# How the raw files write boolean values.
TRUE_VALUES = {"True", "true", "TRUE", "1", "1.0"}
FALSE_VALUES = {"False", "false", "FALSE", "0", "0.0"}


def parse_value(raw: str) -> float:
    """
    Convert one recorded value into a number.

    Booleans become 1 and 0. Numbers stay numbers. Anything else (text states)
    becomes NaN here and is turned into a category code later.
    """
    text = str(raw)
    if text in TRUE_VALUES:
        return 1.0
    if text in FALSE_VALUES:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return np.nan


def learn_healthy_values(recordings: list[Path], variables: list[str]) -> dict[str, float]:
    """
    Work out what each variable looks like when the machine is healthy.

    We read the healthy recordings and take the value each variable spends most
    of its time at. This gives us a sensible value to fill in when a variable is
    absent from a fault recording because it never changed.

    Alarms never fire during healthy operation, so they never appear in those
    files at all - they default to 0, meaning "not firing", which is correct.
    """
    observed: dict[str, Counter] = {name: Counter() for name in variables}

    for path in recordings:
        log = read_change_log(path)
        log = log[log["node"].isin(variables)]
        for name, group in log.groupby("node"):
            for raw in group["value"]:
                value = parse_value(raw)
                if not np.isnan(value):
                    observed[name][value] += 1

    healthy = {}
    for name in variables:
        counts = observed[name]
        healthy[name] = counts.most_common(1)[0][0] if counts else 0.0
    return healthy


def to_table(
    change_log: pd.DataFrame,
    variables: list[str],
    seconds_per_step: float,
    healthy_values: dict[str, float],
) -> pd.DataFrame:
    """
    Convert one change log into a regular table.

    Returns a DataFrame indexed by time in seconds, with one column per
    variable. Every requested variable gets a column, even if it never appeared
    in this recording.
    """
    log = change_log[change_log["node"].isin(variables)].copy()
    log["parsed"] = log["value"].map(parse_value)

    duration = float(change_log["time_s"].max())
    grid = np.arange(0.0, duration + seconds_per_step, seconds_per_step)
    table = pd.DataFrame(index=grid, columns=variables, dtype=float)
    table.index.name = "time_s"

    for name in variables:
        changes = log[log["node"] == name]
        if changes.empty:
            # Variable never changed in this recording: hold it at its healthy value.
            table[name] = healthy_values.get(name, 0.0)
            continue

        # Place each change at the first grid point at or after it happened,
        # then carry the value forward until the next change.
        series = pd.Series(changes["parsed"].to_numpy(), index=changes["time_s"].to_numpy())
        series = series[~series.index.duplicated(keep="last")]
        placed = series.reindex(series.index.union(grid)).ffill().reindex(grid)

        # Anything before the first recorded change takes the healthy value.
        table[name] = placed.fillna(healthy_values.get(name, 0.0)).to_numpy()

    return table


def build_tables(
    fault_runs,
    normal_recordings: list[Path],
    variables: list[str],
    seconds_per_step: float,
    min_rows: int,
) -> tuple[list[pd.DataFrame], list[str], dict[str, float]]:
    """
    Build one table per recording, ready for causal discovery.

    Both healthy and fault recordings are included. Healthy ones show the
    machine's normal behaviour; fault ones show what happens when something
    breaks. Causal discovery needs both, because alarm variables never change
    during healthy operation and so carry no information on their own.

    Returns the tables, a label for each one, and the healthy values used.
    """
    healthy_values = learn_healthy_values(normal_recordings, variables)

    tables: list[pd.DataFrame] = []
    labels: list[str] = []

    for path in normal_recordings:
        table = to_table(read_change_log(path), variables, seconds_per_step, healthy_values)
        if len(table) >= min_rows:
            tables.append(discretise(table))
            labels.append(f"normal/{path.stem}")

    for run in fault_runs:
        table = to_table(read_change_log(run.path), variables, seconds_per_step, healthy_values)
        if len(table) >= min_rows:
            tables.append(discretise(table))
            labels.append(f"fault/{run.name}")

    return tables, labels, healthy_values


def discretise(table: pd.DataFrame, max_levels: int = 4) -> pd.DataFrame:
    """
    Bin any signal that is not already a small set of states.

    The coolant and probe subsystems are entirely true/false, but the full
    machine includes a handful of analog readings - spindle speeds, cycle
    times, hydraulic pressure. The G-test needs a small number of states, so
    those are cut into equal-sized bins. Signals that already take only a few
    values are left exactly as they are.
    """
    out = table.copy()
    for column in out.columns:
        if out[column].nunique() <= max_levels:
            continue
        try:
            out[column] = pd.qcut(out[column], max_levels, labels=False,
                                  duplicates="drop").astype(float)
        except ValueError:
            out[column] = pd.cut(out[column], max_levels, labels=False).astype(float)
        out[column] = out[column].fillna(0.0)
    return out


def describe_tables(tables: list[pd.DataFrame], variables: list[str]) -> pd.DataFrame:
    """How much each variable actually varies once pooled - a sanity check.

    A variable that never changes across every recording carries no information
    and cannot take part in causal discovery, so it is worth spotting early.
    """
    pooled = pd.concat(tables, ignore_index=True)
    rows = []
    for name in variables:
        values = pooled[name]
        rows.append({
            "variable": name,
            "distinct_values": int(values.nunique()),
            "share_of_time_on": float((values > 0).mean()),
            "varies": bool(values.nunique() > 1),
        })
    return pd.DataFrame(rows)
