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

The few analog signals are then cut into a handful of states, because the
independence test needs discrete data. The cut points are learned once over
all recordings, so the same state means the same thing in every recording.
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


def learn_healthy_values(recordings: list[Path], variables: list[str],
                         fault_runs=()) -> dict[str, float]:
    """
    Work out what each variable looks like when the machine is healthy.

    We read the healthy recordings and take the value each variable spends most
    of its time at. This gives us a sensible value to fill in when a variable is
    absent from a fault recording because it never changed.

    Some signals never change during normal operation, so they are never logged
    in a healthy recording at all. Hydraulic pressure is the case that matters:
    it is absent from all 170, and used to fall back to 0 while the real machine
    sits at 10000. For those signals we read the part of each fault recording
    before the fault was injected, which is healthy by construction.

    Alarms are absent from healthy recordings too, and come out as 0 - "not
    firing" - either way.
    """
    observed: dict[str, Counter] = {name: Counter() for name in variables}

    for path in recordings:
        _count_values(read_change_log(path), variables, observed)

    never_logged = [name for name in variables if not observed[name]]
    for run in fault_runs if never_logged else ():
        log = read_change_log(run.path)
        _count_values(log[log["time_s"] < run.fault_started_at], never_logged, observed)

    healthy = {}
    for name in variables:
        counts = observed[name]
        healthy[name] = counts.most_common(1)[0][0] if counts else 0.0
    return healthy


def _count_values(log: pd.DataFrame, variables: list[str],
                  observed: dict[str, Counter]) -> None:
    log = log[log["node"].isin(variables)]
    for name, group in log.groupby("node"):
        for raw in group["value"]:
            value = parse_value(raw)
            if not np.isnan(value):
                observed[name][value] += 1


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
) -> tuple[list[pd.DataFrame], list[str], dict[str, float], dict[str, np.ndarray]]:
    """
    Build one table per recording, ready for causal discovery.

    Both healthy and fault recordings are included. Healthy ones show the
    machine's normal behaviour; fault ones show what happens when something
    breaks. Causal discovery needs both, because alarm variables never change
    during healthy operation and so carry no information on their own.

    Returns the tables, a label for each one, the healthy value of each
    variable in the same units as the tables, and the cut points used for the
    analog signals (needed to translate any raw value into a table state).
    """
    raw_healthy = learn_healthy_values(normal_recordings, variables, fault_runs)

    raw_tables: list[pd.DataFrame] = []
    labels: list[str] = []

    for path in normal_recordings:
        table = to_table(read_change_log(path), variables, seconds_per_step, raw_healthy)
        if len(table) >= min_rows:
            raw_tables.append(table)
            labels.append(f"normal/{path.stem}")

    for run in fault_runs:
        table = to_table(read_change_log(run.path), variables, seconds_per_step, raw_healthy)
        if len(table) >= min_rows:
            raw_tables.append(table)
            labels.append(f"fault/{run.name}")

    cut_points = learn_cut_points(raw_tables, raw_healthy)
    tables = [discretise(table, cut_points) for table in raw_tables]
    healthy = {name: to_state(name, value, cut_points) for name, value in raw_healthy.items()}
    return tables, labels, healthy, cut_points


def learn_cut_points(tables: list[pd.DataFrame], healthy_values: dict[str, float],
                     max_levels: int = 4) -> dict[str, np.ndarray]:
    """
    Decide once, over every recording, where to cut each analog signal.

    The G-test needs a few states, not a continuous reading. Cutting each
    recording separately would make "state 3" mean a different pressure in
    every recording, so the cut points are learned from all of them together.

    Two kinds of analog signal need different treatment:

      Signals that sit at one value in normal operation - hydraulic pressure
      holds 10000 in about 95% of all rows. Equal-sized bins collapse to a
      single cut there, so these become three states: below healthy, healthy,
      above healthy.

      Signals that genuinely vary in normal operation - spindle speeds,
      positions. These are cut into `max_levels` equal-sized bins.

    Signals that already take only a few values are left exactly as they are.
    """
    pooled = pd.concat(tables, ignore_index=True)
    cut_points = {}
    for column in pooled.columns:
        values = pooled[column].dropna()
        if values.nunique() <= max_levels:
            continue
        healthy = healthy_values.get(column)
        if healthy is not None and (values == healthy).mean() > 0.5:
            cut_points[column] = np.array([healthy, np.nextafter(healthy, np.inf)])
        else:
            quantiles = np.linspace(0, 1, max_levels + 1)[1:-1]
            cut_points[column] = np.unique(np.quantile(values, quantiles))
    return cut_points


def discretise(table: pd.DataFrame, cut_points: dict[str, np.ndarray]) -> pd.DataFrame:
    """Turn every analog signal into its state number, using the shared cut points."""
    out = table.copy()
    for column, cuts in cut_points.items():
        if column in out.columns:
            out[column] = np.searchsorted(cuts, out[column].to_numpy(), side="right").astype(float)
    return out


def to_state(variable: str, value: float, cut_points: dict[str, np.ndarray]) -> float:
    """Translate one raw value into the state the tables use for it."""
    cuts = cut_points.get(variable)
    if cuts is None:
        return float(value)
    return float(np.searchsorted(cuts, value, side="right"))


def forced_state(run, variable: str, cut_points: dict[str, np.ndarray]) -> float:
    """
    The value a fault really forced `variable` to, in table units.

    The labels give True/False for switches and a number for some pressure
    faults; those are used as written, with one exception below. Other
    pressure faults are labelled with a condition - "< 6600", "!= 10000" -
    which is not a value, and "!= 10000" went upwards in some runs and
    downwards in others. For those we use the median value the recording
    actually shows between the fault starting and the first alarm.

    The exception: in exp_10 and exp_15 the labels say the temperature
    switches were forced True, which is their healthy state. In every run the
    recording shows them going False, and the expert graph's own rule says the
    temperature alarms fire on False. Checked across all 100 runs, these are
    the only labels that contradict their recordings. So where a switch
    changed during the fault and never took its labelled value, we trust the
    recording.
    """
    label = run.forced_states[variable]
    log = read_change_log(run.path)
    until = run.alarms_fired_at if run.alarms_fired_at is not None else run.fault_ended_at
    during = log[(log["node"] == variable) & (log["time_s"] >= run.fault_started_at)
                 & (log["time_s"] <= until)]
    values = during["value"].map(parse_value).dropna()

    if isinstance(label, bool):
        if len(values) and float(label) not in set(values):
            return float(values.iloc[-1])
        return float(label)
    if isinstance(label, (int, float)):
        return to_state(variable, float(label), cut_points)

    parsed = parse_value(label)
    if not np.isnan(parsed):
        return to_state(variable, parsed, cut_points)

    if values.empty:
        raise ValueError(f"{run.name}: no logged value for {variable} to resolve '{label}'")
    return to_state(variable, float(values.median()), cut_points)


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
