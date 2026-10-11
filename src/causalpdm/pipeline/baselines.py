"""
The published root cause baselines, re-implemented so we can run them here.

Comparing our numbers against numbers printed in a paper only works if both
were measured the same way. The safest check is to run one of their methods
through our own evaluation code and see whether we get their number back. If
we do, our metric and our reading of the data match theirs.

Methods follow the causRCA authors' own code (github.com/causalgraph/causRCA,
src/causrca/rca_models/unsupervised_rca_models.py, Apache-2.0), rewritten here
rather than copied, and described in Mehling et al., Procedia CIRP 139 (2026).

Like theirs, these work on the raw change logs, not on our resampled tables,
and they diagnose at the labelled diagnosis time.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .data import read_change_log


def state_at(log: pd.DataFrame, moment: float) -> pd.DataFrame:
    """The last recorded change of every signal at or before `moment`."""
    before = log[log["time_s"] <= moment]
    return before.groupby("node").last().reset_index()


def time_recency(log: pd.DataFrame, diagnosis_time: float,
                 variables: list[str] | None = None,
                 tie_order: dict[str, int] | None = None) -> list[str]:
    """
    TimeRank: suspect whatever changed most recently before an alarm.

    For each alarm, newest first, list the non-alarm signals that last changed
    before it, newest first. Then take the first entry of every alarm's list,
    then the second of every list, and so on, skipping repeats.

    `variables` limits the signals considered - the subsystem's own signals in
    the "Sub" setting, everything in the recording in the "Full" setting.

    Several signals often change at the same moment (every signal is logged at
    t = 0), and the method does not say which of them comes first. By default
    the ties fall exactly as in the authors' code, which re-sorts each list
    with pandas' default sort. `tie_order` breaks them by a given ranking
    instead, which is how we measure how much the ties alone move the score.
    """
    if variables is not None:
        log = log[log["node"].isin(variables)]
    state = _newest_first(state_at(log, diagnosis_time), tie_order)
    signals = state[state["type"] != "Alarm"]
    alarms = _newest_first(state[state["type"] == "Alarm"], tie_order)

    if alarms.empty:
        return signals["node"].tolist()

    per_alarm = [_newest_first(signals[signals["time_s"] < alarm_time], tie_order)["node"].tolist()
                 for alarm_time in alarms["time_s"]]

    ranking: list[str] = []
    for position in range(max(len(changes) for changes in per_alarm)):
        for changes in per_alarm:
            if position < len(changes) and changes[position] not in ranking:
                ranking.append(changes[position])
    return ranking


def _newest_first(state: pd.DataFrame, tie_order: dict[str, int] | None) -> pd.DataFrame:
    if tie_order is None:
        return state.sort_values("time_s", ascending=False)
    ranked = state.assign(_tie=state["node"].map(tie_order))
    return ranked.sort_values(["time_s", "_tie"], ascending=[False, True]).drop(columns="_tie")


def rank_all_faults(method, fault_runs, variables: list[str] | None = None, **options):
    """
    Run a baseline on every fault and pair each ranking with the true causes.

    Faults the method returns nothing for are left out, as in the authors'
    evaluation, so the averages are taken over the same faults as theirs.
    """
    pairs = []
    for run in fault_runs:
        ranking = method(read_change_log(run.path), run.diagnosed_at, variables, **options)
        if ranking:
            pairs.append((ranking, run.caused_by))
    return pairs


def resample_for_baro(log: pd.DataFrame, step_seconds: float = 0.5) -> pd.DataFrame:
    """
    The authors' resampling for Baro: a 0.5 s grid, each value carried forward,
    and the first recorded value also used before it. True/False become 1/0.
    """
    values = log.assign(value=log["value"].map(
        lambda v: 1.0 if str(v) == "True" else 0.0 if str(v) == "False" else pd.to_numeric(v, errors="coerce")))
    grid = np.arange(0, values["time_s"].max() + step_seconds, step_seconds)
    table = pd.DataFrame(index=grid)
    for node in values["node"].unique():
        series = values[values["node"] == node].sort_values("time_s")
        placed = pd.Series(series["value"].to_numpy(), index=series["time_s"].to_numpy())
        placed = placed[~placed.index.duplicated(keep="last")].reindex(grid, method="pad")
        table[node] = placed.fillna(series["value"].iloc[0])
    return table


def baro(log: pd.DataFrame, diagnosis_time: float,
         variables: list[str] | None = None) -> list[str]:
    """
    Baro: rank signals by how far they leave their normal range after the alarm.

    The recording is split one second before the oldest alarm still active at
    diagnosis time. Each signal's normal period is robust-scaled (median and
    interquartile range), and its score is the largest scaled deviation in the
    abnormal period. Alarms are not ranked.
    """
    if variables is not None:
        log = log[log["node"].isin(variables)]
    if log.empty:
        return []
    table = resample_for_baro(log)
    state = state_at(log, diagnosis_time)
    active = state[(state["type"] == "Alarm") & (state["value"].astype(str) == "True")]
    if active.empty:
        split = diagnosis_time
    else:
        target = active["time_s"].min() - 1.0
        split = table.index[np.abs(table.index.to_numpy() - target).argmin()]
    normal = table[table.index < split]
    abnormal = table[(table.index >= split) & (table.index <= diagnosis_time)]
    if normal.empty or abnormal.empty:
        return []

    scores = []
    for column in table.columns:
        before, after = normal[column].dropna(), abnormal[column].dropna()
        if before.empty or after.empty:
            continue
        scaler = RobustScaler().fit(before.to_numpy().reshape(-1, 1))
        scaled = scaler.transform(after.to_numpy().reshape(-1, 1))[:, 0]
        scores.append((column, float(np.max(np.abs(scaled)))))
    alarms = set(log.loc[log["type"] == "Alarm", "node"])
    ranked = sorted(scores, key=lambda pair: pair[1], reverse=True)
    return [name for name, _ in ranked if name not in alarms]


def caustr(log: pd.DataFrame, diagnosis_time: float,
           variables: list[str] | None = None,
           graph: set[tuple[str, str]] | None = None) -> list[str]:
    """
    CausTR: TimeRank, restricted first to the alarms active at diagnosis time
    and their direct causes in `graph`.

    As in the authors' code, three rankings are joined in order, without
    repeats: TimeRank over the causal candidates within the subsystem, over
    the causal candidates alone, and over the subsystem.
    """
    state = state_at(log, diagnosis_time)
    active = state[(state["type"] == "Alarm") & (state["value"].astype(str) == "True")]
    nodes_in_graph = {name for edge in graph for name in edge}
    alarms = [name for name in active["node"] if name in nodes_in_graph]
    if not alarms:
        return time_recency(log, diagnosis_time, variables)
    causal = set(alarms) | {cause for cause, effect in graph if effect in alarms}
    within = list(causal & set(variables)) if variables is not None else list(causal)

    ranking: list[str] = []
    for chosen in (within, list(causal), variables):
        for name in time_recency(log, diagnosis_time, chosen):
            if name not in ranking:
                ranking.append(name)
    return ranking


def pagerank(log: pd.DataFrame, diagnosis_time: float,
             variables: list[str] | None = None,
             graph: set[tuple[str, str]] | None = None) -> list[str]:
    """
    PageRank: rank signals by their importance in the causal graph, reversed
    so that importance flows from effects back to causes. Ignores the
    recording apart from which signals appear in it. Alarms are not ranked.
    """
    if variables is not None:
        log = log[log["node"].isin(variables)]
    present = set(log["node"])
    network = nx.DiGraph()
    network.add_nodes_from(sorted(present & {name for edge in graph for name in edge}))
    network.add_edges_from((a, b) for a, b in graph if a in network and b in network)
    if network.number_of_nodes() == 0:
        return sorted(present)
    scores = nx.pagerank(network.reverse(), alpha=0.85)
    alarms = set(log.loc[log["type"] == "Alarm", "node"])
    ranked = sorted(sorted(scores), key=lambda name: scores[name], reverse=True)
    return [name for name in ranked if name not in alarms]
