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

import pandas as pd

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
