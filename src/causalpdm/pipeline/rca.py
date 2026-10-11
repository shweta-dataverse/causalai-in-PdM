"""
Finding which variable caused a fault.

When something goes wrong on a machine, many signals look abnormal at once. The
alarm panel lights up, pumps switch off, levels drop. Ranking signals by how
abnormal they look therefore finds the loudest symptom, not the cause - and an
engineer sent to fix a symptom fixes nothing.

The idea that separates the two is simple once stated:

    A symptom is explained by its own causes. A root cause is not.

When the coolant pump fails, the pump alarm firing is completely expected - the
model knew the alarm would fire, because the pump had failed. The alarm is
explained. But the pump failing is not explained by anything upstream of it. It
came out of nowhere as far as the model is concerned.

So we score each variable by how badly its own causes explain its behaviour
during the fault. That score is high exactly for variables something external
acted on, and low for everything merely reacting downstream. The idea is
inspired by Budhathoki et al. (2022), who attribute an outlier to each
variable's own random term. Theirs is a Shapley-based attribution of an
outlier score; ours is simpler - the average surprise of each variable given
its causes over the fault window.

Conveniently, this is also precisely what causRCA's ground truth records: the
digital twin forced certain variables to a value, and those are the ones whose
behaviour their causes cannot account for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .scm import StructuralCausalModel

# Avoid taking the logarithm of zero when a variable does something the model
# considered impossible.
SMALLEST_PROBABILITY = 1e-6


@dataclass
class RootCauseResult:
    """The ranking for one fault, with everything needed to explain it."""

    ranking: pd.DataFrame
    fault_window: tuple[int, int]
    alarms_that_fired: list[str]
    recording_name: str

    @property
    def ranked_variables(self) -> list[str]:
        return self.ranking["variable"].tolist()

    def top(self, k: int = 3) -> list[str]:
        return self.ranked_variables[:k]


def find_fault_window(recording: pd.DataFrame, alarm_names: list[str],
                      look_back_steps: int = 10) -> tuple[int, int, list[str]]:
    """
    Work out when the fault happened, using only what the machine itself knows.

    We look for the first moment any alarm fires, and take a window reaching
    back from there. Reaching back matters: the cause always acts before the
    alarm it triggers, so a window starting at the alarm would miss it.

    Deliberately uses no ground-truth labels, so it works the same way in
    production as it does here.
    """
    present = [name for name in alarm_names if name in recording.columns]
    firing_steps = []
    for name in present:
        values = recording[name].to_numpy(dtype=float)
        fired = np.flatnonzero(values > 0.5)
        if len(fired):
            firing_steps.append(int(fired[0]))

    if not firing_steps:
        # No alarm ever fired: fall back to the last part of the recording.
        return max(0, len(recording) - look_back_steps), len(recording) - 1, []

    first_alarm = min(firing_steps)
    which_fired = [name for name in present
                   if (recording[name].to_numpy(dtype=float) > 0.5).any()]
    return max(0, first_alarm - look_back_steps), min(first_alarm + 2, len(recording) - 1), which_fired


def score_unexplained_behaviour(model: StructuralCausalModel, recording: pd.DataFrame,
                                window: tuple[int, int]) -> pd.DataFrame:
    """
    Score every variable by how poorly its own causes explain it.

    For each moment in the fault window we ask the variable's equation what it
    expected, compare that with what actually happened, and measure the
    surprise. A variable that keeps doing things its causes cannot account for
    is one something external acted on.
    """
    start, end = window
    rows = []

    for name in model.variables:
        mechanism = model.mechanisms.get(name)
        if mechanism is None:
            # No equation - the variable never changed, so it explains nothing.
            rows.append({"variable": name, "unexplained_score": 0.0,
                         "has_equation": False, "changed_in_window": False,
                         "number_of_causes": 0})
            continue

        observed = recording[name].to_numpy(dtype=float)
        cause_series = {cause: recording[cause].to_numpy(dtype=float)
                        for cause, _ in mechanism.causes}

        surprises = []
        for step in range(max(start, model.max_lag), end + 1):
            values = np.array([cause_series[cause][step - lag]
                               for cause, lag in mechanism.causes])
            chance_on = mechanism.probability(values)
            chance_of_what_happened = chance_on if observed[step] > 0.5 else 1 - chance_on
            surprises.append(-np.log(max(chance_of_what_happened, SMALLEST_PROBABILITY)))

        window_values = observed[start:end + 1]
        rows.append({
            "variable": name,
            "unexplained_score": round(float(np.mean(surprises)), 4) if surprises else 0.0,
            "has_equation": True,
            "changed_in_window": bool(len(np.unique(window_values)) > 1),
            "number_of_causes": len(mechanism.causes),
        })

    ranking = pd.DataFrame(rows)
    return ranking.sort_values("unexplained_score", ascending=False, ignore_index=True)


def analyse(model: StructuralCausalModel, recording: pd.DataFrame,
            alarm_names: list[str], recording_name: str = "",
            look_back_steps: int = 10, only_variables_that_moved: bool = True) -> RootCauseResult:
    """
    Run root cause analysis on one fault recording.

    Variables that never moved during the fault window are pushed to the bottom
    of the ranking: whatever caused this fault, it was not something that sat
    perfectly still throughout.
    """
    start, end, fired = find_fault_window(recording, alarm_names, look_back_steps)
    ranking = score_unexplained_behaviour(model, recording, (start, end))

    if only_variables_that_moved:
        ranking["unexplained_score"] = np.where(
            ranking["changed_in_window"], ranking["unexplained_score"],
            ranking["unexplained_score"] * 0.0)
        ranking = ranking.sort_values(
            ["changed_in_window", "unexplained_score"],
            ascending=[False, False], ignore_index=True)

    # Alarms report the fault, they never cause it, so they are not candidates.
    ranking["is_alarm"] = ranking["variable"].isin(alarm_names)
    ranking = ranking.sort_values(["is_alarm", "unexplained_score"],
                                  ascending=[True, False], ignore_index=True)

    return RootCauseResult(ranking=ranking, fault_window=(start, end),
                           alarms_that_fired=fired, recording_name=recording_name)
