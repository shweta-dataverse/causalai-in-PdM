"""
Answering "what if" questions, and checking the answers.

Two different questions get asked about a fault, and they need different tools.

  Intervention   "If we hold this valve open, what happens?"
                 A forward-looking question about what to do next. We force the
                 variable to a value and let the model play out.

  Counterfactual "Would this machine have failed if the pump had stayed on?"
                 A backward-looking question about one specific breakdown. We
                 first recover what was unusual about this particular run, then
                 change one thing and replay with everything else held as it
                 really was.

**The part that is unusual here**

Counterfactual claims are normally impossible to check. You cannot rewind a
machine and run the week again, so papers generate counterfactuals and ask the
reader to trust them.

causRCA is different. Its faults were injected deliberately: the digital twin
forced a known variable to a known value at a known moment, and recorded what
followed. So we can take a healthy recording, apply the same forced change in
our model, and compare our prediction against the breakdown that actually
happened. That is a real test of whether the model's what-if answers are right,
and it is what `validate_intervention` below does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import scm
from .scm import StructuralCausalModel


@dataclass
class InterventionResult:
    """What the model predicts will happen if we force a variable to a value."""

    held_fixed: dict[str, float]
    from_step: int
    before: pd.DataFrame
    after: pd.DataFrame
    alarm_changes: pd.DataFrame

    @property
    def alarms_prevented(self) -> list[str]:
        stopped = self.alarm_changes["fired_before"] & ~self.alarm_changes["fired_after"]
        return self.alarm_changes.loc[stopped, "alarm"].tolist()

    @property
    def alarms_caused(self) -> list[str]:
        started = ~self.alarm_changes["fired_before"] & self.alarm_changes["fired_after"]
        return self.alarm_changes.loc[started, "alarm"].tolist()


@dataclass
class CounterfactualResult:
    """
    What would have happened if one thing had been different.

    Faults in causRCA usually have more than one cause - a leak trips both the
    pump and the tank level, for example. So fixing a single cause normally
    silences the alarm belonging to that cause and leaves the others ringing.
    That is the right answer, not a failure, which is why prevention is
    reported as "which alarms stopped" rather than a single yes or no.
    """

    changed: dict[str, float]
    from_step: int
    actual: pd.DataFrame
    what_would_have_happened: pd.DataFrame
    alarm_changes: pd.DataFrame
    alarms_prevented: list[str]
    alarms_still_firing: list[str]
    prevented_everything: bool
    prevented_something: bool
    explanation: str

    @property
    def prevented_the_fault(self) -> bool:
        """True only if no alarm fires at all afterwards."""
        return self.prevented_everything


def _compare_alarms(before: pd.DataFrame, after: pd.DataFrame,
                    alarm_names: list[str], from_step: int) -> pd.DataFrame:
    """Which alarms fired before the change, and which fire after it."""
    rows = []
    for alarm in alarm_names:
        if alarm not in before.columns:
            continue
        fired_before = bool((before[alarm].to_numpy(dtype=float)[from_step:] > 0.5).any())
        fired_after = bool((after[alarm].to_numpy(dtype=float)[from_step:] > 0.5).any())
        rows.append({"alarm": alarm, "fired_before": fired_before,
                     "fired_after": fired_after,
                     "changed": fired_before != fired_after})
    return pd.DataFrame(rows)


def intervene(model: StructuralCausalModel, recording: pd.DataFrame,
              hold_fixed: dict[str, float], alarm_names: list[str],
              from_step: int = 0) -> InterventionResult:
    """
    Force one or more variables to a value and see what the model predicts.

    This is the do-operation. It is not the same as filtering the data to times
    when the variable happened to have that value: we are overriding the
    variable, so whatever normally drives it no longer applies.
    """
    before = scm.simulate(model, recording, hidden_randomness=None)
    after = scm.simulate(model, recording, hidden_randomness=None,
                         hold_fixed=hold_fixed, from_step=from_step)
    return InterventionResult(
        held_fixed=hold_fixed, from_step=from_step, before=before, after=after,
        alarm_changes=_compare_alarms(before, after, alarm_names, from_step))


def ask_what_if(model: StructuralCausalModel, recording: pd.DataFrame,
                change: dict[str, float], alarm_names: list[str],
                from_step: int = 0, describe: str = "") -> CounterfactualResult:
    """
    Ask whether changing one thing would have prevented this breakdown.

    The recording's own hidden randomness is recovered first and then held
    fixed, so the only difference between what really happened and what we
    predict is the change we made. Without that step we would be comparing this
    machine against an average machine, and any difference could just be the
    difference between machines.
    """
    hidden = scm.recover_hidden_randomness(model, recording)

    actual = scm.simulate(model, recording, hidden_randomness=hidden)
    what_if = scm.simulate(model, recording, hidden_randomness=hidden,
                           hold_fixed=change, from_step=from_step)

    comparison = _compare_alarms(actual, what_if, alarm_names, from_step)
    stopped = comparison.loc[
        comparison["fired_before"] & ~comparison["fired_after"], "alarm"].tolist()
    still_firing = comparison.loc[comparison["fired_after"], "alarm"].tolist()
    fired_before = bool(comparison["fired_before"].any())

    prevented_everything = bool(fired_before and not still_firing)
    prevented_something = bool(stopped)

    changes = ", ".join(
        f"{name} {'on' if value > 0.5 else 'off'}" for name, value in change.items())

    if prevented_everything:
        explanation = (
            f"Holding {changes} would have prevented this fault completely. "
            f"Every alarm that fired in reality ({', '.join(stopped)}) would "
            f"have stayed silent."
        )
    elif prevented_something:
        explanation = (
            f"Holding {changes} would have stopped {', '.join(stopped)}, but "
            f"{', '.join(still_firing)} would still have fired. Something else "
            f"is also wrong, so this fix on its own is not enough."
        )
    elif fired_before:
        explanation = (
            f"Holding {changes} would have changed nothing. The same alarms "
            f"({', '.join(still_firing)}) still fire, so this is not the cause."
        )
    else:
        explanation = "No alarms fired in this recording, so there was nothing to prevent."

    if describe:
        explanation = f"{describe} {explanation}"

    return CounterfactualResult(
        changed=change, from_step=from_step, actual=actual,
        what_would_have_happened=what_if, alarm_changes=comparison,
        alarms_prevented=stopped, alarms_still_firing=still_firing,
        prevented_everything=prevented_everything,
        prevented_something=prevented_something,
        explanation=explanation)


def validate_intervention(model: StructuralCausalModel, healthy_recording: pd.DataFrame,
                          forced: dict[str, float],
                          alarms_that_really_fired: list[str], alarm_names: list[str],
                          from_step: int = 5) -> dict:
    """
    Check the model's what-if answers against a fault that really happened.

    causRCA injected each fault by forcing specific variables to a value, and
    recorded which alarms followed. So we take a healthy recording, force the
    same variables in the model, and compare the alarms we predict against the
    alarms the machine actually raised.

    `forced` holds the value each variable was really forced to, in table
    units. It must come from the dataset's labels per variable: the faults
    force some switches on and others off, so a single shared value would
    replay a different fault from the one that happened.

    This is the step that turns "here is a prediction" into "here is a
    prediction that was checked", and it is possible only because this dataset
    recorded its interventions.
    """
    result = intervene(model, healthy_recording, forced, alarm_names, from_step=from_step)

    predicted = set(result.alarm_changes.loc[result.alarm_changes["fired_after"], "alarm"])
    actual = set(alarms_that_really_fired)

    correct = predicted & actual
    precision = len(correct) / len(predicted) if predicted else 0.0
    recall = len(correct) / len(actual) if actual else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "forced": sorted(forced),
        "forced_to": forced,
        "alarms_predicted": sorted(predicted),
        "alarms_actually_fired": sorted(actual),
        "correctly_predicted": sorted(correct),
        "missed": sorted(actual - predicted),
        "false_alarms": sorted(predicted - actual),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }
