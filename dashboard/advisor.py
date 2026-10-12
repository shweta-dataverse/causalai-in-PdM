"""
The Fault advisor - the page an engineer actually uses.

Everything else in the dashboard explains how the pipeline works. This page
does not: it takes one alarm event and answers, in the engineer's words,

  1. what to inspect first, and what not to chase,
  2. why the fault happened - and what a fix would have changed, live,
  3. what prevents this type of fault next time,
  4. how far to trust the answer.

It reads the frozen results and the frozen model. The only thing computed here
is the what-if in section 2, which replays the saved model with the fix the
engineer chooses.
"""

from __future__ import annotations

import ast
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

import charts
import explain
from causalpdm.pipeline import data, paths
from causalpdm.pipeline import explain as whatif

DRAWS = 200


@st.cache_data
def _expert_graph(subsystem: str):
    graph = data.load_expert_graph(subsystem)
    variables = graph.variables.set_index("label")
    return graph.edge_pairs, variables["comment"].to_dict(), variables["datatype"].to_dict()


@st.cache_data
def _table(name: str) -> pd.DataFrame | None:
    for folder in (paths.RESULTS_DIR / "final", paths.RESULTS_DIR):
        if (folder / name).exists():
            return pd.read_csv(folder / name)
    return None


@st.cache_data(show_spinner=False)
def _what_if(_model, _recording, run_name: str, fix: tuple, from_step: int,
             _alarms: tuple, _healthy: dict):
    """Cached on the run, the fix and the moment - the model itself never changes."""
    return whatif.prevention_probability(
        _model, _recording, {name: _healthy.get(name, 1.0) for name in fix},
        list(_alarms), from_step=from_step, draws=DRAWS)


def render(results, subsystem: str, fault_tables: dict, by_name: dict,
           seconds_per_step: float) -> None:
    model = results.causal_model
    alarms = tuple(name for name in model.variables if explain.is_alarm(name))
    expert_links, comments, datatypes = _expert_graph(subsystem)

    def describe(signal: str) -> str:
        if signal in explain.SIGNAL_MEANING:
            return explain.SIGNAL_MEANING[signal]
        return comments.get(signal) or signal

    st.title("Fault advisor")
    st.caption("Pick an alarm event. The advisor says what to inspect first, what not to "
               "chase, and what would have prevented it.")

    # --- pick the event ----------------------------------------------------
    def event_label(name: str) -> str:
        _, ranking = by_name[name]
        fired = ", ".join(_short(describe(a)) for a in ranking.alarms_that_fired) or "no alarm"
        return f"{name.replace('/', ' · ')}  -  {fired}"

    # Say it up front when this part of the machine is known to be unreliable,
    # rather than only in section 4 where it could be missed.
    held_out = _table("held_out_scenarios.csv")
    if held_out is not None and subsystem in set(held_out["subsystem"]):
        trust = held_out[held_out["subsystem"] == subsystem].iloc[0]
        if trust["root_cause_top1"] < 0.6 or trust["intervention_f1"] < 0.5:
            st.warning(f"Advice for the {subsystem} is less reliable: on fault types it had "
                       f"not seen before, it named the right cause first only "
                       f"{trust['root_cause_top1']:.0%} of the time. Confirm on the machine "
                       f"before acting.")

    chosen = st.selectbox("Alarm event", list(by_name), format_func=event_label)
    run, ranking = by_name[chosen]
    recording = fault_tables[chosen].reset_index(drop=True)
    table = ranking.ranking
    fired = ranking.alarms_that_fired

    # --- 1. diagnose -------------------------------------------------------
    st.header("1 · What to inspect")
    suspects = table[table["prime_suspect"] & ~table["is_alarm"]]["variable"].tolist()[:3]
    reacted = table[table["changed_in_window"] & ~table["prime_suspect"]
                    & ~table["is_alarm"]]["variable"].tolist()

    left, right = st.columns(2)
    with left:
        st.subheader("Inspect first")
        if not suspects:
            st.info("No signal is linked to these alarms in the learned map. "
                    "Inspect the components behind the alarms directly.")
        for position, signal in enumerate(suspects, start=1):
            confirmed = any((signal, alarm) in expert_links for alarm in fired)
            badge = ("confirmed by the engineers' map" if confirmed
                     else "learned from data only - not in the engineers' map")
            (st.error if position == 1 else st.warning)(
                f"**{position}. {describe(signal)}**  \n`{signal}` · {badge}")
    with right:
        st.subheader("Don't chase")
        st.caption("These went wrong *because* of the fault. Fixing them treats a symptom.")
        for alarm in fired:
            st.markdown(f"- {describe(alarm)}  \n  `{alarm}` · alarm")
        for signal in reacted[:4]:
            st.markdown(f"- {describe(signal)}  \n  `{signal}` · reacted to the fault")

    # --- 2. why it happened, and what a fix would have changed ---------------
    st.header("2 · What would have prevented it")
    estimate = results.prevention_estimates.get(chosen, {}).get("our_fix")
    default_step = estimate.from_step if estimate else model.max_lag
    fixes = _table("minimal_fixes_by_fault.csv")
    minimal = []
    if fixes is not None:
        row = fixes[(fixes["subsystem"] == subsystem) & (fixes["fault"] == chosen)]
        if len(row) and bool(row.iloc[0]["found"]):
            minimal = ast.literal_eval(row.iloc[0]["fix"])
    levers = [v for v in table["variable"] if not explain.is_alarm(v)]

    controls, outcome = st.columns([1, 1])
    with controls:
        fix = st.multiselect(
            "Keep these healthy", levers, default=minimal or suspects,
            format_func=lambda v: f"{describe(v)} ({_state(results.healthy_values.get(v, 1.0), datatypes.get(v))})",
            help="Opens on the smallest fix the advisor found for this event.")
        last_step = len(recording) - 1
        first_alarm = ranking.fault_window[1]
        from_step = st.slider(
            "Apply the fix from second", min_value=model.max_lag * seconds_per_step,
            max_value=last_step * seconds_per_step,
            value=float(min(default_step, last_step)) * seconds_per_step,
            step=seconds_per_step,
            help=f"The first alarm is around second {first_alarm * seconds_per_step:.0f}. "
                 "A fix applied after the alarm cannot un-fire it.")
    step = int(round(from_step / seconds_per_step))
    result = _what_if(model, recording, chosen, tuple(sorted(fix)), step, alarms,
                      results.healthy_values)

    with outcome:
        chance = result.probability_prevented
        st.metric("Chance this fix would have prevented the fault", f"{chance:.0%}",
                  help=f"Over {DRAWS} equally plausible versions of this run's circumstances.")
        if not fix:
            st.info("Choose at least one signal to keep healthy.")
        elif chance >= 0.9:
            st.success("This fix would almost certainly have prevented the fault.")
        elif chance >= 0.5:
            st.warning("This fix would probably have prevented it, but not reliably.")
        else:
            st.error("This fix is not enough. Try another signal, an earlier moment, or "
                     "inspect the component directly.")
        for alarm, stopped in result.alarm_prevented.items():
            st.caption(f"{describe(alarm)}: silenced in {stopped:.0%} of cases")

    seconds = np.arange(len(recording)) * seconds_per_step
    st.plotly_chart(charts.alarm_risk(seconds, result.risk_actual, result.risk_with_fix,
                                      step * seconds_per_step), width="stretch")

    # --- 3. this type of fault, next time -----------------------------------
    st.header("3 · Preventing this type of fault")
    if fixes is not None:
        same = fixes[(fixes["subsystem"] == subsystem)
                     & fixes["fault"].str.startswith(run.experiment + "/")]
        found = same[same["found"]]
        if len(found):
            common = Counter(tuple(ast.literal_eval(f)) for f in found["fix"]).most_common(1)[0]
            st.markdown(
                f"This fault has been recorded **{len(same)} times**. In **{common[1]}** of them "
                f"the smallest fix was the same - keeping these in their normal state "
                f"(average chance of preventing it {found['probability'].mean():.0%}):")
            for signal in common[0]:
                st.markdown(f"- {describe(signal)}  \n  `{signal}`")
            st.caption("A recurring fix like this belongs in the maintenance plan: "
                       "inspect or service these components before they fail.")
        else:
            st.info("No small fix prevents this fault reliably. Plan component inspection "
                    "or replacement instead of an operating change.")
    check = next((c for c in results.intervention_checks
                  if c.get("scenario") == run.experiment), None)
    if check:
        st.caption(f"Validation for this fault type: forcing the same signals in the model "
                   f"predicted the real alarms with accuracy {check['f1']:.2f} (F1).")

    # --- 4. trust ------------------------------------------------------------
    st.header("4 · How far to trust this")
    score = results.root_cause_score
    a, b, c = st.columns(3)
    a.metric("Right cause named first", f"{score['found_in_top_1']:.0%}",
             help=f"Across all {score['faults_scored']} recorded faults of this part of the machine.")
    if held_out is not None and subsystem in set(held_out["subsystem"]):
        row = held_out[held_out["subsystem"] == subsystem].iloc[0]
        b.metric("...on fault types never seen before", f"{row['root_cause_top1']:.0%}")
        c.metric("What-if predictions correct (unseen faults)", f"{row['intervention_f1']:.2f}")
    st.caption("All answers come from a causal model learned from this machine's own "
               "recordings. Interlocks (an alarm switching a pump off) are not learned. "
               "A fault type whose alarm never fired in the history cannot be fully explained.")

    with st.expander("Check against what really happened (test rig)"):
        st.write("The test rig forced: " + ", ".join(f"`{v}`" for v in run.caused_by))
        for signal in suspects:
            st.write(f"{'✓' if signal in run.caused_by else '✗'} {describe(signal)}")


def _state(value: float, datatype: str | None) -> str:
    """How the healthy state reads to an engineer: on/off for switches, normal otherwise."""
    if datatype != "bool":
        return "normal"
    return "on" if value > 0.5 else "off"


def _short(text: str) -> str:
    text = text.replace("ALARM: ", "")
    return text if len(text) <= 48 else text[:45] + "..."
