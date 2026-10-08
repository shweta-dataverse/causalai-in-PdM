"""
Root cause analysis for the CNC lathe coolant system.

    streamlit run dashboard/app.py

Reads the results produced by scripts/run_pipeline.py. If those are missing it
says so and tells you what to run.

The dashboard answers four questions, one per page:

    What is this?        what the system learned about the machine
    What went wrong?     the cause of a particular fault
    What should I do?    try a fix and see which alarms stop
    Can I trust it?      how often the system is right
"""

import pickle
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import charts
from causalpdm.pipeline import explain, paths

st.set_page_config(page_title="Coolant system root cause analysis",
                   page_icon="*", layout="wide")


@st.cache_resource(show_spinner="Loading results...")
def load_results():
    saved = paths.CACHE_DIR / "pipeline_result.pkl"
    if not saved.exists():
        return None
    return pickle.loads(saved.read_bytes())


results = load_results()
if results is None:
    st.title("No results yet")
    st.write("Run the pipeline first, then reload this page:")
    st.code("python scripts/run_pipeline.py", language="bash")
    st.stop()

model = results.causal_model
alarms = [name for name in model.variables if "_A_" in name]
describe = {}
seconds_per_step = results.settings_used["seconds_per_step"]

fault_tables = {label.split("/", 1)[1]: table
                for label, table in zip(results.table_labels, results.tables)
                if label.startswith("fault")}
healthy_table = next((t for l, t in zip(results.table_labels, results.tables)
                      if l.startswith("normal")), None)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.title("Coolant system")
    st.caption("Root cause analysis for a CNC vertical lathe")
    page = st.radio("Page", [
        "What the system learned",
        "What went wrong",
        "What should I do",
        "Can I trust it",
    ], label_visibility="collapsed")

    st.divider()
    st.metric("Signals watched", len(model.variables))
    st.metric("Causal links found", len(results.causal_graph.edges))
    st.metric("Faults on record", len(results.root_cause_results))

# --------------------------------------------------------------------------
# Page 1
# --------------------------------------------------------------------------

if page == "What the system learned":
    st.title("What the system learned about the machine")
    st.write(
        "The system was shown recordings of the lathe running normally and "
        "recordings of it breaking down, and worked out on its own which "
        "signals affect which. That map is what everything else is built on."
    )

    graph_score = results.graph_score
    a, b, c = st.columns(3)
    a.metric("Links found", graph_score["edges_found"])
    b.metric("Of those, correct", graph_score["correct"],
             help="Checked against the causal map the machine's engineers drew by hand.")
    c.metric("Links the engineers drew", graph_score["edges_expected"])

    st.plotly_chart(charts.causal_map(results.causal_graph.edges, alarms),
                    width="stretch")
    st.caption("Orange circles are alarms. Arrows point from a cause to what it affects. "
               "Hover a circle to see what it is connected to.")

    st.subheader("The links it found")
    links = results.causal_graph.edges.copy()
    links["acts"] = links["lag"].map(
        lambda l: "straight away" if l == 0
        else f"after {l * seconds_per_step:.0f} seconds")
    st.dataframe(links[["cause", "effect", "acts"]].rename(columns={
        "cause": "when this changes", "effect": "it affects this", "acts": "how quickly"}),
        width="stretch", hide_index=True)

    with st.expander("Which links it got right, and which it missed"):
        comparison = results.edge_comparison.copy()
        comparison["verdict"] = comparison["verdict"].map({
            "found": "found it",
            "missed": "missed it",
            "false alarm": "found a link the engineers did not draw"})
        st.dataframe(comparison, width="stretch", hide_index=True)
        st.caption(
            "Missed links matter more than they look: if a link is missing, the "
            "system cannot work out that fixing one signal would quiet the alarm "
            "on the other end of it.")

# --------------------------------------------------------------------------
# Page 2
# --------------------------------------------------------------------------

elif page == "What went wrong":
    st.title("What went wrong")
    st.write(
        "Pick a recorded fault. The system looks at which signals behaved in a "
        "way nothing else explains, and names the most likely cause."
    )

    by_name = {run.name: (run, result) for run, result in results.root_cause_results}
    chosen = st.selectbox("Which fault?", list(by_name))
    run_info, rca_result = by_name[chosen]
    table = fault_tables.get(chosen)

    suspects = rca_result.ranked_variables[:3]
    top = suspects[0] if suspects else None
    correct = top in run_info.caused_by if top else False

    st.subheader("The system's answer")
    if top:
        if correct:
            st.success(f"**{top}** is the most likely cause of this fault.", icon=":material/check:")
        else:
            st.warning(f"**{top}** is the most likely cause of this fault.")
        st.caption(f"Next most likely: {', '.join(suspects[1:]) or 'nothing else stands out'}")

    with st.expander("What actually caused it (recorded when the fault was created)"):
        st.write(f"**Forced by the test rig:** {', '.join(run_info.caused_by)}")
        st.write(f"**Diagnosis:** {run_info.diagnosis}")
        st.write(f"**Alarms that fired:** {', '.join(rca_result.alarms_that_fired) or 'none'}")
        st.write("Correct" if correct else "The top answer was not one of the real causes.")

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(charts.suspect_ranking(rca_result.ranking, run_info.caused_by),
                        width="stretch")
        st.caption("Red bars are the signals that really were tampered with.")
    with right:
        if table is not None:
            st.plotly_chart(charts.alarm_timeline(
                table, alarms, seconds_per_step, rca_result.fault_window),
                width="stretch")

    st.subheader("Why it picked that signal")
    st.write(
        "Every signal has an expected behaviour, worked out from the signals "
        "that feed into it. A signal that keeps doing something its own causes "
        "cannot account for is one that something external acted on. Alarms, by "
        "contrast, are perfectly well explained by whatever tripped them, which "
        "is how the system tells a cause from a symptom."
    )

# --------------------------------------------------------------------------
# Page 3
# --------------------------------------------------------------------------

elif page == "What should I do":
    st.title("What should I do")
    st.write(
        "Choose a signal and the state you would hold it in. The system replays "
        "the machine with that change in place and shows which alarms stop."
    )

    by_name = {run.name: (run, result) for run, result in results.root_cause_results}
    chosen_fault = st.selectbox("Start from which fault?", list(by_name))
    run_info, rca_result = by_name[chosen_fault]
    table = fault_tables.get(chosen_fault)

    if table is None:
        st.error("That recording is not available.")
        st.stop()

    fixable = [n for n in model.variables if n not in alarms]
    suggested = next((s for s in rca_result.ranked_variables if s in fixable), fixable[0])

    left, right = st.columns(2)
    with left:
        signal = st.selectbox("Which signal would you control?", fixable,
                              index=fixable.index(suggested))
    with right:
        healthy = results.healthy_values.get(signal, 1.0)
        hold_on = st.radio(f"Hold {signal}", ["as it is when healthy", "on", "off"],
                           horizontal=True)

    value = {"as it is when healthy": healthy, "on": 1.0, "off": 0.0}[hold_on]
    from_step = max(rca_result.fault_window[0] - 2, model.max_lag)

    answer = explain.ask_what_if(model, table, {signal: value}, alarms,
                                 from_step=from_step)

    st.subheader("What would happen")
    if answer.prevented_everything:
        st.success(answer.explanation, icon=":material/check:")
    elif answer.prevented_something:
        st.info(answer.explanation)
    else:
        st.warning(answer.explanation)

    a, b = st.columns(2)
    a.metric("Alarms silenced", len(answer.alarms_prevented))
    b.metric("Alarms still firing", len(answer.alarms_still_firing))

    st.plotly_chart(charts.before_and_after(
        answer.actual, answer.what_would_have_happened, alarms, seconds_per_step),
        width="stretch")

    st.caption(
        "This is a replay of this exact recording, not an average. Everything "
        "unusual about this particular run is kept as it was, and only the "
        "signal you chose is changed, so the difference you see is caused by "
        "your change and nothing else."
    )

# --------------------------------------------------------------------------
# Page 4
# --------------------------------------------------------------------------

elif page == "Can I trust it":
    st.title("Can I trust it")
    st.write(
        "Every number here was checked against something recorded in the "
        "dataset, not against the system's own opinion. That is unusual, and it "
        "is the reason this dataset was chosen."
    )

    st.subheader("Finding the causal map")
    st.caption("Compared against the map the machine's engineers drew by hand.")
    graph_score = results.graph_score
    st.plotly_chart(charts.score_bars({
        "of the links we found, correct": graph_score["precision"],
        "of the real links, found": graph_score["recall"],
        "overall (F1)": graph_score["f1"],
    }, "Causal map accuracy"), width="stretch")

    st.subheader("Naming the cause of a fault")
    st.caption("Compared against the signals the test rig really tampered with.")
    rca_score = results.root_cause_score
    st.plotly_chart(charts.score_bars({
        "right cause ranked first": rca_score["found_in_top_1"],
        "right cause in the top 3": rca_score["recall@3"],
        "MAP@3": rca_score["map@3"],
    }, "Root cause accuracy"), width="stretch")

    st.subheader("Predicting what a fault would do")
    st.caption(
        "The strongest check available. Each fault in this dataset was created "
        "by forcing a known signal to a known value. We force the same signal in "
        "the model and compare the alarms it predicts against the alarms the "
        "machine really raised.")
    action = results.intervention_score
    st.plotly_chart(charts.score_bars({
        "alarms predicted that really fired": action["mean_precision"],
        "real alarms we predicted": action["mean_recall"],
        "overall (F1)": action["mean_f1"],
    }, "What-if accuracy"), width="stretch")

    st.subheader("Saying what would have prevented a fault")
    what_if = results.counterfactual_score
    a, b, c = st.columns(3)
    a.metric("Faults examined", what_if["faults_examined"])
    b.metric("Fixing the top suspect helped", what_if["top_suspect_silenced_an_alarm"])
    c.metric("Fixing the real causes worked",
             f"{what_if['known_causes_success_rate']:.0%}")
    st.caption(
        "The last number is the honest test of the machinery itself: if we fix "
        "exactly the signals that were really tampered with, the fault should "
        "disappear. Where it does not, it is because a link is missing from the "
        "causal map, and the fix is to improve the map rather than the what-if "
        "calculation.")

    with st.expander("Settings used for this run"):
        st.json(results.settings_used)
