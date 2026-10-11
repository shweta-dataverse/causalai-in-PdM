"""
Causal root cause analysis for a CNC lathe - interactive dashboard.

    streamlit run dashboard/app.py

One page per stage of the pipeline. Each page explains what that stage does,
why it is needed, and how to read what it produced.

Reads whatever scripts/run_pipeline.py last wrote. If nothing has been run yet
it says so.
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
import explain
from causalpdm.pipeline import explain as whatif
from causalpdm.pipeline import paths

st.set_page_config(page_title="Causal root cause analysis",
                   page_icon="*", layout="wide")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading results...")
def load(subsystem: str):
    name = ("pipeline_result.pkl" if subsystem == "coolant"
            else f"pipeline_result_{subsystem}.pkl")
    saved = paths.CACHE_DIR / name
    if not saved.exists():
        return None
    return pickle.loads(saved.read_bytes())


@st.cache_data
def available_subsystems() -> list[str]:
    found = []
    if (paths.CACHE_DIR / "pipeline_result.pkl").exists():
        found.append("coolant")
    for extra in ("hydraulics", "probe"):
        if (paths.CACHE_DIR / f"pipeline_result_{extra}.pkl").exists():
            found.append(extra)
    return found


@st.cache_data
def tuning_table():
    path = paths.RESULTS_DIR / "tuning_results.csv"
    return pd.read_csv(path) if path.exists() else None


subsystems = available_subsystems()
if not subsystems:
    st.title("Nothing has been analysed yet")
    st.write("Run the pipeline first, then reload this page:")
    st.code("python scripts/run_pipeline.py", language="bash")
    st.stop()


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

PAGES = [
    "Overview",
    "1 - The data",
    "2 - Preparing the data",
    "3 - Finding the causal map",
    "4 - Checking the map",
    "5 - Building a working model",
    "6 - Finding the cause",
    "7 - Predicting an action",
    "8 - What would have prevented it",
    "Settings and tuning",
    "How accurate is it",
]

with st.sidebar:
    st.title("Causal root cause analysis")
    st.caption("CNC vertical lathe")

    subsystem = (st.selectbox("Part of the machine", subsystems)
                 if len(subsystems) > 1 else subsystems[0])
    page = st.radio("Page", PAGES, label_visibility="collapsed")

results = load(subsystem)
model = results.causal_model
alarms = [name for name in model.variables if explain.is_alarm(name)]
seconds_per_step = results.settings_used["seconds_per_step"]

fault_tables = {label.split("/", 1)[1]: table
                for label, table in zip(results.table_labels, results.tables)
                if label.startswith("fault")}
by_name = {run.name: (run, result) for run, result in results.root_cause_results}

with st.sidebar:
    st.divider()
    st.metric("Signals watched", len(model.variables))
    st.metric("Causal links found", len(results.causal_graph.edges))
    st.metric("Faults on record", len(results.root_cause_results))
    st.caption(f"Each row of data covers {seconds_per_step} seconds.")


def stage_header(key: str) -> None:
    """Title, one-line summary, and the full explanation in an expander."""
    stage = explain.STAGES[key]
    st.title(stage["title"])
    st.markdown(f"**{stage['one_line']}**")
    with st.expander("How this works, and why it is done this way", expanded=True):
        st.markdown(stage["body"])
    st.divider()


def pick_fault(label: str = "Which fault?") -> tuple:
    chosen = st.selectbox(label, list(by_name))
    run_info, rca_result = by_name[chosen]
    return chosen, run_info, rca_result


# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------

if page == "Overview":
    st.title("Finding out why a machine broke")
    st.markdown(
        "**Not predicting that it will break. Working out what caused it, what "
        "to do about it, and what would have prevented it.**")

    left, right = st.columns(2)
    with left:
        st.markdown("#### What ordinary predictive maintenance gives you")
        st.error(
            "*\"This machine will probably fail soon.\"*\n\n"
            "An engineer still has to work out which of six flashing alarms is "
            "the actual problem, and which are just consequences of it.")
    with right:
        st.markdown("#### What this system gives you")
        st.success(
            "*\"The high-pressure pump failed. The alarm and the pump "
            "switching off are consequences of that, not separate problems. "
            "Holding the pump healthy would have silenced that alarm.\"*")

    st.divider()
    st.subheader("The eight stages")
    st.caption("Each has its own page, with a full explanation.")

    for row in (list(explain.STAGES.items())[:4], list(explain.STAGES.items())[4:]):
        columns = st.columns(len(row))
        for column, (key, stage) in zip(columns, row):
            with column:
                st.markdown(f"**{stage['title']}**")
                st.caption(stage["one_line"])

    st.divider()
    st.subheader("How it did")
    graph = results.graph_score
    rca = results.root_cause_score
    action = results.intervention_score
    what_if = results.counterfactual_score

    a, b, c, d = st.columns(4)
    a.metric("Causal map accuracy", f"{graph['f1']:.2f}",
             help="F1 against the map the machine's engineers drew by hand.")
    b.metric("Right cause ranked first", f"{rca['found_in_top_1']:.0%}",
             help=f"Across {rca['faults_scored']} recorded faults.")
    c.metric("Fault predictions correct", f"{action['mean_f1']:.2f}",
             help="Alarms we predicted, against alarms that really fired.")
    d.metric("Faults prevented by our fix", f"{what_if['suspects_success_rate']:.0%}",
             help="Fixing the suspects our root cause analysis named made every "
                  "alarm disappear. Upper bound, fixing the known true causes: "
                  f"{what_if['known_causes_success_rate']:.0%}.")

    st.divider()
    st.subheader("How far these results can be pushed")
    for title, body, kind in explain.HONESTY_NOTES:
        (st.success if kind == "good" else st.warning)(f"**{title}** — {body}")


# --------------------------------------------------------------------------
# 1 - The data
# --------------------------------------------------------------------------

elif page == "1 - The data":
    stage_header("data")

    summary = results.dataset_summary
    a, b, c, d = st.columns(4)
    a.metric("Signals", summary["variables"])
    b.metric("Faults recorded", summary["fault_runs"])
    c.metric("Different fault scenarios", summary["fault_scenarios"])
    d.metric("Healthy recordings used", summary["normal_recordings"])

    st.subheader("What each signal means")
    st.caption("Taken from the machine documentation. Alarms are marked.")
    rows = []
    for name in model.variables:
        rows.append({
            "signal": name,
            "what it means": explain.meaning(name),
            "kind": "alarm" if explain.is_alarm(name) else "status signal",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.subheader("The faults, and what really caused them")
    st.caption(
        "This is the dataset's answer key. Each scenario was created by forcing "
        "the listed signals to a faulty value. Everything this system claims is "
        "checked against this.")
    scenarios = {}
    for run, _ in results.root_cause_results:
        scenarios.setdefault(run.experiment, {
            "scenario": run.experiment, "repeats": 0,
            "signals forced": ", ".join(run.caused_by),
            "alarms expected": ", ".join(run.expected_alarms),
            "diagnosis": run.diagnosis})
        scenarios[run.experiment]["repeats"] += 1
    st.dataframe(pd.DataFrame(list(scenarios.values())),
                 width="stretch", hide_index=True)


# --------------------------------------------------------------------------
# 2 - Preparing the data
# --------------------------------------------------------------------------

elif page == "2 - Preparing the data":
    stage_header("preprocess")

    a, b, c = st.columns(3)
    a.metric("Recordings prepared", len(results.tables))
    b.metric("Seconds per row", seconds_per_step)
    c.metric("Rows in total", f"{sum(len(t) for t in results.tables):,}")

    st.subheader("What healthy looks like")
    st.caption(
        "Learned from the healthy recordings, not written in by hand. The system "
        "worked out on its own that every health signal sits on and every alarm "
        "sits off when nothing is wrong. These values fill the gaps when a "
        "signal is missing from a recording.")
    healthy_rows = [{"signal": name,
                     "what it means": explain.meaning(name),
                     "when healthy": "on" if value > 0.5 else "off"}
                    for name, value in results.healthy_values.items()]
    st.dataframe(pd.DataFrame(healthy_rows), width="stretch", hide_index=True)

    st.subheader("A prepared recording")
    st.caption(
        "One row per time step, one column per signal, 1 meaning on. This is "
        "what the causal method actually reads.")
    chosen, run_info, rca_result = pick_fault("Show which fault?")
    table = fault_tables.get(chosen)
    if table is not None:
        st.plotly_chart(charts.alarm_timeline(
            table, alarms, seconds_per_step, rca_result.fault_window,
            title="When each alarm fired during this recording"), width="stretch")
        st.caption(
            "Each bar is one alarm, drawn only while it is ringing. The shaded "
            "band is the window the system examined when looking for the cause - "
            "it reaches back **before** the first alarm, because a cause always "
            "acts before the alarm it triggers.")
        with st.expander("See the underlying numbers"):
            st.dataframe(table.head(40), width="stretch")


# --------------------------------------------------------------------------
# 3 - Finding the causal map
# --------------------------------------------------------------------------

elif page == "3 - Finding the causal map":
    stage_header("discovery")

    settings_used = results.causal_graph.settings
    a, b, c, d = st.columns(4)
    a.metric("Links found", len(results.causal_graph.edges))
    b.metric("Looks back", f"{settings_used['max_lag'] * seconds_per_step:.1f}s")
    c.metric("Evidence needed", settings_used["significance"])
    d.metric("Test used", "G-test" if settings_used["independence_test"] == "discrete"
             else "correlation")

    st.subheader("The map it found")
    st.plotly_chart(charts.causal_map(results.causal_graph.edges, alarms),
                    width="stretch")
    st.caption(
        "**How to read this.** Grey circles are status signals, orange circles "
        "are alarms. An arrow means the signal it starts from causes the one it "
        "points to. Causes are on the left, effects on the right. Hover any "
        "circle to see what it is connected to.")

    st.subheader("Every link, with its delay")
    st.caption(
        "The delay is the system's own finding, read out of the data. Notice "
        "that the filter and fleece alarms come back at five seconds, which is "
        "exactly the five second hold written in the machine's manual - and "
        "nobody told the system about that rule.")
    links = results.causal_graph.edges.copy()
    links["when this changes"] = links["cause"].map(
        lambda n: f"{n} — {explain.meaning(n)}")
    links["it affects"] = links["effect"].map(
        lambda n: f"{n} — {explain.meaning(n)}")
    links["after"] = links["lag"].map(
        lambda l: "straight away" if l == 0 else f"{l * seconds_per_step:.1f} seconds")
    st.dataframe(links[["when this changes", "it affects", "after"]],
                 width="stretch", hide_index=True)


# --------------------------------------------------------------------------
# 4 - Checking the map
# --------------------------------------------------------------------------

elif page == "4 - Checking the map":
    stage_header("validation")

    graph = results.graph_score
    a, b, c, d = st.columns(4)
    a.metric("Links we claimed", graph["edges_found"])
    b.metric("Of those, real", graph["correct"])
    c.metric("Links the engineers drew", graph["edges_expected"])
    d.metric("Accuracy (F1)", f"{graph['f1']:.2f}")

    st.plotly_chart(charts.score_bars({
        "of the links we found, real": graph["precision"],
        "of the real links, found": graph["recall"],
        "the two combined (F1)": graph["f1"],
    }, "How close our map is to the engineers' map"), width="stretch")
    st.caption(
        "**How to read this.** The first bar is about not making things up. The "
        "second is about not missing things. The third balances them, so a "
        "method cannot win by being timid or by being reckless.")

    st.subheader("Link by link")
    comparison = results.edge_comparison.copy()
    comparison["verdict"] = comparison["verdict"].map({
        "found": "found it",
        "missed": "MISSED it",
        "false alarm": "we claimed it, the engineers did not"})
    comparison["link"] = comparison.apply(
        lambda r: f"{r['cause']}  ->  {r['effect']}", axis=1)
    st.dataframe(comparison[["link", "verdict"]], width="stretch", hide_index=True)

    missed = comparison[comparison["verdict"] == "MISSED it"]
    if len(missed):
        st.warning(
            f"**{len(missed)} real links were missed, and this has consequences "
            f"further down.** When a link is missing, the system cannot work out "
            f"that fixing one signal would quiet the alarm on the other end of "
            f"it. On the 'what would have prevented it' page you can see exactly "
            f"this happen: six faults are explained incorrectly, and all six "
            f"trace back to a missing link here.")


# --------------------------------------------------------------------------
# 5 - Building a working model
# --------------------------------------------------------------------------

elif page == "5 - Building a working model":
    stage_header("scm")

    summary = results.causal_model.summary()
    a, b, c = st.columns(3)
    a.metric("Equations fitted", len(results.causal_model.mechanisms))
    b.metric("Average accuracy", f"{summary['accuracy'].mean():.2f}")
    c.metric("Signals with no causes", len(results.causal_model.always_on_or_off))

    st.subheader("How well each signal is explained by its causes")
    st.plotly_chart(charts.mechanism_accuracy(summary), width="stretch")
    st.caption(
        "**How to read this.** A long bar means that signal's behaviour is well "
        "explained by the signals feeding into it. A short bar means most of "
        "what it does comes from outside the model. Short is not bad - a signal "
        "driven from outside is exactly what a root cause looks like.")

    st.subheader("The equations themselves")
    st.caption("Each signal, what drives it, and how well that works.")
    table = summary.copy()
    table["signal"] = table["variable"].map(lambda n: f"{n} — {explain.meaning(n)}")
    st.dataframe(table[["signal", "causes", "accuracy", "share_of_time_on"]].rename(
        columns={"causes": "driven by", "accuracy": "how well that explains it",
                 "share_of_time_on": "share of time on"}),
        width="stretch", hide_index=True)

    st.info(
        "**The guarantee this model has to meet.** Replay a recording through "
        "the model using its own unexplained part, change nothing, and you must "
        "get back exactly what really happened. The test suite checks this on "
        "every run. If it ever failed, every explanation built on the model "
        "would be suspect.")


# --------------------------------------------------------------------------
# 6 - Finding the cause
# --------------------------------------------------------------------------

elif page == "6 - Finding the cause":
    stage_header("rca")

    rca = results.root_cause_score
    a, b, c = st.columns(3)
    a.metric("Faults analysed", rca["faults_scored"])
    b.metric("Right cause ranked first", f"{rca['found_in_top_1']:.0%}")
    c.metric("MAP@3", rca["map@3"])

    st.subheader("Try it on a fault")
    chosen, run_info, rca_result = pick_fault()
    suspects = rca_result.ranked_variables[:3]
    top = suspects[0] if suspects else None
    correct = top in run_info.caused_by if top else False

    if top:
        message = f"**{top}** — {explain.meaning(top)}"
        if correct:
            st.success(f"The system says: {message}\n\nThis is one of the signals "
                       f"that was really tampered with.")
        else:
            st.warning(f"The system says: {message}\n\nThis was not one of the "
                       f"real causes.")

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(charts.suspect_ranking(rca_result.ranking, run_info.caused_by),
                        width="stretch")
        st.caption(
            "**How to read this.** The bar length is how badly that signal's own "
            "causes explain its behaviour during the fault. Red bars are the "
            "signals that really were tampered with. Alarms are excluded, since "
            "an alarm reports a fault and cannot create one.")
    with right:
        st.markdown("**What really happened**")
        st.write(f"Signals forced: {', '.join(run_info.caused_by)}")
        st.write(f"Diagnosis: {run_info.diagnosis}")
        st.write(f"Alarms that fired: {', '.join(rca_result.alarms_that_fired) or 'none'}")
        st.markdown("**The system's three suspects**")
        for position, name in enumerate(suspects, start=1):
            mark = "correct" if name in run_info.caused_by else "not a real cause"
            st.write(f"{position}. {name} — {mark}")


# --------------------------------------------------------------------------
# 7 - Predicting an action
# --------------------------------------------------------------------------

elif page == "7 - Predicting an action":
    stage_header("intervention")

    action = results.intervention_score
    a, b, c = st.columns(3)
    a.metric("Fault scenarios replayed", action["scenarios_checked"],
             help="One check per scenario. Small numbers mean each scenario "
                  "moves the averages a lot.")
    b.metric("Alarms we predicted that really fired", f"{action['mean_precision']:.0%}")
    c.metric("Overall accuracy (F1)", f"{action['mean_f1']:.2f}")

    st.subheader("Each scenario, predicted against what really happened")
    st.caption(
        "**How to read this.** For each scenario we forced the same signals the "
        "test rig forced, then compared the alarms our model raised against the "
        "alarms the real machine raised. This is the strongest evidence on the "
        "dashboard, because it is a prediction checked against reality rather "
        "than against our own assumptions.")
    checks = pd.DataFrame(results.intervention_checks)
    if not checks.empty:
        display = pd.DataFrame({
            "scenario": checks["scenario"],
            "runs": checks["runs"],
            "signals forced": checks["forced_to"].map(
                lambda forced: ", ".join(f"{name} = {value:g}" for name, value in forced.items())),
            "alarms we predicted": checks["alarms_predicted"].map(
                lambda v: ", ".join(v) or "none"),
            "alarms that really fired": checks["alarms_actually_fired"].map(", ".join),
            "we missed": checks["missed"].map(lambda v: ", ".join(v) or "nothing"),
            "we invented": checks["false_alarms"].map(lambda v: ", ".join(v) or "nothing"),
            "accuracy": checks["f1"],
        })
        st.dataframe(display, width="stretch", hide_index=True)

    st.info(
        "**Why 'forcing' is different from 'observing'.** Seeing the pump healthy "
        "tells you the machine is fine. *Forcing* the pump healthy is a "
        "maintenance action, and only a causal model can say what that action "
        "would achieve. Treating the two as the same is the mistake that causal "
        "inference exists to prevent.")


# --------------------------------------------------------------------------
# 8 - What would have prevented it
# --------------------------------------------------------------------------

elif page == "8 - What would have prevented it":
    stage_header("counterfactual")

    what_if = results.counterfactual_score
    a, b, c = st.columns(3)
    a.metric("Faults examined", what_if["faults_examined"])
    b.metric("Fixing our suspects prevented the fault",
             f"{what_if['suspects_success_rate']:.0%}")
    c.metric("Upper bound: fixing the true causes",
             f"{what_if['known_causes_success_rate']:.0%}",
             help="Uses the known answer, which a real plant never has. Shows "
                  "what the model can do when root cause analysis is perfect.")

    st.subheader("Try a fix yourself")
    chosen, run_info, rca_result = pick_fault("Start from which fault?")
    table = fault_tables.get(chosen)

    if table is None:
        st.error("That recording is not available.")
        st.stop()

    fixable = [n for n in model.variables if not explain.is_alarm(n)]
    suggested = next((s for s in rca_result.ranked_variables if s in fixable), fixable[0])
    numeric = [n for n in fixable if results.tables[0][n].nunique() > 2]

    left, middle, right = st.columns([2, 1, 1])
    with left:
        signal = st.selectbox("Which signal would you change?", fixable,
                              index=fixable.index(suggested),
                              format_func=lambda n: f"{n} — {explain.meaning(n)}")
    with middle:
        if signal in numeric:
            series = results.tables[0][signal]
            value = st.slider("Hold it at", float(series.min()), float(series.max()),
                              float(series.median()))
        else:
            healthy = results.healthy_values.get(signal, 1.0)
            state = st.radio("Hold it", ["healthy", "on", "off"], horizontal=False)
            value = {"healthy": healthy, "on": 1.0, "off": 0.0}[state]
    with right:
        earliest = model.max_lag
        latest = max(earliest + 1, rca_result.fault_window[1])
        act_at = st.slider("Act at which step", earliest, latest,
                           max(rca_result.fault_window[0] - 2, earliest),
                           help="Earlier action gives the fix more time to work.")

    answer = whatif.ask_what_if(model, table, {signal: value}, alarms, from_step=act_at)

    if answer.prevented_everything:
        st.success(answer.explanation)
    elif answer.prevented_something:
        st.info(answer.explanation)
    else:
        st.warning(answer.explanation)

    a, b = st.columns(2)
    a.metric("Alarms silenced", len(answer.alarms_prevented))
    b.metric("Alarms still ringing", len(answer.alarms_still_firing))

    st.plotly_chart(charts.before_and_after(
        answer.actual, answer.what_would_have_happened, alarms, seconds_per_step),
        width="stretch")
    st.caption(
        "**How to read this.** Orange is what really happened. Green is what "
        "would have happened with your change. Both lines are the same machine "
        "with the same unexplained quirks - only the signal you chose differs, "
        "so any difference you see is caused by your change and nothing else.")

    st.info(
        "**Why fixing one thing often is not enough here.** Most faults in this "
        "dataset were created by forcing two or three signals at once. Fixing "
        "one silences its own alarm and leaves the others ringing. That is the "
        "correct answer, not a failure of the method.")


# --------------------------------------------------------------------------
# Settings and tuning
# --------------------------------------------------------------------------

elif page == "Settings and tuning":
    st.title("Settings, and how they were chosen")
    st.markdown(
        "**Nothing here was guessed.** Every value was picked by trying it and "
        "scoring the result against the engineers' own causal map.")

    st.info(
        "Choosing settings for causal discovery is normally awkward, because "
        "there is no correct answer to compare against - finding the answer is "
        "the whole task. This dataset comes with one, so ordinary model "
        "selection becomes possible. That is a direct benefit of the dataset "
        "choice, not of the method.")

    st.subheader("What each setting does")
    current = results.settings_used
    for item in explain.SETTINGS_GUIDE:
        value = current.get(item["name"].lower(), current.get(
            item["name"].lower().replace("resample_seconds", "seconds_per_step"), ""))
        with st.expander(f"**{item['plain']}**  ({item['name']})"):
            st.markdown(item["detail"])
            st.markdown(f"**Sensible range:** {item['range']}")
            st.markdown(f"**What changing it does:** {item['effect']}")

    st.divider()
    st.subheader("The search")
    table = tuning_table()
    if table is None:
        st.caption("No tuning results saved yet. Run `python scripts/tune_settings.py`.")
    else:
        finished = table[table["finished"]]
        a, b, c = st.columns(3)
        a.metric("Combinations tried", len(table))
        b.metric("Finished in time", len(finished))
        c.metric("Best accuracy found", f"{finished['f1'].max():.2f}")

        st.plotly_chart(charts.tuning_scatter(finished), width="stretch")
        st.caption(
            "**How to read this.** Each dot is one combination of settings. "
            "Further right is more accurate; further down is faster. The best "
            "place to be is the bottom right. Note that the most accurate "
            "settings are also among the fastest - the slow combinations are "
            "slow because they let in extra noise, which is also why they score "
            "worse.")

        st.dataframe(table.rename(columns={
            "seconds_per_step": "seconds per row", "max_lag": "steps back",
            "significance": "evidence needed", "seconds_to_run": "seconds to run",
            "edges_found": "links found", "correct": "of those, real",
            "f1": "accuracy"}), width="stretch", hide_index=True)

    st.divider()
    st.subheader("Settings used for the results on this dashboard")
    st.json(current)


# --------------------------------------------------------------------------
# How accurate is it
# --------------------------------------------------------------------------

elif page == "How accurate is it":
    st.title("How accurate is it")
    st.markdown(
        "**Every number here was checked against something recorded in the "
        "dataset**, not against the system's own opinion of itself.")

    graph = results.graph_score
    st.subheader("1. Finding the causal map")
    st.caption("Checked against the map the machine's engineers drew by hand.")
    st.plotly_chart(charts.score_bars({
        "of the links we found, real": graph["precision"],
        "of the real links, found": graph["recall"],
        "combined (F1)": graph["f1"],
    }, "Causal map"), width="stretch")

    rca = results.root_cause_score
    st.subheader("2. Naming the cause of a fault")
    st.caption("Checked against the signals the test rig really tampered with.")
    st.plotly_chart(charts.score_bars({
        "right cause ranked first": rca["found_in_top_1"],
        "right cause somewhere in top 3": rca["recall@3"],
        "MAP@3": rca["map@3"],
    }, "Root cause"), width="stretch")

    action = results.intervention_score
    st.subheader("3. Predicting what a fault would do")
    st.caption("Checked against faults that were really injected into the machine.")
    st.plotly_chart(charts.score_bars({
        "alarms we predicted that really fired": action["mean_precision"],
        "real alarms we predicted": action["mean_recall"],
        "combined (F1)": action["mean_f1"],
    }, "Fault prediction"), width="stretch")

    what_if = results.counterfactual_score
    st.subheader("4. Explaining what would have prevented a fault")
    a, b, c = st.columns(3)
    a.metric("Faults examined", what_if["faults_examined"])
    b.metric("Fixing our suspects prevented the fault",
             f"{what_if['suspects_success_rate']:.0%}")
    c.metric("Upper bound: fixing the true causes",
             f"{what_if['known_causes_success_rate']:.0%}")
    st.caption(
        "The middle number is what the pipeline achieves on its own. The last "
        "one uses the known answer, so it is an upper bound and a test of the "
        "reasoning itself: if fixing exactly the tampered signals does not make "
        "the fault disappear, a link is missing from the map - so the fix is to "
        "improve the map, not the reasoning.")

    st.divider()
    for title, body, kind in explain.HONESTY_NOTES:
        (st.success if kind == "good" else st.warning)(f"**{title}** — {body}")
