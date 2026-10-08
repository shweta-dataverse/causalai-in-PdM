"""
Actionable Root Cause Analysis — interactive decision support dashboard.

Run it with:

    streamlit run dashboard/app.py

It loads the saved pipeline results from artifacts/framework/, so it starts
instantly. The intervention and counterfactual pages re-simulate the causal
model live as you move the sliders.

If artifacts/framework/ is empty, run the pipeline first:

    python scripts/run_pipeline.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make our own code importable, whether streamlit is run from the repo root
# or from inside the dashboard folder.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import charts
import explain
from explain import label, short_label

st.set_page_config(
    page_title="Actionable RCA — Causal AI for Predictive Maintenance",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Loading the saved pipeline results
# =============================================================================

@st.cache_resource(show_spinner="Loading pipeline results…")
def load_everything():
    """Load every saved stage. Cached, so this happens once per session."""
    import pipeline_config as cfg
    from causalpdm.framework import artifacts, scm

    needed = ["prep", "feat", "disc", "rca", "interv", "cf"]
    missing = [n for n in needed if not artifacts.exists(n)]
    if missing:
        return {"error": missing}

    data = {n: artifacts.load(n) for n in needed}
    data["valid"] = artifacts.load("valid") if artifacts.exists("valid") else None
    data["edges"] = (artifacts.load("edges_final") if artifacts.exists("edges_final")
                     else data["disc"]["edges"])
    data["cfg"] = cfg

    variables = ([v for v in cfg.CAUSAL_VARIABLES if v in data["prep"]["feature_cols"]]
                 if cfg.CAUSAL_VARIABLES else data["prep"]["feature_cols"])
    data["variables"] = variables

    # Refit the SCM here rather than loading it: it takes a fraction of a second
    # and this way it always matches the edge set currently on screen.
    data["scm_result"] = scm.run(
        data["prep"]["normalised"], variables, data["edges"], tau_max=cfg.MAX_LAG
    )
    return data


@st.cache_data(show_spinner="Recovering this engine's noise (abduction)…")
def get_noise(_model, engine_key: int, engine: pd.DataFrame):
    """
    Abduction step, cached because it is reused by every slider movement.

    The leading underscore on _model tells Streamlit not to try to hash it.
    """
    from causalpdm.framework import scm as scm_module
    return scm_module.recover_noise(_model, engine)


def simulate_health(_model, engine, readout, variable, value, start_idx, noise=None):
    """Force `variable` to `value` from `start_idx`, then read off the health score."""
    from causalpdm.framework import scm as scm_module
    simulated = scm_module.simulate(
        _model, engine, noise=noise,
        interventions={variable: float(value)}, intervention_start=int(start_idx),
    )
    return readout.transform(simulated), simulated


@st.cache_data(show_spinner="Sweeping the value range…")
def sweep_values(_model, engine_key: int, engine: pd.DataFrame, _readout,
                 variable: str, start_idx: int, lo: float, hi: float,
                 n_points: int = 11, use_noise: bool = False, _noise=None):
    """Health outcome across a range of forced values, for the dose-response curve."""
    grid = np.linspace(lo, hi, n_points)
    out = []
    for value in grid:
        health, _ = simulate_health(_model, engine, _readout, variable, value,
                                    start_idx, noise=_noise if use_noise else None)
        out.append(float(health.iloc[int(start_idx):].max()))
    return grid, out


DATA = load_everything()

if isinstance(DATA, dict) and "error" in DATA:
    st.error("### The pipeline has not been run yet")
    st.write(f"Missing saved results: `{'`, `'.join(DATA['error'])}`")
    st.code("python scripts/run_pipeline.py", language="bash")
    st.stop()

cfg = DATA["cfg"]
prep, feat, disc, valid = DATA["prep"], DATA["feat"], DATA["disc"], DATA["valid"]
rca_result, interv, cf = DATA["rca"], DATA["interv"], DATA["cf"]
edges, variables = DATA["edges"], DATA["variables"]
scm_result = DATA["scm_result"]
model = scm_result["scm"]
readout = feat["health_readout"]
threshold = feat["failure_threshold"]

engine = rca_result["engine"]
unit = rca_result["unit"]
onset_idx = rca_result["onset_idx"]

# Who is a cause, who is a symptom - used to colour every graph consistently.
ROLES = {v: "cause" for v in rca_result["root_causes"]["variable"]}
ROLES.update({v: "symptom" for v in rca_result["symptoms"]["variable"]})


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.title("🔧 Actionable RCA")
    st.caption("Causal Discovery + Counterfactual XAI for Predictive Maintenance")

    page = st.radio(
        "Go to",
        [
            "Overview",
            "1–2 · Data & Health",
            "3 · Causal Discovery",
            "4 · Graph Validation",
            "5 · Structural Causal Model",
            "6 · Root Cause Analysis",
            "7 · Interventions  🎚️",
            "8 · Counterfactual XAI  🎚️",
            "Hyperparameters",
            "Recommendations",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.metric("Engine under analysis", f"#{unit}")
    st.metric("Degraded at cycle", f"{rca_result['onset_cycle']} / {rca_result['total_cycles']}")
    st.metric("Causal links kept", len(edges))
    st.caption(f"Dataset: C-MAPSS {cfg.DATASET} · {disc['n_realisations']} engines pooled "
               f"for discovery")

    st.divider()
    st.caption("🎚️ = interactive page with sliders")


# =============================================================================
# Overview
# =============================================================================

if page == "Overview":
    st.title("Toward Actionable Root Cause Analysis")
    st.markdown(
        "#### Integrating Causal Discovery and Counterfactual XAI for Predictive Maintenance"
    )
    st.caption(f"Prototype on NASA C-MAPSS {cfg.DATASET}. Every number on this dashboard is "
               f"computed by the pipeline in `src/causalpdm/framework/`.")

    st.divider()

    st.subheader("The problem this framework solves")
    left, right = st.columns(2)
    with left:
        st.markdown("""
**Standard predictive maintenance tells you *when* a machine will fail.**

It gives you a number — 30 cycles left — and stops there. The engineer still has to guess what is
actually wrong, and which action would help.
        """)
        st.error("**Predict only:** *\"This engine will fail in about 30 cycles.\"*", icon="⚠️")
    with right:
        st.markdown("""
**This framework tells you *why*, *what to do*, and *what would have prevented it*.**

That requires causality. Correlation cannot distinguish a cause from a symptom, because
correlation has no direction.
        """)
        st.success(
            "**Explain and act:** *\"HPC outlet pressure is the root cause; the temperature rise "
            "is a downstream symptom. Holding pressure in range from cycle 99 would have kept "
            "this engine below the failure threshold.\"*", icon="✅")

    st.divider()
    st.subheader("The nine-module pipeline")

    stages = [
        ("1", "Preprocessing", f"{prep['n_rows']:,} rows", "ok"),
        ("2", "Features & health", f"{feat['health_indicator_meta']['explained_variance_ratio']:.0%} variance", "ok"),
        ("3", "Causal discovery", f"{disc['n_directed_edges']} links found", "warn"),
        ("4", "Validation", f"{len(edges)} links kept" if valid else "not run", "warn"),
        ("5", "SCM fitting", f"R² {scm_result['mean_r2']:.2f}", "ok"),
        ("6", "Root cause", f"{len(rca_result['root_causes'])} causes", "ok"),
        ("7", "Interventions", f"do({interv['best_variable']})", "est"),
        ("8", "Counterfactual", "found" if cf["found"] else "none", "est"),
        ("9", "Dashboard", "this page", "ok"),
    ]
    cols = st.columns(len(stages))
    for col, (num, name, value, kind) in zip(cols, stages):
        with col:
            mark = {"ok": "🟢", "warn": "🟡", "est": "🟠"}[kind]
            st.markdown(f"**{mark} {num}. {name}**")
            st.caption(value)

    st.caption("🟢 computed from data · 🟡 robustness-checked but no ground truth available · "
               "🟠 model-based estimate that C-MAPSS cannot validate")

    st.divider()
    st.subheader("How much of this can be trusted?")
    st.markdown(
        "A decision-support tool that hides which of its numbers are validated is a demo, not "
        "decision support. So this is stated explicitly:"
    )
    for title, body, kind in explain.SCOPE_NOTES:
        (st.success if kind == "ok" else st.warning)(f"**{title}** — {body}")


# =============================================================================
# Modules 1 & 2
# =============================================================================

elif page == "1–2 · Data & Health":
    st.title("Modules 1 & 2 · Data Preprocessing and Health Score")

    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{prep['n_rows']:,}")
    b.metric("Engines", prep["n_units"])
    c.metric("Missing values", f"{prep['missing_share']:.1%}")
    d.metric("Sensors used", len(variables))

    tab1, tab2 = st.tabs(["📖 What this step does", "📊 Results"])

    with tab1:
        st.markdown(explain.MODULES["preprocessing"]["long"])
        st.divider()
        st.markdown(explain.MODULES["features"]["long"])

    with tab2:
        st.subheader("The two versions of the data")
        st.caption("Same engine, same sensors. Left keeps the degradation trend; right removes it. "
                   "Causal discovery uses the right-hand version; everything else uses the left.")

        sample = prep["normalised"]["unit"].iloc[0]
        n_df = prep["normalised"][prep["normalised"]["unit"] == sample]
        d_df = prep["detrended"][prep["detrended"]["unit"] == sample]
        show = [v for v in ["s_2", "s_4", "s_11"] if v in n_df.columns]

        import plotly.graph_objects as go
        c1, c2 = st.columns(2)
        for col, frame, name in [(c1, n_df, "trend intact → used for health score & simulation"),
                                 (c2, d_df, "trend removed → used only for causal discovery")]:
            fig = go.Figure()
            for s in show:
                fig.add_trace(go.Scatter(x=frame["cycle"], y=frame[s], mode="lines",
                                         name=short_label(s)))
            fig.update_layout(title=name, xaxis_title="cycle", height=330,
                              margin=dict(l=10, r=10, t=45, b=10),
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              legend=dict(orientation="h", y=1.02))
            col.plotly_chart(fig, width="stretch")

        st.divider()
        st.subheader("The health score, and the failure event it defines")
        e1, e2 = st.columns([2, 1])
        with e1:
            st.plotly_chart(
                charts.health_trajectory(engine, threshold, rca_result["onset_cycle"], unit,
                                         fleet=feat["featured"]),
                width="stretch")
        with e2:
            st.metric("Variance captured", f"{feat['health_indicator_meta']['explained_variance_ratio']:.0%}")
            st.metric("Failure threshold", f"{threshold:.3f}")
            st.metric("Engines reaching it", f"{feat['n_engines_with_onset']} / {prep['n_units']}")
            onsets = feat["featured"][feat["featured"]["onset"] == 1].groupby("unit")["cycle"].first()
            lifetimes = feat["featured"].groupby("unit")["cycle"].max()
            warning = (lifetimes - onsets).dropna()
            st.metric("Median warning time", f"{warning.median():.0f} cycles")
            st.caption("How long before failure the health score first crosses the threshold — "
                       "this is the window in which maintenance could act.")

        st.info("**Kept out of the causal graph on purpose.** The health score combines many "
                "sensors, so it is a deterministic function of other variables. Putting it in the "
                "graph would break the faithfulness assumption that causal discovery relies on. "
                "Instead it is a *readout*: it reports on the system without being part of it.",
                icon="🧠")


# =============================================================================
# Module 3
# =============================================================================

elif page == "3 · Causal Discovery":
    st.title("Module 3 · Causal Discovery with PCMCI+")

    a, b, c, d = st.columns(4)
    a.metric("Links discovered", disc["n_directed_edges"])
    b.metric("Engines pooled", disc["n_realisations"])
    c.metric("Max lag", f"{disc['tau_max']} cycles")
    d.metric("Significance", disc["pc_alpha"])

    tab1, tab2, tab3 = st.tabs(["📖 What PCMCI+ is", "🕸️ The causal graph", "📋 Every link"])

    with tab1:
        st.markdown(explain.MODULES["discovery"]["long"])

    with tab2:
        st.caption("Hover any node to see what causes it and what it affects. Arrow direction is "
                   "the causal direction; thicker lines are stronger links. Drag to pan, scroll "
                   "to zoom.")
        colour_mode = st.radio("Colour the nodes by",
                               ["Role in this failure", "Part of the engine"],
                               horizontal=True)
        st.plotly_chart(
            charts.causal_graph(
                edges, ROLES,
                title=f"Causal graph — {len(edges)} links between {len(variables)} sensors",
                colour_by_section=(colour_mode == "Part of the engine"),
            ),
            width="stretch")

        if colour_mode == "Role in this failure":
            st.caption("🔴 root cause · 🟠 downstream symptom · ⚪ not implicated in this failure")
        else:
            legend = " · ".join(f"{sec}" for sec in charts.SECTION_COLOURS)
            st.caption(f"Coloured by engine section: {legend}")

    with tab3:
        st.subheader("All discovered links, strongest first")
        show = disc["edges"].copy()
        show["from"] = show["source"].map(label)
        show["to"] = show["target"].map(label)
        show["acts after"] = show["lag"].map(lambda l: "same cycle" if l == 0 else f"{l} cycle(s)")
        st.dataframe(
            show[["from", "to", "acts after", "strength"]].round(3),
            width="stretch", hide_index=True)

        st.divider()
        st.subheader("What PCMCI+ reported, including links it could not orient")
        st.caption("Not every dependency can be given a direction from data alone. Links marked "
                   "`o-o` are dependencies of unknown direction — they are **not** counted as "
                   "causal edges here, because treating them as directed would overstate the "
                   "finding.")
        st.dataframe(disc["link_types"].rename("count").to_frame(),
                     width="stretch")


# =============================================================================
# Module 4
# =============================================================================

elif page == "4 · Graph Validation":
    st.title("Module 4 · Graph Validation")

    if valid is None:
        st.warning("Validation was not run. Set `RUN_VALIDATION = True` in `pipeline_config.py` "
                   "and re-run the pipeline.", icon="⚠️")
        st.markdown(explain.MODULES["validation"]["long"])
    else:
        a, b, c = st.columns(3)
        a.metric("Links kept", f"{valid['n_validated']} of {len(disc['edges'])}")
        b.metric("Bootstrap runs", cfg.BOOTSTRAP_RUNS)
        dom = valid["domain"]
        if dom.get("n_checked"):
            c.metric("Physically plausible", f"{dom['share_downstream_consistent']:.0%}")

        tab1, tab2, tab3, tab4 = st.tabs([
            "📖 What validation means here", "🔁 Stability", "🎛️ Sensitivity",
            "🔬 Cross-check & physics"])

        with tab1:
            st.markdown(explain.MODULES["validation"]["long"])

        with tab2:
            st.caption("Discovery was re-run on random subsets of engines. An edge that keeps "
                       "reappearing is a real finding; one that appears rarely is probably noise.")
            st.plotly_chart(
                charts.stability_bars(valid["stability"], cfg.STABILITY_THRESHOLD),
                width="stretch")

        with tab3:
            st.caption("If the graph changed completely every time a setting moved, it would not "
                       "be a result. Jaccard similarity compares each setting's edge set against "
                       "the chosen one — 1.0 means identical.")
            st.dataframe(valid["sensitivity"].round(3), width="stretch", hide_index=True)

        with tab4:
            st.subheader("Does an independent method agree?")
            st.caption("Lagged linear regression is score-based — a completely different family "
                       "from PCMCI+'s constraint-based approach. Agreement on the strongest links "
                       "is evidence they are not artefacts of one particular test.")
            cross = valid["cross_method"].copy()
            cross["from"] = cross["source"].map(short_label)
            cross["to"] = cross["target"].map(short_label)
            st.dataframe(cross[["from", "to", "lag", "coef"]].round(3).head(12),
                         width="stretch", hide_index=True)

            st.divider()
            st.subheader("Do the links respect the engine's physics?")
            st.markdown("""
Air flows through a turbofan in a fixed order:

`fan → low-pressure compressor → high-pressure compressor → combustor → high-pressure turbine → low-pressure turbine`

A causal link following that flow is physically expected. One running against it needs a
feedback-control explanation. A high share of flow-consistent links is evidence the graph is
picking up real physics rather than statistical noise.
            """)
            if dom.get("n_checked"):
                st.metric("Links consistent with air flow",
                          f"{dom['n_downstream_consistent']} of {dom['n_checked']}",
                          f"{dom['share_downstream_consistent']:.0%}")


# =============================================================================
# Module 5
# =============================================================================

elif page == "5 · Structural Causal Model":
    st.title("Module 5 · Structural Causal Model")
    st.caption("The graph says *which* sensors matter. The SCM says *how much* — and that is what "
               "makes the next two pages possible.")

    a, b, c = st.columns(3)
    a.metric("Equations fitted", scm_result["n_mechanisms"])
    b.metric("Average fit (R²)", f"{scm_result['mean_r2']:.2f}")
    c.metric("Exogenous sensors", scm_result["n_exogenous"])

    tab1, tab2, tab3 = st.tabs(["📖 Why this step exists", "🧮 The equations", "📊 Fit quality"])

    with tab1:
        st.markdown(explain.MODULES["scm"]["long"])

    with tab2:
        st.subheader("Every fitted equation, in readable form")
        st.caption("Read these as: *this sensor now* = weighted sum of *its causes* + *its own "
                   "unexplained noise*. `(t-1)` means one cycle ago.")

        for name, mech in model.mechanisms.items():
            terms = []
            for (parent, lag), coef in zip(mech.parents, mech.coefficients):
                when = "t" if lag == 0 else f"t−{lag}"
                sign = "+" if coef >= 0 else "−"
                terms.append(f"{sign} {abs(coef):.3f}·{parent}({when})")
            equation = " ".join(terms).lstrip("+ ").strip()

            with st.expander(f"**{label(name)}**  ·  R² = {mech.r2:.2f}"):
                st.code(f"{name}(t) = {equation}\n"
                        f"{' ' * (len(name) + 7)}+ noise (σ = {mech.noise_std:.3f})",
                        language="text")
                st.caption(
                    f"Driven by: {', '.join(short_label(p) for p, _ in mech.parents)}. "
                    f"An R² of {mech.r2:.2f} means "
                    f"{mech.r2:.0%} of this sensor's behaviour is explained by its causes, and "
                    f"{1 - mech.r2:.0%} is its own noise."
                )

    with tab3:
        st.plotly_chart(charts.mechanism_fit(scm_result["summary"]), width="stretch")
        st.info("**A low R² is not a failure.** It means most of that sensor's behaviour comes "
                "from outside the model — which makes it a poor *prediction target* but often an "
                "excellent *root cause*, because it acts on the system rather than merely "
                "reacting to it.", icon="💡")
        st.divider()
        st.subheader("Simulation order")
        st.caption("Same-cycle links must form an acyclic order, otherwise the model could not be "
                   "simulated forward at all. This is that order:")
        st.code(" → ".join(model.topological_order), language="text")


# =============================================================================
# Module 6
# =============================================================================

elif page == "6 · Root Cause Analysis":
    st.title("Module 6 · Root Cause Analysis")
    st.caption(f"Engine #{unit}, which crossed the failure threshold at cycle "
               f"{rca_result['onset_cycle']} of {rca_result['total_cycles']}.")

    a, b, c = st.columns(3)
    a.metric("Root causes identified", len(rca_result["root_causes"]))
    b.metric("Downstream symptoms", len(rca_result["symptoms"]))
    c.metric("Drift threshold", f"{rca_result['deviation_threshold']} sd")

    tab1, tab2, tab3 = st.tabs(["📖 Cause vs symptom", "📊 The verdict", "🔗 Causal chains"])

    with tab1:
        st.markdown(explain.MODULES["rca"]["long"])

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(charts.deviation_bars(rca_result["deviations"], ROLES),
                            width="stretch")
            st.caption("🔴 root cause · 🟠 symptom · ⚪ not implicated")
        with c2:
            st.plotly_chart(charts.root_cause_scatter(rca_result["ranking"]),
                            width="stretch")
            st.caption("Top-right is the dangerous corner: strongly abnormal **and** explains "
                       "much of what else went wrong.")

        st.divider()
        st.subheader("Full ranking")
        show = rca_result["ranking"].copy()
        show["sensor"] = show["variable"].map(label)
        show["verdict"] = show["is_root_cause"].map({True: "🔴 ROOT CAUSE", False: "🟠 symptom"})
        st.dataframe(
            show[["sensor", "verdict", "abs_deviation", "causal_reach",
                  "n_symptoms_explained", "root_cause_score"]].round(3),
            width="stretch", hide_index=True,
            column_config={
                "abs_deviation": st.column_config.NumberColumn("drift (sd)"),
                "causal_reach": st.column_config.NumberColumn("causal reach"),
                "n_symptoms_explained": st.column_config.NumberColumn("symptoms explained"),
                "root_cause_score": st.column_config.ProgressColumn(
                    "score", min_value=0.0,
                    max_value=float(show["root_cause_score"].max() or 1.0)),
            })

    with tab3:
        st.subheader("The actual chains behind each verdict")
        st.caption("This is what makes the result an explanation rather than a score: each root "
                   "cause comes with the causal paths through which it produces the symptoms.")
        if rca_result["root_causes"].empty:
            st.info("No root causes were identified above the drift threshold.")
        for _, row in rca_result["root_causes"].iterrows():
            with st.expander(f"🔴 **{label(row['variable'])}** — drift "
                             f"{row['abs_deviation']:.2f} sd, explains "
                             f"{int(row['n_symptoms_explained'])} symptom(s)", expanded=True):
                if row["explained_symptoms"]:
                    for path in row["explained_symptoms"]:
                        pretty = "  →  ".join(short_label(p.strip())
                                              for p in path.split("->"))
                        st.markdown(f"- `{path}`  \n  {pretty}")
                else:
                    st.caption("Deviates strongly but has no traced path to other deviating "
                               "sensors in the validated graph.")


# =============================================================================
# Module 7 — INTERACTIVE
# =============================================================================

elif page == "7 · Interventions  🎚️":
    st.title("Module 7 · Intervention Analysis")
    st.caption("Pick a sensor, choose a value to hold it at, and the causal model is re-simulated "
               "live to show what would happen to engine health.")

    with st.expander("📖 What an intervention is, and what `do()` means"):
        st.markdown(explain.MODULES["intervention"]["long"])

    st.divider()

    # --- controls ---------------------------------------------------------
    candidates = rca_result["ranking"]["variable"].tolist() or variables
    candidates = [v for v in candidates if v in model.var_names]
    default_var = (interv["best_variable"] if interv["best_variable"] in candidates
                   else candidates[0])

    c1, c2 = st.columns([1, 1])
    with c1:
        chosen = st.selectbox(
            "Which sensor would you control?",
            candidates, index=candidates.index(default_var),
            format_func=lambda v: (f"{label(v)}"
                                   + ("  🔴 root cause" if ROLES.get(v) == "cause"
                                      else "  🟠 symptom" if ROLES.get(v) == "symptom" else "")),
        )
    with c2:
        lead = st.slider(
            "How many cycles before failure does maintenance act?",
            min_value=5, max_value=min(60, max(10, onset_idx - model.tau_max)),
            value=int(cfg.MAINTENANCE_LEAD_TIME), step=5,
            help="Acting earlier gives the intervention more time to work, but assumes the "
                 "problem was spotted sooner.",
        )

    start_idx = max(model.tau_max, onset_idx - lead)

    observed = engine[chosen].to_numpy(dtype=float)
    lo, hi = float(np.percentile(observed, 2)), float(np.percentile(observed, 98))
    actual_level = float(observed[start_idx:].mean())
    healthy_level = float(observed[: max(10, int(0.3 * onset_idx))].mean())

    hold_at = st.slider(
        f"🎚️  Hold **{short_label(chosen)}** at this value from cycle "
        f"{int(engine.loc[start_idx, 'cycle'])} onward",
        min_value=round(lo, 2), max_value=round(hi, 2),
        value=round(healthy_level, 2), step=round(max((hi - lo) / 100, 0.01), 3),
        help=f"Its actual average over this period was {actual_level:.2f}. "
             f"Its healthy early-life average was {healthy_level:.2f}.",
    )

    # --- simulate ---------------------------------------------------------
    health_modified, _ = simulate_health(model, engine, readout, chosen, hold_at, start_idx)
    health_factual = readout.transform(engine[model.var_names])

    tail = slice(start_idx, len(engine))
    peak_factual = float(health_factual.iloc[tail].max())
    peak_modified = float(health_modified.iloc[tail].max())
    avoided = peak_modified < threshold

    # --- results ----------------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Peak health loss — actual", f"{peak_factual:.3f}")
    m2.metric("Peak health loss — with your action", f"{peak_modified:.3f}",
              delta=f"{peak_modified - peak_factual:+.3f}", delta_color="inverse")
    m3.metric("Failure threshold", f"{threshold:.3f}")
    m4.metric("Failure avoided?", "✅ Yes" if avoided else "❌ No")

    if avoided:
        st.success(f"**Holding {short_label(chosen)} at {hold_at:.2f} keeps this engine below the "
                   f"failure threshold.** Peak health loss falls from {peak_factual:.3f} to "
                   f"{peak_modified:.3f}.", icon="✅")
    else:
        st.warning(f"**Holding {short_label(chosen)} at {hold_at:.2f} is not enough.** Peak "
                   f"health loss is {peak_modified:.3f}, still above the {threshold:.3f} "
                   f"threshold. Try another value, an earlier action, or a different sensor.",
                   icon="⚠️")

    st.plotly_chart(
        charts.before_after_health(
            engine["cycle"].to_numpy(), health_factual.to_numpy(), health_modified.to_numpy(),
            threshold, start_idx,
            label_modified=f"if {short_label(chosen)} held at {hold_at:.2f}",
            title=f"Engine #{unit} — effect of controlling {short_label(chosen)}",
        ),
        width="stretch")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Every value, swept")
        st.caption("Rather than guessing one value at a time, this sweeps the whole range.")
        grid, peaks = sweep_values(model, int(unit), engine, readout, chosen,
                                   int(start_idx), lo, hi, 11, False, None)
        st.plotly_chart(
            charts.dose_response(grid, peaks, actual_level, threshold, short_label(chosen)),
            width="stretch")
        st.caption("A sloping curve supports a *'keep this sensor below X'* rule. A flat curve "
                   "means this sensor is not a useful lever at any value — which is itself "
                   "actionable: replace the component instead of adjusting operations.")
    with c2:
        st.subheader("Which lever helps most")
        st.caption("Each root-cause candidate clamped to its own healthy range, for comparison.")
        st.plotly_chart(charts.intervention_ranking(interv["ranking"]),
                        width="stretch")
        show = interv["ranking"].copy()
        show["sensor"] = show["variable"].map(short_label)
        st.dataframe(show[["sensor", "do_value", "health_improvement"]].round(4),
                     width="stretch", hide_index=True)

    st.info("**Scope.** These are model-based estimates. C-MAPSS contains no interventional data, "
            "so none of them can be checked against ground truth. The same code run on causRCA — "
            "which has 100 labelled real interventions — can be validated.", icon="🟠")


# =============================================================================
# Module 8 — INTERACTIVE
# =============================================================================

elif page == "8 · Counterfactual XAI  🎚️":
    st.title("Module 8 · Counterfactual XAI")
    st.caption("The retrospective question: *would this specific engine have failed, if one thing "
               "had been different?*")

    with st.expander("📖 Pearl's three steps: abduction → action → prediction"):
        st.markdown(explain.MODULES["counterfactual"]["long"])

    st.divider()

    candidates = [v for v in (rca_result["ranking"]["variable"].tolist() or variables)
                  if v in model.var_names]
    default_var = cf.get("variable") if cf.get("variable") in candidates else candidates[0]

    c1, c2 = st.columns([1, 1])
    with c1:
        chosen = st.selectbox(
            "Which sensor should we rewrite history for?",
            candidates, index=candidates.index(default_var),
            format_func=lambda v: (f"{label(v)}"
                                   + ("  🔴 root cause" if ROLES.get(v) == "cause"
                                      else "  🟠 symptom" if ROLES.get(v) == "symptom" else "")),
        )
    with c2:
        lead = st.slider(
            "From how many cycles before failure?",
            min_value=5, max_value=min(60, max(10, onset_idx - model.tau_max)),
            value=int(cfg.MAINTENANCE_LEAD_TIME), step=5,
        )

    start_idx = max(model.tau_max, onset_idx - lead)
    observed = engine[chosen].to_numpy(dtype=float)
    lo, hi = float(np.percentile(observed, 2)), float(np.percentile(observed, 98))
    actual_level = float(observed[start_idx:].mean())

    # Open on the value the automated search found to work, so the page starts
    # on a real answer. The slider can then be moved to explore alternatives.
    if cf.get("found") and cf.get("best") and cf["best"]["result"]["variable"] == chosen:
        default_value = float(cf["best"]["result"]["cf_value"])
    else:
        default_value = float(np.percentile(observed, 10))
    default_value = float(np.clip(default_value, lo, hi))

    what_if = st.slider(
        f"🎚️  What if **{short_label(chosen)}** had been held at…",
        min_value=round(lo, 2), max_value=round(hi, 2),
        value=round(default_value, 2),
        step=round(max((hi - lo) / 100, 0.01), 3),
        help=f"Its actual average from this point was {actual_level:.2f}. The slider opens on "
             f"the value the automated search found to work, where one exists.",
    )

    # Abduction — recover this engine's own noise (cached, so sliders stay fast).
    noise = get_noise(model, int(unit), engine)

    health_cf, _ = simulate_health(model, engine, readout, chosen, what_if, start_idx, noise=noise)
    from causalpdm.framework import scm as scm_module
    twin_factual = scm_module.simulate(model, engine, noise=noise)
    health_twin = readout.transform(twin_factual)

    tail = slice(start_idx, len(engine))
    peak_twin = float(health_twin.iloc[tail].max())
    peak_cf = float(health_cf.iloc[tail].max())
    valid_cf = peak_cf < threshold

    spread = float(observed.std()) or 1.0
    proximity = abs(what_if - actual_level) / spread
    plausible = bool(observed.min() <= what_if <= observed.max())

    st.subheader("Would it have worked?")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Valid", "✅ Yes" if valid_cf else "❌ No",
              help="Does this change actually keep the engine below the failure threshold?")
    m2.metric("Proximity", f"{proximity:.2f} sd",
              help="How big a change was needed. Smaller is a better explanation.")
    m3.metric("Sparsity", "1 sensor",
              help="How many variables had to change. Fewer is a better explanation.")
    m4.metric("Plausible", "✅ Yes" if plausible else "⚠️ Outside range",
              help="Is this value inside the engine's real observed operating range?")

    if valid_cf and plausible:
        st.success(
            f"**Counterfactual explanation.** Engine #{unit} reached a degraded state. Had "
            f"{short_label(chosen)} (`{chosen}`) been held at **{what_if:.2f}** from cycle "
            f"**{int(engine.loc[start_idx, 'cycle'])}** onward — a shift of {proximity:.1f} "
            f"standard deviations from its actual level of {actual_level:.2f} — the health "
            f"indicator would have peaked at **{peak_cf:.3f}** instead of **{peak_twin:.3f}**, "
            f"staying below the {threshold:.3f} failure threshold.", icon="✅")
    elif valid_cf and not plausible:
        st.warning(f"This value does avoid failure, but **{what_if:.2f} lies outside the range "
                   f"this engine was ever observed in** "
                   f"({observed.min():.2f} to {observed.max():.2f}). A physically unreachable "
                   f"explanation is not a maintenance instruction.", icon="⚠️")
    else:
        st.error(f"**Holding {short_label(chosen)} at {what_if:.2f} would not have prevented "
                 f"failure.** Health still peaks at {peak_cf:.3f}, above the {threshold:.3f} "
                 f"threshold. Move the slider, act earlier, or try a different sensor.", icon="❌")

    st.plotly_chart(
        charts.before_after_health(
            engine["cycle"].to_numpy(), health_twin.to_numpy(), health_cf.to_numpy(),
            threshold, start_idx,
            label_modified=f"counterfactual: {short_label(chosen)} = {what_if:.2f}",
            title=f"The twin world — same engine, same noise, one sensor rewritten",
        ),
        width="stretch")

    st.caption("Both lines are the *same engine with the same recovered noise*. They differ only "
               "through the intervention — which is exactly what makes this a counterfactual "
               "rather than a comparison against some other engine.")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("The smallest change that works")
        st.caption("Values tried nearest-first, from the automated search in the pipeline.")
        if cf.get("found"):
            st.dataframe(cf["attempts"].round(3), width="stretch", hide_index=True)
        else:
            st.info("The automated search found no single-sensor change that avoided failure. "
                    "That is a legitimate result: it points to component replacement rather than "
                    "an operating-point adjustment.")
    with c2:
        st.subheader("Sweep the whole range")
        grid, peaks = sweep_values(model, int(unit), engine, readout, chosen,
                                   int(start_idx), lo, hi, 11, True, noise)
        st.plotly_chart(
            charts.dose_response(grid, peaks, actual_level, threshold, short_label(chosen)),
            width="stretch")
        st.caption("Where this curve dips below the red line, failure would have been avoided.")


# =============================================================================
# Hyperparameters
# =============================================================================

elif page == "Hyperparameters":
    st.title("Hyperparameters")
    st.caption("Every setting the pipeline uses: what it means, what range is sensible, what we "
               "chose, and why. All of these live in `pipeline_config.py`.")

    st.info("These are **prototype** settings, chosen so the whole pipeline finishes in seconds on "
            "a laptop. Systematic tuning is a later step — the goal right now is a working "
            "end-to-end framework.", icon="🧪")

    current = {
        "DATASET": cfg.DATASET,
        "ENGINES_FOR_DISCOVERY": cfg.ENGINES_FOR_DISCOVERY,
        "CAUSAL_VARIABLES": f"{len(variables)} sensors",
        "MAX_LAG": cfg.MAX_LAG,
        "SIGNIFICANCE": cfg.SIGNIFICANCE,
        "MAX_CONDITION_SET_SIZE": cfg.MAX_CONDITION_SET_SIZE,
        "MAX_COMBINATIONS": cfg.MAX_COMBINATIONS,
        "REMOVE_TREND_FOR_DISCOVERY": cfg.REMOVE_TREND_FOR_DISCOVERY,
        "BOOTSTRAP_RUNS": cfg.BOOTSTRAP_RUNS,
        "STABILITY_THRESHOLD": cfg.STABILITY_THRESHOLD,
        "MAINTENANCE_LEAD_TIME": cfg.MAINTENANCE_LEAD_TIME,
        "INTERVENTION_CANDIDATES": cfg.INTERVENTION_CANDIDATES,
    }

    for group in ["Data", "Discovery", "Validation", "Analysis"]:
        st.subheader(group)
        rows = [p for p in explain.HYPERPARAMETERS if p["group"] == group]
        for p in rows:
            with st.expander(f"**{p['name']}**  —  currently `{current.get(p['name'], '?')}`"):
                st.markdown(f"**What it means.** {p['meaning']}")
                st.markdown(f"**Sensible range.** `{p['range']}`")
                st.markdown(f"**Effect of changing it.** {p['effect']}")

    st.divider()
    st.subheader("Summary table")
    table = pd.DataFrame([
        {"setting": p["name"], "group": p["group"],
         "current value": str(current.get(p["name"], "—")), "sensible range": p["range"]}
        for p in explain.HYPERPARAMETERS
    ])
    st.dataframe(table, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Changing them")
    st.markdown("Edit `pipeline_config.py`, then re-run the pipeline and refresh this page:")
    st.code("python scripts/run_pipeline.py", language="bash")
    st.caption("Cached stages are reused, so unless you change a discovery setting it finishes in "
               "about two seconds. To force everything to recompute, set `USE_CACHE = False`.")


# =============================================================================
# Recommendations
# =============================================================================

elif page == "Recommendations":
    st.title("Decision Support")
    st.caption(f"What the framework concludes for engine #{unit}.")

    roots = rca_result["root_causes"]
    symptoms = rca_result["symptoms"]

    st.subheader("1 · What to inspect")
    if roots.empty:
        st.info("No root cause exceeded the drift threshold for this engine.")
    else:
        top = roots.iloc[0]
        st.error(f"**Start with {label(top['variable'])}.** It drifted "
                 f"{top['abs_deviation']:.1f} standard deviations from this engine's healthy "
                 f"baseline, and it causally precedes "
                 f"{int(top['n_symptoms_explained'])} of the other abnormal signals — so it is "
                 f"an upstream cause, not a knock-on effect.", icon="🔴")
        if len(roots) > 1:
            others = ", ".join(short_label(v) for v in roots["variable"].iloc[1:4])
            st.markdown(f"**Also check in the same visit:** {others}")

    st.subheader("2 · What NOT to touch")
    if symptoms.empty:
        st.caption("No downstream symptoms were identified.")
    else:
        names = ", ".join(f"**{short_label(v)}**" for v in symptoms["variable"].head(3))
        st.warning(f"{names} look alarming but sit *downstream* of the identified causes in the "
                   f"causal graph. Acting on them treats the symptom while the cause continues. "
                   f"This distinction is the whole point of using causal inference rather than "
                   f"correlation.", icon="🟠")

    st.subheader("3 · What action to take")
    ranking = interv["ranking"]
    if not ranking.empty and float(ranking.iloc[0]["health_improvement"]) > 0:
        best = ranking.iloc[0]
        st.success(f"**Control {label(best['variable'])}.** Holding it near "
                   f"{best['do_value']:.2f} from {cfg.MAINTENANCE_LEAD_TIME} cycles before onset "
                   f"lowers projected health loss by "
                   f"{best['health_improvement']:.3f} — the highest-yield single action among "
                   f"those tested.", icon="✅")
    else:
        st.warning("No single-sensor action improved projected health, which points to component "
                   "replacement rather than an operational adjustment.", icon="🔧")

    st.subheader("4 · Would it have prevented the failure?")
    if cf.get("found") and cf.get("best"):
        m = cf["best"]["metrics"]
        st.success(cf["explanation"], icon="🔮")
        c1, c2, c3 = st.columns(3)
        c1.metric("Change required", f"{m['proximity_sd']:.2f} sd")
        c2.metric("Physically achievable", "Yes" if m["plausible"] else "Borderline")
        c3.metric("Health loss avoided", f"{m['health_reduction']:.3f}")
    else:
        st.warning("No achievable single-sensor change would have prevented this failure — "
                   "schedule component replacement rather than an operating-point change.",
                   icon="🔧")
    st.caption("Explore other possibilities yourself on the Interventions and Counterfactual "
               "pages, using the sliders.")

    st.divider()
    st.subheader("How far these conclusions can be pushed")
    for title, body, kind in explain.SCOPE_NOTES:
        (st.success if kind == "ok" else st.warning)(f"**{title}** — {body}")
