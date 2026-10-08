"""
Interactive Plotly charts for the dashboard.

Everything here returns a Plotly figure. Charts are interactive by design -
hover for detail, zoom, pan - because the whole point of the dashboard is
letting someone explore the results rather than look at a fixed picture.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from explain import SENSOR_SECTION, short_label

# Colours used consistently across every chart.
COLOURS = {
    "cause": "#dc2626",      # red - a root cause
    "symptom": "#f59e0b",    # amber - a downstream symptom
    "neutral": "#94a3b8",    # grey - everything else
    "primary": "#2563eb",    # blue - factual / main series
    "good": "#059669",       # green - counterfactual / improved
    "edge": "#cbd5e1",
}

SECTION_COLOURS = {
    "Fan / inlet": "#60a5fa",
    "Low-pressure compressor": "#34d399",
    "High-pressure compressor": "#fbbf24",
    "Combustor": "#f87171",
    "High-pressure turbine": "#c084fc",
    "Low-pressure turbine": "#38bdf8",
    "Whole engine": "#94a3b8",
}


def _clean(fig: go.Figure, height: int = 420, legend_below: bool = False) -> go.Figure:
    """Apply one consistent look to every chart.

    `legend_below` puts the legend under the plot instead of above it. Charts
    with a title and two or more series need this, otherwise the legend and the
    title fight for the same strip of space.
    """
    if legend_below:
        legend = dict(orientation="h", yanchor="top", y=-0.18, x=0)
        margin = dict(l=10, r=10, t=50, b=60)
    else:
        legend = dict(orientation="h", yanchor="bottom", y=1.02, x=0)
        margin = dict(l=10, r=10, t=45, b=10)

    fig.update_layout(
        height=height,
        margin=margin,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        legend=legend,
        title=dict(x=0, xanchor="left", font=dict(size=15)),
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.20)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.20)", zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# Health / degradation
# ---------------------------------------------------------------------------

def health_trajectory(engine: pd.DataFrame, threshold: float, onset_cycle: int,
                      unit: int, fleet: pd.DataFrame | None = None) -> go.Figure:
    """One engine's health score over its life, against the fleet envelope."""
    fig = go.Figure()

    if fleet is not None:
        band = fleet.groupby("cycle")["health_indicator"].quantile([0.1, 0.9]).unstack()
        fig.add_trace(go.Scatter(
            x=list(band.index) + list(band.index[::-1]),
            y=list(band[0.9]) + list(band[0.1][::-1]),
            fill="toself", fillcolor="rgba(37,99,235,0.10)",
            line=dict(width=0), name="fleet 10–90%", hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=engine["cycle"], y=engine["health_indicator"],
        mode="lines", line=dict(color=COLOURS["primary"], width=3),
        name=f"engine {unit}",
        hovertemplate="cycle %{x}<br>health loss %{y:.3f}<extra></extra>",
    ))

    fig.add_hline(y=threshold, line=dict(color=COLOURS["cause"], dash="dash", width=2),
                  annotation_text="failure threshold", annotation_position="right")
    fig.add_vline(x=onset_cycle, line=dict(color="#d97706", dash="dot", width=2),
                  annotation_text=f"degraded @ {onset_cycle}", annotation_position="top left")

    fig.update_layout(title=f"Engine {unit} — health loss over its life",
                      xaxis_title="operating cycle", yaxis_title="health loss (0 = new)")
    return _clean(fig, legend_below=True)


# ---------------------------------------------------------------------------
# Causal graph
# ---------------------------------------------------------------------------

def _causal_flow_layout(g: nx.DiGraph) -> dict:
    """Lay the graph out left-to-right so causes sit left of their effects.

    A spring layout scatters nodes by connection strength, which looks organic
    but hides the thing that matters most here: direction. Placing each node in
    a column according to how far downstream it is makes the graph readable as
    a flow - upstream causes on the left, final effects on the right.

    Falls back to a spring layout if the graph has a cycle (feedback loops are
    physically possible in a controlled engine, so this is not an error).
    """
    try:
        generations = list(nx.topological_generations(g))
    except nx.NetworkXUnfeasible:
        return nx.spring_layout(g, seed=11, k=1.6 / np.sqrt(max(g.number_of_nodes(), 1)),
                                iterations=250)

    pos = {}
    n_cols = max(len(generations), 1)
    for col, layer in enumerate(generations):
        layer = sorted(layer)
        x = col / max(n_cols - 1, 1)
        for row, node in enumerate(layer):
            # Centre each column vertically, and stagger alternate columns a
            # little so long edges are less likely to pass through a node.
            y = 0.5 if len(layer) == 1 else row / (len(layer) - 1)
            pos[node] = (x, y + (0.04 if col % 2 else -0.04))
    return pos


def causal_graph(edges: pd.DataFrame, roles: dict[str, str] | None = None,
                 title: str = "Discovered causal graph",
                 colour_by_section: bool = False) -> go.Figure:
    """Interactive causal graph. Hover a node to see its role and connections."""
    roles = roles or {}

    g = nx.DiGraph()
    for _, e in edges.iterrows():
        if e["source"] == e["target"]:
            continue
        w = abs(float(e["abs_strength"]))
        if g.has_edge(e["source"], e["target"]):
            g[e["source"]][e["target"]]["weight"] = max(g[e["source"]][e["target"]]["weight"], w)
            g[e["source"]][e["target"]]["lags"].append(int(e["lag"]))
        else:
            g.add_edge(e["source"], e["target"], weight=w, lags=[int(e["lag"])])

    if g.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(text="No causal edges to display", showarrow=False)
        return _clean(fig)

    pos = _causal_flow_layout(g)
    max_w = max(d["weight"] for *_, d in g.edges(data=True)) or 1.0

    fig = go.Figure()

    # Edges, drawn one at a time so each can carry its own width and hover text.
    for u, v, d in g.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        lags = ", ".join(str(l) for l in sorted(set(d["lags"])))
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(width=1 + 4 * d["weight"] / max_w, color=COLOURS["edge"]),
            hovertemplate=(f"<b>{short_label(u)} → {short_label(v)}</b><br>"
                           f"strength {d['weight']:.3f}<br>"
                           f"acts after {lags} cycle(s)<extra></extra>"),
            showlegend=False,
        ))
        # Arrowhead, so direction is unmistakable.
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.3, arrowwidth=1.4,
            arrowcolor="#94a3b8", opacity=0.85, standoff=17,
        )

    # Nodes
    node_x, node_y, node_colour, node_text, node_size = [], [], [], [], []
    for n in g.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)

        role = roles.get(n, "other")
        if colour_by_section:
            colour = SECTION_COLOURS.get(SENSOR_SECTION.get(n, "Whole engine"), COLOURS["neutral"])
        elif role == "cause":
            colour = COLOURS["cause"]
        elif role == "symptom":
            colour = COLOURS["symptom"]
        else:
            colour = COLOURS["neutral"]
        node_colour.append(colour)

        role_word = {"cause": "ROOT CAUSE", "symptom": "downstream symptom"}.get(role, "")
        causes = [short_label(p) for p in g.predecessors(n)]
        affects = [short_label(s) for s in g.successors(n)]
        node_text.append(
            f"<b>{n} — {short_label(n)}</b>"
            + (f"<br><b>{role_word}</b>" if role_word else "")
            + f"<br>part: {SENSOR_SECTION.get(n, 'n/a')}"
            + f"<br>caused by: {', '.join(causes) if causes else 'nothing in graph'}"
            + f"<br>affects: {', '.join(affects) if affects else 'nothing in graph'}"
        )
        node_size.append(34 if role == "cause" else 28 if role == "symptom" else 22)

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=node_size, color=node_colour,
                    line=dict(width=2, color="rgba(255,255,255,0.9)")),
        text=list(g.nodes()), textposition="middle center",
        textfont=dict(size=9, color="white", family="monospace"),
        hovertext=node_text, hoverinfo="text", showlegend=False,
    ))

    fig.update_layout(title=title)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _clean(fig, height=560)


# ---------------------------------------------------------------------------
# Root cause analysis
# ---------------------------------------------------------------------------

def deviation_bars(deviations: pd.DataFrame, roles: dict[str, str], top_n: int = 12) -> go.Figure:
    """How far each sensor drifted from this engine's own healthy baseline."""
    d = deviations.head(top_n).iloc[::-1]
    colours = [COLOURS["cause"] if roles.get(v) == "cause"
               else COLOURS["symptom"] if roles.get(v) == "symptom"
               else COLOURS["neutral"] for v in d["variable"]]

    fig = go.Figure(go.Bar(
        x=d["deviation_z"], y=[short_label(v) for v in d["variable"]],
        orientation="h", marker_color=colours,
        customdata=d["variable"],
        hovertemplate="<b>%{customdata}</b><br>%{y}<br>drift %{x:.2f} sd<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="#64748b", width=1))
    fig.update_layout(title="How far each sensor drifted before failure",
                      xaxis_title="standard deviations from healthy baseline")
    return _clean(fig, height=420)


def root_cause_scatter(ranking: pd.DataFrame) -> go.Figure:
    """Drift vs causal reach - the two ingredients of the root-cause score."""
    fig = go.Figure()
    for is_cause, group, colour, name in [
        (True, ranking[ranking["is_root_cause"]], COLOURS["cause"], "root cause"),
        (False, ranking[~ranking["is_root_cause"]], COLOURS["symptom"], "symptom"),
    ]:
        if group.empty:
            continue
        fig.add_trace(go.Scatter(
            x=group["abs_deviation"], y=group["causal_reach"],
            mode="markers+text", name=name,
            marker=dict(size=16, color=colour, line=dict(width=1.5, color="white")),
            text=group["variable"], textposition="top center", textfont=dict(size=9),
            hovertemplate="<b>%{text}</b><br>drift %{x:.2f} sd<br>"
                          "causal reach %{y:.3f}<extra></extra>",
        ))
    fig.update_layout(
        title="Why each sensor got its verdict",
        xaxis_title="drift from healthy baseline (sd)  →  how abnormal",
        yaxis_title="causal reach  →  how much else it explains",
    )
    return _clean(fig, height=460, legend_below=True)


# ---------------------------------------------------------------------------
# Interventions and counterfactuals
# ---------------------------------------------------------------------------

def before_after_health(cycles, factual, modified, threshold: float,
                        start_idx: int, label_modified: str, title: str) -> go.Figure:
    """Two health trajectories: what happened, and what the model says would have."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cycles, y=factual, mode="lines", name="what actually happened",
        line=dict(color=COLOURS["cause"], width=3),
        hovertemplate="cycle %{x}<br>health loss %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=cycles, y=modified, mode="lines", name=label_modified,
        line=dict(color=COLOURS["good"], width=3),
        hovertemplate="cycle %{x}<br>health loss %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=threshold, line=dict(color="#334155", dash="dash", width=2),
                  annotation_text="failure threshold", annotation_position="right")
    if start_idx is not None and start_idx < len(cycles):
        fig.add_vline(x=cycles[start_idx], line=dict(color="#d97706", dash="dot", width=2),
                      annotation_text="action starts", annotation_position="top left")
    fig.update_layout(title=title, xaxis_title="operating cycle",
                      yaxis_title="health loss")
    return _clean(fig, height=460, legend_below=True)


def dose_response(values, health, current_value: float, threshold: float,
                  variable: str) -> go.Figure:
    """Health outcome across the whole range of values we could force."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=values, y=health, mode="lines+markers",
        line=dict(color=COLOURS["primary"], width=3), marker=dict(size=7),
        name="resulting health loss",
        hovertemplate="hold at %{x:.2f}<br>health loss %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=threshold, line=dict(color=COLOURS["cause"], dash="dash", width=2),
                  annotation_text="failure threshold", annotation_position="right")
    fig.add_vline(x=current_value, line=dict(color="#64748b", dash="dot", width=2),
                  annotation_text="actual level", annotation_position="top")
    fig.update_layout(
        title=f"Dose–response: what different values of {variable} would achieve",
        xaxis_title=f"value we force {variable} to hold",
        yaxis_title="resulting health loss",
    )
    return _clean(fig, height=400)


def intervention_ranking(ranking: pd.DataFrame) -> go.Figure:
    """Which lever helps most."""
    d = ranking.iloc[::-1]
    colours = [COLOURS["good"] if v > 0 else COLOURS["neutral"] for v in d["health_improvement"]]
    fig = go.Figure(go.Bar(
        x=d["health_improvement"], y=[short_label(v) for v in d["variable"]],
        orientation="h", marker_color=colours, customdata=d["variable"],
        hovertemplate="<b>%{customdata}</b><br>health improvement %{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="#64748b", width=1))
    fig.update_layout(title="Which action helps most (estimated)",
                      xaxis_title="reduction in health loss under do()")
    return _clean(fig, height=350)


# ---------------------------------------------------------------------------
# Validation and SCM
# ---------------------------------------------------------------------------

def stability_bars(stability: pd.DataFrame, threshold: float, top_n: int = 18) -> go.Figure:
    """How often each edge reappeared when we re-ran discovery on subsets."""
    d = stability.head(top_n).iloc[::-1]
    labels = [f"{r['source']} → {r['target']} (lag {int(r['lag'])})" for _, r in d.iterrows()]
    colours = [COLOURS["good"] if r >= threshold else COLOURS["neutral"]
               for r in d["recovery_rate"]]
    fig = go.Figure(go.Bar(
        x=d["recovery_rate"], y=labels, orientation="h", marker_color=colours,
        hovertemplate="%{y}<br>reappeared in %{x:.0%} of runs<extra></extra>",
    ))
    fig.add_vline(x=threshold, line=dict(color=COLOURS["cause"], dash="dash", width=2),
                  annotation_text="keep threshold")
    fig.update_layout(title="Edge stability — green edges are kept",
                      xaxis_title="fraction of bootstrap runs the edge reappeared in",
                      xaxis=dict(tickformat=".0%"))
    return _clean(fig, height=520)


def mechanism_fit(summary: pd.DataFrame) -> go.Figure:
    """How well each sensor is explained by its causal parents."""
    d = summary.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d["r2"], y=[short_label(v) for v in d["variable"]], orientation="h",
        marker_color=COLOURS["primary"], customdata=d["variable"],
        hovertemplate="<b>%{customdata}</b><br>R² = %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(title="How well each sensor is explained by its causes",
                      xaxis_title="R² of the fitted equation", xaxis_range=[0, 1])
    return _clean(fig, height=400)
