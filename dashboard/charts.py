"""
The charts used by the dashboard.

Everything here returns a Plotly figure. Charts are kept deliberately plain:
one idea per chart, no decoration that does not carry information.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

COLOURS = {
    "cause": "#c0392b",      # red - something that went wrong at the source
    "alarm": "#e67e22",      # orange - an alarm
    "normal": "#7f8c8d",     # grey - everything else
    "good": "#27ae60",       # green - fixed, or working
    "accent": "#2563eb",
    "line": "#bdc3c7",
}


def _tidy(figure: go.Figure, height: int = 420) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        title=dict(x=0, xanchor="left", font=dict(size=15)),
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0),
    )
    figure.update_xaxes(gridcolor="rgba(0,0,0,0.07)", zeroline=False)
    figure.update_yaxes(gridcolor="rgba(0,0,0,0.07)", zeroline=False)
    return figure


def _columns_left_to_right(graph: nx.DiGraph) -> list[list[str]]:
    """Group nodes into columns, ignoring feedback loops while doing so."""
    without_loops = graph.copy()
    while True:
        try:
            cycle = nx.find_cycle(without_loops, orientation="original")
        except nx.NetworkXNoCycle:
            break
        without_loops.remove_edge(cycle[0][0], cycle[0][1])
    return [sorted(column) for column in nx.topological_generations(without_loops)]


def causal_map(edges: pd.DataFrame, alarms: list[str],
               highlight: list[str] | None = None,
               title: str = "How the machine's signals affect each other") -> go.Figure:
    """
    The causal graph, laid out left to right so causes sit left of their effects.

    A spring layout looks organic but hides the one thing that matters here,
    which is direction. Columns make the flow obvious at a glance.
    """
    highlight = highlight or []
    graph = nx.DiGraph()
    for _, edge in edges.iterrows():
        if edge["cause"] != edge["effect"]:
            graph.add_edge(edge["cause"], edge["effect"], lag=int(edge["lag"]))

    if graph.number_of_nodes() == 0:
        figure = go.Figure()
        figure.add_annotation(text="No causal links to show", showarrow=False)
        return _tidy(figure)

    # Lay the nodes out in columns, so causes sit to the left of what they
    # affect. Real machines contain feedback loops - here the two pumps switch
    # each other off - and a loop has no left-to-right order, so we set those
    # edges aside while working out the columns and draw them anyway afterwards.
    columns = _columns_left_to_right(graph)

    position = {}
    for column_index, column in enumerate(columns):
        for row_index, name in enumerate(sorted(column)):
            x = column_index / max(len(columns) - 1, 1)
            y = 0.5 if len(column) == 1 else row_index / (len(column) - 1)
            position[name] = (x, y)

    figure = go.Figure()

    for cause, effect, info in graph.edges(data=True):
        x0, y0 = position[cause]
        x1, y1 = position[effect]
        delay = "straight away" if info["lag"] == 0 else f"after {info['lag']} steps"
        figure.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(width=2, color=COLOURS["line"]),
            hovertemplate=f"<b>{cause}</b> affects <b>{effect}</b><br>{delay}<extra></extra>",
            showlegend=False))
        figure.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y",
                              axref="x", ayref="y", showarrow=True, arrowhead=3,
                              arrowsize=1.2, arrowwidth=1.6,
                              arrowcolor=COLOURS["line"], standoff=20)

    xs, ys, colours, labels, hovers, sizes = [], [], [], [], [], []
    for name in graph.nodes():
        x, y = position[name]
        xs.append(x)
        ys.append(y)
        if name in highlight:
            colours.append(COLOURS["cause"])
            sizes.append(34)
        elif name in alarms:
            colours.append(COLOURS["alarm"])
            sizes.append(26)
        else:
            colours.append(COLOURS["normal"])
            sizes.append(22)
        labels.append(name)
        affects = list(graph.successors(name))
        caused_by = list(graph.predecessors(name))
        hovers.append(
            f"<b>{name}</b>"
            f"<br>affected by: {', '.join(caused_by) if caused_by else 'nothing'}"
            f"<br>affects: {', '.join(affects) if affects else 'nothing'}")

    figure.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(size=sizes, color=colours, line=dict(width=2, color="white")),
        text=labels, textposition="bottom center", textfont=dict(size=10),
        hovertext=hovers, hoverinfo="text", showlegend=False))

    figure.update_layout(title=title)
    figure.update_xaxes(visible=False, range=[-0.12, 1.12])
    figure.update_yaxes(visible=False, range=[-0.25, 1.15])
    return _tidy(figure, height=480)


def suspect_ranking(ranking: pd.DataFrame, true_causes: list[str],
                    top_n: int = 8) -> go.Figure:
    """How strongly each signal behaved in a way its own causes cannot explain."""
    top = ranking.head(top_n).iloc[::-1]
    colours = [COLOURS["cause"] if name in true_causes else COLOURS["normal"]
               for name in top["variable"]]
    figure = go.Figure(go.Bar(
        x=top["unexplained_score"], y=top["variable"], orientation="h",
        marker_color=colours,
        hovertemplate="<b>%{y}</b><br>unexplained behaviour: %{x:.2f}<extra></extra>"))
    figure.update_layout(
        title="Which signals behaved in a way nothing else explains",
        xaxis_title="the higher the bar, the less its own causes explain it")
    return _tidy(figure, height=360)


def alarm_timeline(recording: pd.DataFrame, alarms: list[str],
                   seconds_per_step: float, fault_window: tuple[int, int] | None = None,
                   title: str = "When each alarm fired") -> go.Figure:
    """A simple on/off strip per alarm, so the order of events is obvious."""
    present = [a for a in alarms if a in recording.columns]
    figure = go.Figure()
    time = np.arange(len(recording)) * seconds_per_step

    for row, alarm in enumerate(present):
        values = recording[alarm].to_numpy(dtype=float)
        firing = values > 0.5
        figure.add_trace(go.Scatter(
            x=time, y=np.where(firing, row, np.nan), mode="lines",
            line=dict(width=12, color=COLOURS["alarm"]), name=alarm,
            hovertemplate=f"<b>{alarm}</b> firing at %{{x:.0f}}s<extra></extra>",
            showlegend=False))

    if fault_window:
        start, end = fault_window
        figure.add_vrect(x0=start * seconds_per_step, x1=end * seconds_per_step,
                         fillcolor=COLOURS["accent"], opacity=0.08, line_width=0,
                         annotation_text="what we looked at", annotation_position="top left")

    figure.update_layout(title=title, xaxis_title="seconds into the recording")
    figure.update_yaxes(tickmode="array", tickvals=list(range(len(present))),
                        ticktext=present, range=[-0.6, len(present) - 0.4])
    return _tidy(figure, height=90 + 44 * max(len(present), 1))


def before_and_after(before: pd.DataFrame, after: pd.DataFrame, alarms: list[str],
                     seconds_per_step: float) -> go.Figure:
    """Which alarms fire now, and which would fire after the change."""
    present = [a for a in alarms if a in before.columns]
    rows = []
    for alarm in present:
        rows.append({
            "alarm": alarm,
            "now": bool((before[alarm].to_numpy(dtype=float) > 0.5).any()),
            "after": bool((after[alarm].to_numpy(dtype=float) > 0.5).any()),
        })
    table = pd.DataFrame(rows)

    figure = go.Figure()
    figure.add_trace(go.Bar(
        y=table["alarm"], x=table["now"].astype(int), orientation="h",
        name="fires now", marker_color=COLOURS["alarm"]))
    figure.add_trace(go.Bar(
        y=table["alarm"], x=table["after"].astype(int), orientation="h",
        name="fires after your change", marker_color=COLOURS["good"]))
    figure.update_layout(title="Alarms before and after the change", barmode="group",
                         xaxis=dict(tickvals=[0, 1], ticktext=["silent", "firing"]))
    return _tidy(figure, height=120 + 46 * max(len(present), 1))


def score_bars(scores: dict[str, float], title: str) -> go.Figure:
    """A plain bar per score, all on the same 0 to 1 scale."""
    names = list(scores)
    values = [scores[n] for n in names]
    colours = [COLOURS["good"] if v >= 0.7 else
               COLOURS["alarm"] if v >= 0.4 else COLOURS["cause"] for v in values]
    figure = go.Figure(go.Bar(
        x=values, y=names, orientation="h", marker_color=colours,
        text=[f"{v:.2f}" for v in values], textposition="outside",
        hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    figure.update_layout(title=title, xaxis_range=[0, 1.15])
    return _tidy(figure, height=120 + 42 * max(len(names), 1))


def mechanism_accuracy(summary: pd.DataFrame) -> go.Figure:
    """How well each signal is explained by the signals feeding into it."""
    data = summary.sort_values("accuracy").copy()
    figure = go.Figure(go.Bar(
        x=data["accuracy"], y=data["variable"], orientation="h",
        marker_color=COLOURS["accent"],
        customdata=data["causes"],
        hovertemplate="<b>%{y}</b><br>explained %{x:.0%} of the time"
                      "<br>driven by: %{customdata}<extra></extra>"))
    figure.update_layout(
        title="How well each signal is explained by its causes",
        xaxis_title="share of the time the equation gets it right",
        xaxis_range=[0, 1.05])
    return _tidy(figure, height=120 + 26 * max(len(data), 1))


def tuning_scatter(finished: pd.DataFrame) -> go.Figure:
    """
    Every combination of settings that was tried: accuracy against time taken.

    The useful corner is bottom right - accurate and quick.
    """
    figure = go.Figure()
    for grid, group in finished.groupby("seconds_per_step"):
        figure.add_trace(go.Scatter(
            x=group["f1"], y=group["seconds_to_run"], mode="markers",
            name=f"{grid}s per row",
            marker=dict(size=14, line=dict(width=1, color="white")),
            customdata=group[["max_lag", "significance", "edges_found", "correct"]],
            hovertemplate=(
                "accuracy %{x:.2f}<br>took %{y:.0f} seconds"
                "<br>steps back: %{customdata[0]}"
                "<br>evidence needed: %{customdata[1]}"
                "<br>links found: %{customdata[2]}, of those real: %{customdata[3]}"
                "<extra></extra>")))
    best = finished.loc[finished["f1"].idxmax()]
    figure.add_annotation(x=best["f1"], y=best["seconds_to_run"],
                          text="best", showarrow=True, arrowhead=2, ay=-34)
    figure.update_layout(title="Every setting tried: accuracy against time taken",
                         xaxis_title="accuracy (F1) - further right is better",
                         yaxis_title="seconds to run - lower is better")
    return _tidy(figure, height=430)
