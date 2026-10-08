"""Module 6 — Root cause analysis.

The central requirement of the thesis is distinguishing **root causes** from
**downstream symptoms**. A purely associational method cannot: at failure most
sensors deviate at once, and correlation ranks the loudest symptom first. The
causal graph supplies the missing ingredient - direction.

Procedure
---------
1. Define the event. The failure onset comes from Module 2's health readout.
2. Measure deviation. Compare each variable in the pre-failure window against
   that engine's own healthy baseline (a within-engine contrast, so fleet-level
   variation cannot masquerade as a cause).
3. Trace causal paths. For every deviating variable, find which other deviating
   variables it can reach in the validated graph.
4. Separate causes from symptoms. A deviating variable that is itself explained
   by another deviating ancestor is a symptom; one with no deviating ancestor is
   a root cause candidate.
5. Rank by causal influence: deviation magnitude x weighted causal reach.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def build_summary_digraph(edges: pd.DataFrame) -> nx.DiGraph:
    """Collapse the time-lagged graph into a summary DAG for path tracing."""
    g = nx.DiGraph()
    for _, e in edges.iterrows():
        src, tgt = e["source"], e["target"]
        if src == tgt:
            continue
        weight = abs(float(e["abs_strength"]))
        if g.has_edge(src, tgt):
            g[src][tgt]["weight"] = max(g[src][tgt]["weight"], weight)
            g[src][tgt]["min_lag"] = min(g[src][tgt]["min_lag"], int(e["lag"]))
        else:
            g.add_edge(src, tgt, weight=weight, min_lag=int(e["lag"]))
    return g


def deviation_scores(
    engine: pd.DataFrame,
    var_names: list[str],
    onset_idx: int,
    baseline_frac: float = 0.3,
    window: int = 20,
) -> pd.DataFrame:
    """Standardised deviation of each variable in the pre-failure window.

    The baseline is the engine's own early life, making this a within-unit
    contrast: it asks "what changed in *this* engine", not "how does this engine
    compare with the fleet".
    """
    n_baseline = max(10, int(baseline_frac * onset_idx)) if onset_idx > 0 else 10
    baseline = engine.iloc[:n_baseline]
    start = max(0, onset_idx - window)
    event = engine.iloc[start : onset_idx + 1] if onset_idx > start else engine.iloc[start:]

    rows = []
    for v in var_names:
        b_mu, b_sd = float(baseline[v].mean()), float(baseline[v].std())
        e_mu = float(event[v].mean())
        z = (e_mu - b_mu) / b_sd if b_sd > 1e-9 else 0.0
        rows.append(
            {"variable": v, "baseline_mean": b_mu, "event_mean": e_mu,
             "deviation_z": z, "abs_deviation": abs(z)}
        )
    return pd.DataFrame(rows).sort_values("abs_deviation", ascending=False, ignore_index=True)


def rank_root_causes(
    graph: nx.DiGraph,
    deviations: pd.DataFrame,
    deviation_threshold: float = 1.5,
    max_path_length: int = 4,
) -> pd.DataFrame:
    """Rank deviating variables by causal influence over the other deviations."""
    dev = deviations.set_index("variable")["abs_deviation"].to_dict()
    deviating = {v for v, d in dev.items() if d >= deviation_threshold}
    if not deviating:
        return pd.DataFrame(
            columns=["variable", "abs_deviation", "causal_reach", "n_symptoms_explained",
                     "is_root_cause", "root_cause_score", "explained_symptoms"]
        )

    rows = []
    for v in deviating:
        if v not in graph:
            reach, explained = 0.0, []
        else:
            explained, reach = [], 0.0
            for target in deviating - {v}:
                if target not in graph:
                    continue
                try:
                    path = nx.shortest_path(graph, v, target)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                if len(path) - 1 > max_path_length:
                    continue
                strength = np.prod(
                    [graph[a][b]["weight"] for a, b in zip(path[:-1], path[1:])]
                )
                reach += float(strength) * dev[target]
                explained.append(" -> ".join(path))

        # A variable with a deviating ancestor is downstream of it: a symptom.
        ancestors_deviating = (
            bool(set(nx.ancestors(graph, v)) & (deviating - {v})) if v in graph else False
        )
        rows.append(
            {
                "variable": v,
                "abs_deviation": dev[v],
                "causal_reach": reach,
                "n_symptoms_explained": len(explained),
                "is_root_cause": not ancestors_deviating,
                "explained_symptoms": explained[:5],
            }
        )

    out = pd.DataFrame(rows)
    out["root_cause_score"] = out["abs_deviation"] * (1.0 + out["causal_reach"])
    out.loc[~out["is_root_cause"], "root_cause_score"] *= 0.25  # demote symptoms
    return out.sort_values(
        ["is_root_cause", "root_cause_score"], ascending=[False, False], ignore_index=True
    )


def analyse_engine(
    featured: pd.DataFrame,
    var_names: list[str],
    edges: pd.DataFrame,
    unit: int,
    deviation_threshold: float = 1.5,
) -> dict:
    """Run the full RCA chain for one engine."""
    engine = featured[featured["unit"] == unit].reset_index(drop=True)
    onset_positions = engine.index[engine["onset"] == 1]
    onset_idx = int(onset_positions[0]) if len(onset_positions) else int(len(engine) * 0.7)

    graph = build_summary_digraph(edges)
    deviations = deviation_scores(engine, var_names, onset_idx)
    ranking = rank_root_causes(graph, deviations, deviation_threshold)

    return {
        "unit": int(unit),
        "engine": engine,
        "onset_idx": onset_idx,
        "onset_cycle": int(engine.loc[onset_idx, "cycle"]),
        "total_cycles": int(len(engine)),
        "graph": graph,
        "deviations": deviations,
        "ranking": ranking,
        "root_causes": ranking[ranking["is_root_cause"]].head(5),
        "symptoms": ranking[~ranking["is_root_cause"]].head(5),
        "deviation_threshold": deviation_threshold,
    }


def run(
    featured: pd.DataFrame,
    var_names: list[str],
    edges: pd.DataFrame,
    unit: int | None = None,
    **kwargs,
) -> dict:
    """Execute Module 6. Defaults to the engine with the earliest failure onset."""
    if unit is None:
        onsets = featured[featured["onset"] == 1].groupby("unit")["cycle"].first()
        unit = int(onsets.idxmin()) if len(onsets) else int(featured["unit"].iloc[0])
    return analyse_engine(featured, var_names, edges, unit, **kwargs)
