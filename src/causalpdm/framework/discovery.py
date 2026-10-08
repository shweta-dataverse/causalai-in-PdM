"""Module 3 — Causal discovery with PCMCI+.

PCMCI+ (Runge, 2020) is used because it is designed for exactly this setting:
autocorrelated multivariate time series, with both lagged and contemporaneous
links, and false-positive control via the momentary conditional independence
test. All 100 engines are pooled as independent realisations of the same
mechanism using tigramite's multiple-realisation analysis mode, which is the
statistically correct alternative to concatenating trajectories end to end
(concatenation would invent causal links across engine boundaries).

Assumptions this module relies on, stated explicitly because the thesis must
defend them:
  1. Causal sufficiency  - no unobserved common causes. Degradation is such a
     confounder, which is why Module 1 detrends per engine.
  2. Faithfulness        - no deterministic relations. Enforced by keeping
     composite indicators out of the variable set (see `features`).
  3. Stationarity        - within-engine mechanisms are time-invariant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI


def build_dataframe(
    df: pd.DataFrame, var_names: list[str], min_length: int = 40
) -> tuple[pp.DataFrame, list[int]]:
    """Assemble a tigramite DataFrame with one realisation per engine."""
    data_dict, used_units = {}, []
    for idx, (unit, block) in enumerate(df.groupby("unit")):
        if len(block) < min_length:
            continue
        data_dict[len(used_units)] = block[var_names].to_numpy(dtype=float)
        used_units.append(int(unit))
        del idx
    frame = pp.DataFrame(
        data_dict, analysis_mode="multiple", var_names=list(var_names)
    )
    return frame, used_units


def exogenous_link_assumptions(
    var_names: list[str], exogenous: list[str], tau_max: int
) -> dict:
    """Forbid any incoming edge to variables that are exogenous by design.

    In C-MAPSS the operational settings are the *commanded* flight condition:
    altitude, Mach number and throttle-resolver angle are inputs to the engine,
    never consequences of it. Encoding that as a link assumption is both
    physically correct and a large computational saving, because it removes a
    whole block of candidate links from the conditioning-set search.

    tigramite expects, for each target j, a dict of allowed parent links.
    """
    index = {v: i for i, v in enumerate(var_names)}
    assumptions: dict[int, dict] = {}
    for target in var_names:
        j = index[target]
        allowed: dict = {}
        for source in var_names:
            i = index[source]
            for lag in range(0, tau_max + 1):
                if i == j and lag == 0:
                    continue
                if target in exogenous and source not in exogenous:
                    continue  # no sensor may cause a commanded setting
                allowed[(i, -lag)] = "-?>" if lag > 0 else "o?o"
        assumptions[j] = allowed
    return assumptions


def run_pcmci_plus(
    df: pd.DataFrame,
    var_names: list[str],
    tau_max: int = 3,
    pc_alpha: float = 0.01,
    min_length: int = 40,
    max_conds_dim: int | None = 3,
    max_combinations: int = 2,
    exogenous: list[str] | None = None,
) -> dict:
    """Run PCMCI+ and return the graph plus the settings used to produce it.

    `max_conds_dim` and `max_combinations` bound the conditioning-set search.
    They are required in practice, not optional tuning: with 17 variables and
    pooled data the number of candidate conditioning sets grows combinatorially,
    and an unbounded search does not terminate in useful time. Bounding the
    search is the standard remedy (Runge et al.) and trades a small amount of
    recall for tractability - the trade-off is reported in the sensitivity
    analysis in Module 4.
    """
    frame, used_units = build_dataframe(df, var_names, min_length)
    pcmci = PCMCI(dataframe=frame, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)

    link_assumptions = (
        exogenous_link_assumptions(list(var_names), exogenous, tau_max) if exogenous else None
    )
    results = pcmci.run_pcmciplus(
        tau_min=0,
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        max_conds_dim=max_conds_dim,
        max_combinations=max_combinations,
        link_assumptions=link_assumptions,
    )

    return {
        "graph": results["graph"],
        "val_matrix": results["val_matrix"],
        "p_matrix": results["p_matrix"],
        "var_names": list(var_names),
        "tau_max": tau_max,
        "pc_alpha": pc_alpha,
        "max_conds_dim": max_conds_dim,
        "max_combinations": max_combinations,
        "exogenous": list(exogenous) if exogenous else [],
        "n_realisations": len(used_units),
        "units_used": used_units,
    }


def edge_list(result: dict, min_abs_strength: float = 0.0) -> pd.DataFrame:
    """Flatten a PCMCI+ graph into a tidy edge table.

    Only directed links ('-->') are retained as causal edges. Ambiguous or
    unoriented link types are reported separately by `link_type_summary` -
    silently treating them as directed would overstate what was discovered.
    """
    graph, val = result["graph"], result["val_matrix"]
    names = result["var_names"]
    rows = []
    n = len(names)
    for i in range(n):
        for j in range(n):
            for lag in range(graph.shape[2]):
                link = graph[i, j, lag]
                if link != "-->":
                    continue
                if i == j and lag == 0:
                    continue
                strength = float(val[i, j, lag])
                if abs(strength) < min_abs_strength:
                    continue
                rows.append(
                    {
                        "source": names[i],
                        "target": names[j],
                        "lag": int(lag),
                        "strength": strength,
                        "abs_strength": abs(strength),
                    }
                )
    return pd.DataFrame(rows).sort_values("abs_strength", ascending=False, ignore_index=True)


def link_type_summary(result: dict) -> pd.Series:
    """Count each PCMCI+ link type, including unoriented/ambiguous ones."""
    flat = result["graph"].flatten()
    return pd.Series([x for x in flat if x != ""]).value_counts()


def summary_graph(edges: pd.DataFrame) -> pd.DataFrame:
    """Collapse lagged edges into a summary graph (max |strength| per pair)."""
    if edges.empty:
        return edges
    return (
        edges.groupby(["source", "target"], as_index=False)
        .agg(
            abs_strength=("abs_strength", "max"),
            strength=("strength", lambda s: s.loc[s.abs().idxmax()]),
            min_lag=("lag", "min"),
            n_lags=("lag", "nunique"),
        )
        .sort_values("abs_strength", ascending=False, ignore_index=True)
    )


def run(df: pd.DataFrame, var_names: list[str], **kwargs) -> dict:
    """Execute Module 3 end to end."""
    result = run_pcmci_plus(df, var_names, **kwargs)
    edges = edge_list(result)
    return {
        **result,
        "edges": edges,
        "summary": summary_graph(edges),
        "link_types": link_type_summary(result),
        "n_directed_edges": int(len(edges)),
    }
