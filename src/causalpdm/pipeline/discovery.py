"""
Learning the causal graph from data, using PCMCI+.

PCMCI+ (Runge, 2020) is built for exactly this situation: several variables
recorded over time, where a cause can act immediately or after a delay, and
where the signals are strongly autocorrelated.

It works in two passes. The first narrows each variable's long list of possible
causes down to a short list. The second re-tests every surviving link while
accounting for the causes of both ends, which is what keeps the false-positive
rate under control when data is autocorrelated.

Two choices here matter more than any other:

  The independence test must match the data. Every variable in the coolant
  subsystem is True/False, so we use the G-test for discrete data. A test built
  for continuous numbers would be measuring the wrong thing.

  The maximum lag must cover the real delays. causRCA's alarm rules fire either
  immediately or after a five second hold, so a lag window of only one or two
  steps cannot see half of the real links.

Recordings are passed as separate realisations rather than glued end to end,
which would invent causal links across the join between two recordings.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from tigramite import data_processing as pp
from tigramite.independence_tests.gsquared import Gsquared
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

# PCMCI marks each link it finds with a symbol. Only "-->" means "this causes
# that". The others mean a dependence was found but its direction could not be
# settled from the data, so counting them as causal would overstate the result.
DIRECTED = "-->"


@dataclass
class CausalGraph:
    """A learned causal graph, plus the settings that produced it."""

    edges: pd.DataFrame            # cause, effect, lag, strength
    variables: list[str]
    settings: dict
    link_type_counts: pd.Series    # including the links PCMCI could not orient

    @property
    def edge_pairs(self) -> set[tuple[str, str]]:
        """Edges as (cause, effect), collapsing lags, for scoring."""
        return set(zip(self.edges["cause"], self.edges["effect"]))

    def causes_of(self, variable: str) -> list[str]:
        return self.edges.loc[self.edges["effect"] == variable, "cause"].tolist()

    def effects_of(self, variable: str) -> list[str]:
        return self.edges.loc[self.edges["cause"] == variable, "effect"].tolist()


def make_independence_test(kind: str):
    """
    Pick the test that matches the data.

    "discrete"   G-test. Correct for True/False and categorical variables.
    "continuous" partial correlation. Correct for numeric sensor readings.
    """
    if kind == "discrete":
        return Gsquared()
    if kind == "continuous":
        return ParCorr(significance="analytic")
    raise ValueError(f"Unknown independence test '{kind}'. Use 'discrete' or 'continuous'.")


def discover(
    tables: list[pd.DataFrame],
    variables: list[str],
    max_lag: int = 8,
    significance: float | None = 0.01,
    independence_test: str = "discrete",
    multiple_test_correction: str = "fdr_bh",
    max_conditions: int = 3,
    max_combinations: int = 2,
) -> CausalGraph:
    """
    Learn a causal graph from a set of recordings.

    Setting `significance` to None lets PCMCI+ choose it from a standard range
    using its own model-selection score, instead of us guessing.

    `multiple_test_correction` matters more than it sounds. With 15 variables
    and 9 lags there are over two thousand candidate links, so without
    correction a handful of edges would appear purely by chance.
    """
    realisations = {i: table[variables].to_numpy(dtype=float)
                    for i, table in enumerate(tables)}
    frame = pp.DataFrame(realisations, analysis_mode="multiple", var_names=list(variables))

    pcmci = PCMCI(
        dataframe=frame,
        cond_ind_test=make_independence_test(independence_test),
        verbosity=0,
    )
    result = pcmci.run_pcmciplus(
        tau_min=0,
        tau_max=max_lag,
        pc_alpha=significance,
        max_conds_dim=max_conditions,
        max_combinations=max_combinations,
        fdr_method=multiple_test_correction,
    )

    return CausalGraph(
        edges=_collect_edges(result, variables),
        variables=list(variables),
        settings={
            "max_lag": max_lag,
            "significance": significance,
            "independence_test": independence_test,
            "multiple_test_correction": multiple_test_correction,
            "max_conditions": max_conditions,
            "max_combinations": max_combinations,
            "recordings_used": len(tables),
        },
        link_type_counts=_count_link_types(result),
    )


def _collect_edges(result: dict, variables: list[str]) -> pd.DataFrame:
    """Pull the directed links out of PCMCI's result into a readable table."""
    graph = result["graph"]
    strength = result["val_matrix"]
    rows = []
    for i, cause in enumerate(variables):
        for j, effect in enumerate(variables):
            for lag in range(graph.shape[2]):
                if graph[i, j, lag] != DIRECTED:
                    continue
                if i == j:
                    continue  # a variable depending on its own past is not news
                rows.append({
                    "cause": cause,
                    "effect": effect,
                    "lag": int(lag),
                    "strength": round(float(strength[i, j, lag]), 4),
                })
    edges = pd.DataFrame(rows, columns=["cause", "effect", "lag", "strength"])
    if edges.empty:
        return edges
    edges["abs_strength"] = edges["strength"].abs()
    return edges.sort_values("abs_strength", ascending=False, ignore_index=True)


def _count_link_types(result: dict) -> pd.Series:
    """How many links of each kind PCMCI reported, oriented or not."""
    found = [symbol for symbol in result["graph"].flatten() if symbol]
    return pd.Series(found).value_counts()


def strongest_edges(graph: CausalGraph, limit: int = 15) -> pd.DataFrame:
    """The clearest links, for a report or the dashboard."""
    if graph.edges.empty:
        return graph.edges
    return graph.edges.head(limit)[["cause", "effect", "lag", "strength"]]
