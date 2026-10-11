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

Two ways of using the recordings are offered. `discover` pools them all into
one run, which has the most statistical power. `discover_by_vote` runs on
groups of recordings separately and keeps the links most groups agree on,
which is how the causRCA authors worked and is far more robust to one odd
recording.
"""

from __future__ import annotations

import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
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
    max_parents_conditioned: int | None = 6,
    alarms: list[str] | None = None,
    same_moment_partners: int | None = None,
) -> CausalGraph:
    """
    Learn a causal graph from a set of recordings.

    Passing `alarms` applies one piece of domain knowledge: an alarm is the
    controller reporting a condition, so it can be an effect but never a cause.
    Links out of an alarm are never tested, which removes a large share of the
    search and every chance of a false "alarm causes pump" link.

    Passing `same_moment_partners` screens the search before it starts; see
    `screen_same_moment_links`. It is what makes the full machine computable.

    Setting `significance` to None lets PCMCI+ choose it from a standard range
    using its own model-selection score, instead of us guessing.

    `max_parents_conditioned` caps how many already-known causes of each end of
    a link get conditioned on. On small subsystems this changes little, but it
    is not optional on the full 92-signal machine: the discrete test builds one
    array dimension per conditioned signal, and NumPy refuses arrays beyond 32
    dimensions. Unbounded, the full graph reached 39 and crashed.

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
    same_moment_pairs = None
    if same_moment_partners is not None:
        same_moment_pairs = screen_same_moment_links(
            pcmci.cond_ind_test, variables, same_moment_partners, significance or 0.05)
    restricted = alarms or same_moment_pairs is not None
    result = pcmci.run_pcmciplus(
        tau_min=0,
        tau_max=max_lag,
        pc_alpha=significance,
        max_conds_dim=max_conditions,
        max_combinations=max_combinations,
        max_conds_py=max_parents_conditioned,
        max_conds_px=max_parents_conditioned,
        fdr_method=multiple_test_correction,
        link_assumptions=(allowed_links(variables, max_lag, alarms or [], same_moment_pairs)
                          if restricted else None),
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
            "alarms_only_as_effects": bool(alarms),
            "same_moment_partners": same_moment_partners,
            "same_moment_pairs_kept": len(same_moment_pairs) if same_moment_pairs is not None else None,
            "recordings_used": len(tables),
        },
        link_type_counts=_count_link_types(result),
    )


def discover_by_vote(
    tables: list[pd.DataFrame],
    labels: list[str],
    variables: list[str],
    grouping: str = "replicates",
    vote_threshold: float = 0.5,
    workers: int | None = None,
    seed: int = 0,
    **settings,
) -> CausalGraph:
    """
    Learn a graph on groups of recordings separately and keep what most agree on.

    `grouping` decides which recordings vote together:

      "recording"   every recording votes alone. This is the causRCA authors'
                    protocol, kept for a like-for-like comparison. It has a
                    built-in weakness: a healthy recording never shows an alarm
                    and a fault recording only shows its own, so no alarm link
                    can win half the votes.

      "replicates"  every group holds one run of each fault scenario plus a
                    share of the healthy recordings - a repeat of the whole
                    experiment. A link is kept when it replicates.

    `vote_threshold` is fixed at 0.5 in advance, as in the paper. Choosing it by
    scoring against the expert graph would tune on the answer.

    Groups are independent, so they run in parallel. Each kept link records the
    share of groups that found it, which is a confidence per link.
    """
    groups = _vote_groups(labels, grouping, seed)
    jobs = [([tables[i] for i in group], variables, settings) for group in groups]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        found = list(pool.map(_discover_group, jobs))

    finished = [edges for edges in found if edges is not None]
    votes: dict[tuple[str, str], list] = defaultdict(list)
    for edges in finished:
        for (cause, effect), rows in edges.groupby(["cause", "effect"]):
            votes[(cause, effect)].append(rows.iloc[0])   # strongest lag in this group

    rows = []
    for (cause, effect), found_in in votes.items():
        share = len(found_in) / max(len(finished), 1)
        if share < vote_threshold:
            continue
        lags = pd.Series([int(r["lag"]) for r in found_in])
        strengths = pd.Series([float(r["strength"]) for r in found_in])
        rows.append({"cause": cause, "effect": effect, "lag": int(lags.mode().iloc[0]),
                     "strength": round(float(strengths.median()), 4),
                     "vote_share": round(share, 3)})

    edges = pd.DataFrame(rows, columns=["cause", "effect", "lag", "strength", "vote_share"])
    edges["abs_strength"] = edges["strength"].abs()
    edges = edges.sort_values(["vote_share", "abs_strength"], ascending=False, ignore_index=True)
    return CausalGraph(
        edges=edges,
        variables=list(variables),
        settings={**settings, "grouping": grouping, "vote_threshold": vote_threshold,
                  "groups": len(groups), "groups_finished": len(finished),
                  "recordings_used": len(tables)},
        link_type_counts=pd.Series(dtype=int),
    )


def _vote_groups(labels: list[str], grouping: str, seed: int) -> list[list[int]]:
    """Split recordings into voting groups, by their labels ("normal/..." or "fault/exp_X/run_Y")."""
    if grouping == "recording":
        return [[i] for i in range(len(labels))]
    if grouping != "replicates":
        raise ValueError(f"Unknown grouping '{grouping}'. Use 'recording' or 'replicates'.")

    by_scenario: dict[str, list[int]] = defaultdict(list)
    healthy: list[int] = []
    for i, label in enumerate(labels):
        if label.startswith("fault/"):
            by_scenario[label.split("/")[1]].append(i)
        else:
            healthy.append(i)

    # As many replicates as a typical scenario has runs. Scenarios with more
    # runs put two in some replicates; with fewer, they miss some.
    runs_per_scenario = sorted(len(runs) for runs in by_scenario.values())
    replicates = max(2, runs_per_scenario[len(runs_per_scenario) // 2])
    shuffle = random.Random(seed)
    groups: list[list[int]] = [[] for _ in range(replicates)]
    for scenario in sorted(by_scenario):
        runs = by_scenario[scenario][:]
        shuffle.shuffle(runs)
        for position, index in enumerate(runs):
            groups[position % replicates].append(index)
    shuffle.shuffle(healthy)
    for position, index in enumerate(healthy):
        groups[position % replicates].append(index)
    return groups


def _discover_group(job) -> pd.DataFrame | None:
    tables, variables, settings = job
    try:
        return discover(tables, variables, **settings).edges
    except Exception:
        # A group too small or too uniform for the test simply casts no vote.
        return None


def screen_same_moment_links(test, variables: list[str], partners: int,
                             significance: float) -> set[frozenset]:
    """
    Decide which pairs of signals may be linked within the same time step.

    Why this exists: on the full machine, a machining cycle starting or
    stopping switches dozens of signals in the same 2.5 s step, so pooled over
    22,000 rows almost every pair looks related - a signal moves together with
    about 71 others. PCMCI+ then tries to explain each of those ~2,900 pairs
    away with every combination of up to three neighbours (tigramite lifts our
    combination limit in that phase, for correctness). That is ~2.3 million
    tests, or 13-45 hours. Coolant has 8 neighbours per signal and ~800.

    The screen keeps, for each signal, only its `partners` strongest
    same-moment partners, judged by the plain dependence test with nothing
    conditioned on. A pair survives if either signal ranks the other among its
    strongest. Lagged links are not screened. This is ordinary screening for
    high-dimensional problems (sure independence screening, Fan and Lv 2008);
    the cost is that a true same-moment link outside both signals' strongest
    partners cannot be found. `partners` is fixed in advance, not tuned.
    """
    related: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            value, p_value = test.run_test(X=[(i, 0)], Y=[(j, 0)], Z=[])[:2]
            if p_value <= significance:
                related[i].append((abs(float(value)), j))
                related[j].append((abs(float(value)), i))

    kept: set[frozenset] = set()
    for i, found in related.items():
        for _, j in sorted(found, reverse=True)[:partners]:
            kept.add(frozenset((variables[i], variables[j])))
    return kept


def allowed_links(variables: list[str], max_lag: int, alarms: list[str],
                  same_moment_pairs: set[frozenset] | None = None) -> dict:
    """
    Tell PCMCI+ which links it may test.

    Two rules can apply. Alarms are effects only: no link may leave an alarm,
    though an alarm may depend on its own past. And when `same_moment_pairs` is
    given, a link within the same time step is allowed only between a screened
    pair (see `screen_same_moment_links`).

    Tigramite wants every allowed link listed. "-?>" means "may exist, and if
    it does it points this way"; "o?o" leaves both existence and direction to
    the data. A same-moment link is declared from both ends, so a signal that
    may switch an alarm on at once appears as "-?>" at the alarm and "<?-" at
    the signal.

    The cost of the alarm rule: interlocks - an alarm that switches a pump off
    or puts the machine into an error state - cannot be learned. That is 5 of
    18 expert links in hydraulics and 19 of 104 on the full machine. Allowing
    alarms to act one step later recovered none of them (at a 2.5 s grid
    interlocks act within the same step) and added false links on probe. In a
    plant, interlock actions are written down in the controller's alarm
    configuration, so they would be added as known links rather than learned.
    """
    alarm_set = set(alarms)
    assumptions = {}
    for j, effect in enumerate(variables):
        links = {}
        for i, cause in enumerate(variables):
            for lag in range(0, max_lag + 1):
                if i == j and lag == 0:
                    continue
                if lag > 0:
                    if cause not in alarm_set or i == j:
                        links[(i, -lag)] = "-?>"
                    continue
                if same_moment_pairs is not None and frozenset((cause, effect)) not in same_moment_pairs:
                    continue
                if cause in alarm_set and effect in alarm_set:
                    continue
                if effect in alarm_set:
                    links[(i, 0)] = "-?>"
                elif cause in alarm_set:
                    links[(i, 0)] = "<?-"
                else:
                    links[(i, 0)] = "o?o"
        assumptions[j] = links
    return assumptions


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
