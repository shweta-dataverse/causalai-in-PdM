"""
Running the whole pipeline end to end.

This is the orchestrator. Every stage lives in its own module; this file just
calls them in order, passes results along, and keeps a record of what happened
so the dashboard and the reports have something to read.

    load data -> build tables -> discover graph -> score it
              -> fit equations -> find root causes -> check what-if answers

Results are cached, because causal discovery is the slow step and nothing else
in the pipeline takes more than a few seconds.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import data, discovery, evaluate, explain, paths, preprocess, rca, scm


@dataclass
class PipelineResult:
    """Everything the pipeline produced, in one place."""

    subsystem: str
    dataset_summary: dict
    settings_used: dict
    tables: list[pd.DataFrame]
    table_labels: list[str]
    healthy_values: dict
    causal_graph: Any
    graph_score: dict
    graph_score_undirected: dict
    edge_comparison: pd.DataFrame
    causal_model: Any
    root_cause_results: list = field(default_factory=list)
    root_cause_score: dict = field(default_factory=dict)
    intervention_checks: list = field(default_factory=list)
    intervention_score: dict = field(default_factory=dict)
    counterfactuals: list = field(default_factory=list)
    counterfactual_score: dict = field(default_factory=dict)
    seconds_taken: float = 0.0

    def headline(self) -> dict:
        """The numbers worth putting at the top of a report."""
        return {
            "subsystem": self.subsystem,
            "variables": self.dataset_summary["variables"],
            "recordings_used": len(self.tables),
            "expert_edges": self.graph_score["edges_expected"],
            "edges_found": self.graph_score["edges_found"],
            "edges_correct": self.graph_score["correct"],
            "graph_f1": self.graph_score["f1"],
            "root_cause_map@3": self.root_cause_score.get("map@3"),
            "root_cause_top1": self.root_cause_score.get("found_in_top_1"),
            "intervention_f1": self.intervention_score.get("mean_f1"),
            "intervention_scenarios": self.intervention_score.get("scenarios_checked"),
            "counterfactual_success": self.counterfactual_score.get("suspects_success_rate"),
            "counterfactual_upper_bound": self.counterfactual_score.get("known_causes_success_rate"),
            "seconds_taken": round(self.seconds_taken, 1),
        }


def _fingerprint(tables: list[pd.DataFrame]) -> str:
    """A short hash of the table contents, for naming cache files."""
    digest = hashlib.sha1()
    for table in tables:
        digest.update(pd.util.hash_pandas_object(table, index=True).to_numpy().tobytes())
    return digest.hexdigest()[:10]


def _say(step: str, message: str, started: float) -> None:
    print(f"[{time.time() - started:6.1f}s] {step:<18} {message}", flush=True)


def run(subsystem: str = "coolant",
        normal_recordings: int | None = 60,
        seconds_per_step: float = 2.5,
        max_lag: int = 3,
        significance: float | None = 0.001,
        independence_test: str = "discrete",
        multiple_test_correction: str = "fdr_bh",
        min_rows: int = 20,
        top_k: int = 3,
        alarms_only_as_effects: bool = True,
        hold_out_scenario: str | None = None,
        use_cache: bool = True,
        verbose: bool = True) -> PipelineResult:
    """
    Run every stage and return the collected results.

    `hold_out_scenario` answers the question a plant actually faces: can the
    system handle a fault type it has never seen? That scenario's recordings
    are removed before the graph and the equations are learned, and only that
    scenario is then diagnosed and replayed.
    """
    started = time.time()
    paths.make_output_folders()

    settings_used = {
        "subsystem": subsystem, "normal_recordings": normal_recordings,
        "seconds_per_step": seconds_per_step, "max_lag": max_lag,
        "significance": significance, "independence_test": independence_test,
        "multiple_test_correction": multiple_test_correction, "min_rows": min_rows,
        "alarms_only_as_effects": alarms_only_as_effects,
        "hold_out_scenario": hold_out_scenario,
    }

    # --- load ------------------------------------------------------------
    dataset = data.load(subsystem, normal_recordings=normal_recordings)
    variables = dataset.expert_graph.variable_names
    alarm_names = dataset.expert_graph.alarm_names
    if verbose:
        _say("1 load", f"{dataset.summary()['fault_runs']} fault runs, "
             f"{len(dataset.normal_recordings)} healthy recordings, "
             f"{len(variables)} variables", started)

    # --- tables ----------------------------------------------------------
    tables, labels, healthy, cut_points = preprocess.build_tables(
        dataset.fault_runs, dataset.normal_recordings, variables,
        seconds_per_step, min_rows)
    fault_tables = {label.split("/", 1)[1]: table
                    for label, table in zip(labels, tables) if label.startswith("fault")}
    test_runs = dataset.fault_runs
    if hold_out_scenario is not None:
        test_runs = [run for run in dataset.fault_runs if run.experiment == hold_out_scenario]
        if not test_runs:
            raise ValueError(f"No scenario '{hold_out_scenario}' in {subsystem}.")
        kept = [i for i, label in enumerate(labels)
                if not label.startswith(f"fault/{hold_out_scenario}/")]
        tables = [tables[i] for i in kept]
        labels = [labels[i] for i in kept]
    if verbose:
        _say("2 build tables", f"{len(tables)} recordings on a "
             f"{seconds_per_step}s grid", started)

    # --- discover --------------------------------------------------------
    # The cache name includes a fingerprint of the tables themselves, so a
    # change to preprocessing can never silently reuse a graph learned from
    # the old tables.
    cache_file = paths.CACHE_DIR / (
        f"graph_{subsystem}_{seconds_per_step}_{max_lag}_{significance}_"
        f"{multiple_test_correction}_{alarms_only_as_effects}_{_fingerprint(tables)}.pkl")
    if use_cache and cache_file.exists():
        graph = pickle.loads(cache_file.read_bytes())
        if verbose:
            _say("3 discover", f"{len(graph.edges)} links (reused from cache)", started)
    else:
        graph = discovery.discover(
            tables, variables, max_lag=max_lag, significance=significance,
            independence_test=independence_test,
            multiple_test_correction=multiple_test_correction,
            alarms=alarm_names if alarms_only_as_effects else None)
        cache_file.write_bytes(pickle.dumps(graph))
        if verbose:
            _say("3 discover", f"{len(graph.edges)} causal links found", started)

    # --- score the graph -------------------------------------------------
    expert_edges = dataset.expert_graph.edge_pairs
    graph_score = evaluate.score_graph(graph.edge_pairs, expert_edges)
    undirected_score = evaluate.score_graph_ignoring_direction(graph.edge_pairs, expert_edges)
    comparison = evaluate.compare_edges(graph.edge_pairs, expert_edges)
    if verbose:
        _say("4 score graph", f"{graph_score['correct']} of "
             f"{graph_score['edges_expected']} expert links recovered, "
             f"F1 {graph_score['f1']}", started)

    # --- fit equations ---------------------------------------------------
    causal_model = scm.fit(tables, variables, graph.edges, max_lag=max_lag)
    if verbose:
        _say("5 fit equations", f"{len(causal_model.mechanisms)} equations, "
             f"average accuracy "
             f"{causal_model.summary()['accuracy'].mean():.2f}", started)

    # --- root cause analysis ---------------------------------------------
    rca_results, scored_pairs = [], []
    for run_info in test_runs:
        table = fault_tables.get(run_info.name)
        if table is None:
            continue
        result = rca.analyse(causal_model, table, alarm_names, run_info.name)
        rca_results.append((run_info, result))
        scored_pairs.append((result.ranked_variables, run_info.caused_by))
    rca_score = evaluate.score_rankings(scored_pairs, k=top_k)
    if verbose:
        _say("6 root cause", f"{len(rca_results)} faults analysed, "
             f"MAP@{top_k} {rca_score.get(f'map@{top_k}')}, "
             f"top-1 hit rate {rca_score.get('found_in_top_1')}", started)

    # --- check what-if answers against the real injected faults -----------
    # One check per fault scenario: runs of the same scenario force the same
    # thing, so replaying each run would only repeat the same question.
    healthy_table = next((t for l, t in zip(labels, tables) if l.startswith("normal")), None)
    checks = []
    if healthy_table is not None:
        runs_per_scenario = Counter(run.experiment for run in dataset.fault_runs)
        seen = set()
        for run_info in test_runs:
            if run_info.experiment in seen:
                continue
            seen.add(run_info.experiment)
            forced = {name: preprocess.forced_state(run_info, name, cut_points)
                      for name in run_info.caused_by}
            check = explain.validate_intervention(
                causal_model, healthy_table, forced,
                run_info.expected_alarms, alarm_names, from_step=max_lag + 2)
            checks.append({"scenario": run_info.experiment,
                           "runs": runs_per_scenario[run_info.experiment], **check})
    intervention_score = {
        "scenarios_checked": len(checks),
        "mean_precision": round(sum(c["precision"] for c in checks) / len(checks), 3) if checks else 0,
        "mean_recall": round(sum(c["recall"] for c in checks) / len(checks), 3) if checks else 0,
        "mean_f1": round(sum(c["f1"] for c in checks) / len(checks), 3) if checks else 0,
    }
    if verbose:
        _say("7 check what-ifs", f"{len(checks)} fault scenarios replayed, "
             f"alarm prediction F1 {intervention_score['mean_f1']}", started)

    # --- counterfactual for each fault -----------------------------------
    # Most faults here were caused by more than one variable at once, so we ask
    # two questions: what does fixing the single most suspicious variable
    # achieve, and what does fixing all our top suspects together achieve.
    counterfactuals = []
    for run_info, result in rca_results:
        table = fault_tables.get(run_info.name)
        if table is None or not result.ranked_variables:
            continue
        start_at = max(result.fault_window[0] - 2, max_lag)

        top_suspect = result.ranked_variables[0]
        fix_one = explain.ask_what_if(
            causal_model, table, {top_suspect: healthy.get(top_suspect, 1.0)},
            alarm_names, from_step=start_at)

        suspects = result.ranked_variables[:top_k]
        fix_several = explain.ask_what_if(
            causal_model, table,
            {name: healthy.get(name, 1.0) for name in suspects},
            alarm_names, from_step=start_at)

        # Oracle check: fix the causes we know were really forced. This tells
        # us whether the what-if machinery itself is sound, separately from
        # whether root cause analysis picked the right variables. When this
        # fails it is almost always because causal discovery missed a link, so
        # it is also how we trace an error back to the stage that caused it.
        fix_true_causes = explain.ask_what_if(
            causal_model, table,
            {name: healthy.get(name, 1.0) for name in run_info.caused_by},
            alarm_names, from_step=start_at)

        counterfactuals.append((run_info, fix_one, fix_several, fix_true_causes))

    # The headline is what the pipeline achieves on its own: fixing the suspects
    # that root cause analysis named. Fixing the known true causes is reported
    # only as an upper bound - it uses the answer, which a real plant never has.
    examined = len(counterfactuals)
    rate = lambda count: round(count / examined, 3) if examined else 0
    helped = sum(1 for _, one, _, _ in counterfactuals if one.prevented_something)
    suspects = sum(1 for _, _, several, _ in counterfactuals if several.prevented_everything)
    oracle = sum(1 for _, _, _, truth in counterfactuals if truth.prevented_everything)
    counterfactual_score = {
        "faults_examined": examined,
        "top_suspect_silenced_an_alarm": helped,
        "suspects_prevented_fault": suspects,
        "suspects_success_rate": rate(suspects),
        "known_causes_prevented_fault": oracle,
        "known_causes_success_rate": rate(oracle),
    }
    if verbose:
        _say("8 counterfactuals", f"fixing our top {top_k} suspects prevented "
             f"{suspects} of {examined} faults entirely; upper bound with the "
             f"known true causes {oracle} of {examined}", started)

    return PipelineResult(
        subsystem=subsystem, dataset_summary=dataset.summary(),
        settings_used=settings_used, tables=tables, table_labels=labels,
        healthy_values=healthy, causal_graph=graph, graph_score=graph_score,
        graph_score_undirected=undirected_score, edge_comparison=comparison,
        causal_model=causal_model, root_cause_results=rca_results,
        root_cause_score=rca_score, intervention_checks=checks,
        intervention_score=intervention_score, counterfactuals=counterfactuals,
        counterfactual_score=counterfactual_score,
        seconds_taken=time.time() - started)


def save(result: PipelineResult) -> None:
    """
    Write the results where the dashboard and reports can find them.

    Each subsystem gets its own folder, so running one never overwrites
    another's results.
    """
    paths.make_output_folders()
    folder = paths.RESULTS_DIR / result.subsystem
    folder.mkdir(exist_ok=True)
    (folder / "headline.json").write_text(json.dumps(result.headline(), indent=2))
    (folder / "scores.json").write_text(json.dumps({
        "graph": result.graph_score,
        "graph_ignoring_direction": result.graph_score_undirected,
        "root_cause": result.root_cause_score,
        "intervention": result.intervention_score,
        "counterfactual": result.counterfactual_score,
        "settings": result.settings_used,
    }, indent=2))
    result.edge_comparison.to_csv(folder / "edge_comparison.csv", index=False)
    result.causal_graph.edges.to_csv(folder / "causal_graph.csv", index=False)
    result.causal_model.summary().to_csv(folder / "equations.csv", index=False)
    pd.DataFrame(result.intervention_checks).to_csv(
        folder / "intervention_by_scenario.csv", index=False)

    name = ("pipeline_result.pkl" if result.subsystem == "coolant"
            else f"pipeline_result_{result.subsystem}.pkl")
    (paths.CACHE_DIR / name).write_bytes(pickle.dumps(result))
