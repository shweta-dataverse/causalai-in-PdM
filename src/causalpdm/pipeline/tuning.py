"""
Choosing the settings, by measuring instead of guessing.

Picking hyperparameters for causal discovery is normally awkward: there is no
correct answer to check against, so most work falls back on stability checks or
sensitivity sweeps.

causRCA removes that problem. It ships a causal graph drawn by the engineers
who built the machine, so we can try a setting, score the graph it produces
against the expert's, and keep whatever scores best. That is ordinary model
selection, and it is only possible because this dataset has ground truth.

A plant has no such graph, so `choose_significance_by_prediction` offers the
alternative a plant can actually use: keep the setting whose causal model best
predicts recordings it has not seen.

Settings worth searching, in order of how much they matter:

  resampling grid  how many seconds one row covers. This turned out to be the
                   strongest lever, because it decides how far back in time the
                   lag window reaches.
  maximum lag      how many steps back a cause may act.
  significance     how much evidence is needed to keep a link.
"""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import discovery, evaluate, preprocess, scm


@dataclass
class TuningResult:
    """Everything the search found, best first."""

    table: pd.DataFrame
    best: dict

    @property
    def best_settings(self) -> dict:
        """The winning settings, ready to pass straight into discovery."""
        return {
            "seconds_per_step": self.best["seconds_per_step"],
            "max_lag": int(self.best["max_lag"]),
            "significance": self.best["significance"],
        }


class _Timeout(Exception):
    pass


def _raise_timeout(signum, frame):
    raise _Timeout()


def search(
    dataset,
    grids: list[float],
    lags: list[int],
    significances: list[float],
    objective: str = "f1",
    independence_test: str = "discrete",
    min_rows: int = 20,
    seconds_per_run: int = 300,
    verbose: bool = True,
) -> TuningResult:
    """
    Try every combination of settings and keep the one that scores best.

    Each combination gets a time budget. Causal discovery cost grows sharply
    with the size of the lag window, and a combination that cannot finish in
    reasonable time is not a usable setting anyway, so timing out is a valid
    result rather than an error.
    """
    variables = dataset.expert_graph.variable_names
    expert_edges = dataset.expert_graph.edge_pairs
    rows = []

    signal.signal(signal.SIGALRM, _raise_timeout)

    for grid in grids:
        tables, _, _, _ = preprocess.build_tables(
            dataset.fault_runs, dataset.normal_recordings, variables, grid, min_rows)

        for lag in lags:
            for significance in significances:
                signal.alarm(seconds_per_run)
                started = time.time()
                try:
                    graph = discovery.discover(
                        tables, variables,
                        max_lag=lag,
                        significance=significance,
                        independence_test=independence_test,
                    )
                    signal.alarm(0)
                except _Timeout:
                    signal.alarm(0)
                    rows.append({
                        "seconds_per_step": grid, "max_lag": lag,
                        "significance": significance, "seconds_to_run": None,
                        "finished": False, "f1": 0.0, "precision": 0.0,
                        "recall": 0.0, "shd": None, "edges_found": None,
                    })
                    if verbose:
                        print(f"  grid {grid:>4} lag {lag} alpha {significance:<6} "
                              f"timed out after {seconds_per_run}s")
                    continue

                elapsed = time.time() - started
                score = evaluate.score_graph(graph.edge_pairs, expert_edges)
                rows.append({
                    "seconds_per_step": grid, "max_lag": lag,
                    "significance": significance,
                    "seconds_to_run": round(elapsed, 1), "finished": True,
                    **{k: score[k] for k in
                       ("edges_found", "correct", "precision", "recall", "f1", "shd")},
                })
                if verbose:
                    print(f"  grid {grid:>4} lag {lag} alpha {significance:<6} "
                          f"{elapsed:6.1f}s  F1 {score['f1']:.2f}  "
                          f"(found {score['edges_found']}, correct {score['correct']})")

    table = pd.DataFrame(rows)

    # Lower is better for structural hamming distance, higher for everything else.
    ascending = objective == "shd"
    ranked = table[table["finished"]].sort_values(
        objective, ascending=ascending, ignore_index=True)

    if ranked.empty:
        raise RuntimeError("Every setting timed out. Try smaller lags or a coarser grid.")

    return TuningResult(table=table.sort_values(objective, ascending=ascending,
                                                ignore_index=True),
                        best=ranked.iloc[0].to_dict())


def explain_result(result: TuningResult) -> str:
    """A short plain-English summary of what the search decided and why."""
    best = result.best
    finished = result.table[result.table["finished"]]
    return (
        f"Best settings: a {best['seconds_per_step']} second grid with a maximum lag of "
        f"{int(best['max_lag'])} and significance {best['significance']}. "
        f"This recovered {int(best['correct'])} of the expert graph's links "
        f"(F1 {best['f1']:.2f}) in {best['seconds_to_run']} seconds. "
        f"{len(finished)} of {len(result.table)} settings finished within their time budget."
    )


def choose_significance_by_prediction(
    tables: list[pd.DataFrame],
    variables: list[str],
    candidates: list[float],
    max_lag: int,
    folds: int = 5,
    **discover_options,
) -> pd.DataFrame:
    """
    Pick the significance level without the expert graph.

    A plant has no expert graph to score against, so `search` above is not
    available there. What a plant does have is its own recordings. For each
    candidate level we learn the graph and the equations from most recordings
    and ask how well they predict the recordings left out. A level that keeps
    too few links misses real influences; one that keeps too many fits noise.
    Both predict unseen recordings worse.

    Returns, per level, the average log-likelihood per value predicted on the
    held-out recordings, averaged over `folds` splits. Higher is better.
    """
    rows = []
    for significance in candidates:
        fold_scores = []
        for fold in range(folds):
            held_out = [t for i, t in enumerate(tables) if i % folds == fold]
            training = [t for i, t in enumerate(tables) if i % folds != fold]
            graph = discovery.discover(training, variables, max_lag=max_lag,
                                       significance=significance, **discover_options)
            model = scm.fit(training, variables, graph.edges, max_lag=max_lag)
            fold_scores.append(_held_out_log_likelihood(model, training, held_out))
        rows.append({"significance": significance,
                     "held_out_log_likelihood": round(float(np.mean(fold_scores)), 5),
                     "spread_over_folds": round(float(np.std(fold_scores)), 5)})
    return pd.DataFrame(rows)


def _held_out_log_likelihood(model, training: list[pd.DataFrame],
                             held_out: list[pd.DataFrame]) -> float:
    """Average log-probability the fitted equations give to what really happened."""
    floor = 1e-6
    total, count = 0.0, 0
    for variable in model.variables:
        mechanism = model.mechanisms.get(variable)
        if mechanism is None:
            # Never changed in training: the model says it always holds that value.
            constant = training[0][variable].iloc[0]
            for table in held_out:
                values = table[variable].to_numpy()[model.max_lag:]
                total += float(np.sum(np.where(values == constant, 0.0, np.log(floor))))
                count += len(values)
            continue
        inputs, outputs = scm._training_rows(held_out, variable, mechanism.causes, model.max_lag)
        if len(outputs) == 0:
            continue
        probabilities = mechanism.honest_model.predict_proba(inputs)
        classes = list(mechanism.honest_model.classes_)
        seen = np.array([classes.index(v) if v in classes else -1 for v in outputs])
        chance = np.where(seen >= 0, probabilities[np.arange(len(outputs)), np.maximum(seen, 0)], floor)
        total += float(np.sum(np.log(np.maximum(chance, floor))))
        count += len(outputs)
    return total / max(count, 1)
