"""
Scoring what the pipeline produces.

causRCA is unusual: it ships an expert-built causal graph and labelled fault
causes, so for once we can measure whether our answers are right rather than
only whether they look plausible. This module holds every metric used.

Two families of metric:

  Graph metrics  - did causal discovery recover the engineers' graph?
  Ranking metrics - when a fault happened, did we name the right cause, and how
                    high up our list was it?
"""

from __future__ import annotations

import pandas as pd

Edge = tuple[str, str]


# =============================================================================
# Did we recover the right causal graph?
# =============================================================================

def score_graph(found: set[Edge], expert: set[Edge]) -> dict:
    """
    Compare a discovered graph against the expert graph.

    precision  of the links we claimed, how many are real
    recall     of the real links, how many we found
    f1         the balance of the two
    shd        structural hamming distance: how many single edge edits would
               turn our graph into the expert's. Lower is better, 0 is perfect.
    """
    true_positives = found & expert
    false_positives = found - expert
    missed = expert - found

    precision = len(true_positives) / len(found) if found else 0.0
    recall = len(true_positives) / len(expert) if expert else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "edges_found": len(found),
        "edges_expected": len(expert),
        "correct": len(true_positives),
        "wrong": len(false_positives),
        "missed": len(missed),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "shd": len(false_positives) + len(missed),
    }


def score_graph_ignoring_direction(found: set[Edge], expert: set[Edge]) -> dict:
    """
    Same comparison, but counting a link as correct even if we pointed it the
    wrong way.

    Reported alongside the directed score because the gap between the two tells
    you whether the method is failing to find links at all, or finding them but
    getting the direction wrong. Those are different problems with different
    fixes.
    """
    undirected = lambda edges: {tuple(sorted(e)) for e in edges}
    return score_graph(undirected(found), undirected(expert))


def compare_edges(found: set[Edge], expert: set[Edge]) -> pd.DataFrame:
    """Edge-by-edge verdict, for showing in a report or on the dashboard."""
    rows = []
    for cause, effect in sorted(expert):
        rows.append({"cause": cause, "effect": effect,
                     "verdict": "found" if (cause, effect) in found else "missed"})
    for cause, effect in sorted(found - expert):
        rows.append({"cause": cause, "effect": effect, "verdict": "false alarm"})
    return pd.DataFrame(rows)


# =============================================================================
# Did we name the right root cause?
# =============================================================================

def precision_at_k(ranked: list[str], true_causes: list[str], k: int = 3) -> float:
    """Share of our top k guesses that were genuinely causes of the fault."""
    if k <= 0:
        return 0.0
    top = ranked[:k]
    return sum(1 for name in top if name in true_causes) / k


def recall_at_k(ranked: list[str], true_causes: list[str], k: int = 3) -> float:
    """Share of the real causes that made it into our top k."""
    if not true_causes:
        return 0.0
    top = set(ranked[:k])
    return sum(1 for name in true_causes if name in top) / len(true_causes)


def average_precision_at_k(ranked: list[str], true_causes: list[str], k: int = 3) -> float:
    """
    Average precision over the top k.

    Rewards putting real causes near the top rather than merely somewhere in
    the list. This is the metric the causRCA benchmark reports (as MAP@3 once
    averaged over all faults).
    """
    if not true_causes:
        return 0.0
    hits = 0
    running = 0.0
    for position, name in enumerate(ranked[:k], start=1):
        if name in true_causes:
            hits += 1
            running += hits / position
    return running / min(len(true_causes), k)


def reciprocal_rank(ranked: list[str], true_causes: list[str]) -> float:
    """1 divided by the position of the first correct answer. 0 if never found."""
    for position, name in enumerate(ranked, start=1):
        if name in true_causes:
            return 1.0 / position
    return 0.0


def score_rankings(results: list[tuple[list[str], list[str]]], k: int = 3) -> dict:
    """
    Score root cause analysis across many faults.

    `results` is a list of (our ranked guesses, the true causes) pairs, one per
    fault. Returns the averages, which is what gets compared against published
    numbers.
    """
    if not results:
        return {}
    return {
        "faults_scored": len(results),
        f"precision@{k}": round(
            sum(precision_at_k(r, t, k) for r, t in results) / len(results), 3),
        f"recall@{k}": round(
            sum(recall_at_k(r, t, k) for r, t in results) / len(results), 3),
        f"map@{k}": round(
            sum(average_precision_at_k(r, t, k) for r, t in results) / len(results), 3),
        "mean_reciprocal_rank": round(
            sum(reciprocal_rank(r, t) for r, t in results) / len(results), 3),
        "found_in_top_1": round(
            sum(1 for r, t in results if r and r[0] in t) / len(results), 3),
    }
