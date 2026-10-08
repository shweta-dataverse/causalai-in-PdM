"""Module 7 — Intervention analysis (do-operations).

Answers the forward-looking maintenance question: *if we hold this variable in
its healthy range from now on, what happens to machine health?*

Formally, for candidate root cause V and target health readout H:

    effect(V) = E[ H | do(V = v_safe) ] - E[ H ]

The expectation under intervention is obtained by re-simulating the fitted SCM
with V clamped, propagating through the mechanisms in topological order. Because
the health indicator is a readout rather than a graph node, it is recomputed
from the simulated sensor trajectory - so the effect on health is derived, never
assumed.

Important scope statement for the thesis: on C-MAPSS these effects are
**model-based estimates**, not validated interventions. C-MAPSS contains no
interventional data, so nothing here can be checked against a ground truth. The
same code, run on causRCA, *can* be validated against its 100 labelled
do-interventions - that is the point of keeping the interface dataset-agnostic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import scm as scm_module


def safe_value(engine: pd.DataFrame, variable: str, onset_idx: int, baseline_frac: float = 0.3) -> float:
    """Healthy-range target for a variable: its mean over the engine's early life."""
    n_baseline = max(10, int(baseline_frac * onset_idx)) if onset_idx > 0 else 10
    return float(engine.iloc[:n_baseline][variable].mean())


def estimate_effect(
    model: scm_module.StructuralCausalModel,
    engine: pd.DataFrame,
    readout,
    variable: str,
    value: float,
    start_idx: int,
) -> dict:
    """Estimate the effect of do(variable = value) on the health indicator."""
    factual = scm_module.simulate(model, engine, noise=None)
    counter = scm_module.simulate(
        model, engine, noise=None,
        interventions={variable: value}, intervention_start=start_idx,
    )

    h_factual = readout.transform(factual)
    h_counter = readout.transform(counter)
    tail = slice(start_idx, len(engine))

    return {
        "variable": variable,
        "do_value": value,
        "health_factual": float(h_factual.iloc[tail].mean()),
        "health_intervened": float(h_counter.iloc[tail].mean()),
        "effect": float(h_counter.iloc[tail].mean() - h_factual.iloc[tail].mean()),
        "peak_health_factual": float(h_factual.iloc[tail].max()),
        "peak_health_intervened": float(h_counter.iloc[tail].max()),
        "trajectory_factual": h_factual,
        "trajectory_intervened": h_counter,
    }


def rank_interventions(
    model: scm_module.StructuralCausalModel,
    engine: pd.DataFrame,
    readout,
    candidates: list[str],
    onset_idx: int,
    lead_time: int = 20,
) -> pd.DataFrame:
    """Estimate and rank the effect of intervening on each candidate variable.

    `lead_time` sets how far before failure onset maintenance is assumed to act;
    intervening at the moment of failure would be operationally meaningless.
    """
    start_idx = max(model.tau_max, onset_idx - lead_time)
    rows = []
    for variable in candidates:
        if variable not in model.var_names:
            continue
        value = safe_value(engine, variable, onset_idx)
        result = estimate_effect(model, engine, readout, variable, value, start_idx)
        rows.append(
            {k: v for k, v in result.items() if not k.startswith("trajectory")}
        )
    if not rows:
        return pd.DataFrame(
            columns=["variable", "do_value", "health_factual", "health_intervened", "effect"]
        )
    out = pd.DataFrame(rows)
    # Most useful intervention = largest reduction in health-loss score.
    out["health_improvement"] = -out["effect"]
    return out.sort_values("health_improvement", ascending=False, ignore_index=True)


def dose_response(
    model: scm_module.StructuralCausalModel,
    engine: pd.DataFrame,
    readout,
    variable: str,
    onset_idx: int,
    n_points: int = 7,
    lead_time: int = 20,
) -> pd.DataFrame:
    """Sweep the intervention value to expose the dose-response relationship.

    A monotone curve supports a "keep this variable below X" recommendation; a
    flat curve means the variable is not a useful lever however hard it is
    pushed, which is itself an actionable finding.
    """
    start_idx = max(model.tau_max, onset_idx - lead_time)
    observed = engine[variable].to_numpy(dtype=float)
    grid = np.linspace(np.percentile(observed, 5), np.percentile(observed, 95), n_points)
    rows = []
    for value in grid:
        res = estimate_effect(model, engine, readout, variable, float(value), start_idx)
        rows.append(
            {"variable": variable, "do_value": float(value),
             "health_intervened": res["health_intervened"], "effect": res["effect"]}
        )
    return pd.DataFrame(rows)


def run(
    model: scm_module.StructuralCausalModel,
    engine: pd.DataFrame,
    readout,
    candidates: list[str],
    onset_idx: int,
    lead_time: int = 20,
) -> dict:
    """Execute Module 7 end to end."""
    ranking = rank_interventions(model, engine, readout, candidates, onset_idx, lead_time)
    best = ranking.iloc[0]["variable"] if not ranking.empty else None
    curve = (
        dose_response(model, engine, readout, best, onset_idx, lead_time=lead_time)
        if best else pd.DataFrame()
    )
    detail = (
        estimate_effect(
            model, engine, readout, best,
            safe_value(engine, best, onset_idx),
            max(model.tau_max, onset_idx - lead_time),
        )
        if best else {}
    )
    return {
        "ranking": ranking,
        "best_variable": best,
        "dose_response": curve,
        "best_detail": detail,
        "lead_time": lead_time,
        "start_idx": max(model.tau_max, onset_idx - lead_time),
    }
