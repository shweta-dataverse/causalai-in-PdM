"""Module 8 — Counterfactual XAI (abduction → action → prediction).

Interventions (Module 7) answer a population question: what happens *in general*
if we clamp a variable. Counterfactuals answer the retrospective, engine-specific
question the maintenance engineer actually asks:

    "This engine failed. Would it have failed had sensor X stayed in range?"

That requires Pearl's three-step procedure, executed literally here:

  **Abduction**  - recover the exogenous noise e(t) of *this* engine from its
                   observed trajectory (`scm.recover_noise`). This is what makes
                   the answer specific to this unit rather than to the fleet.
  **Action**     - modify the model with do(X = x') from a chosen cycle onward.
  **Prediction** - re-simulate holding the recovered noise fixed, so the twin
                   world differs from the factual one *only* through the
                   intervention.

Explanations are scored with the standard counterfactual-XAI criteria: validity
(did the outcome actually change), proximity (how large a change was required),
sparsity (how many variables had to move) and plausibility (does the required
value lie inside the engine's own observed operating range).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import scm as scm_module


def generate(
    model: scm_module.StructuralCausalModel,
    engine: pd.DataFrame,
    readout,
    variable: str,
    value: float,
    start_idx: int,
    noise: pd.DataFrame | None = None,
) -> dict:
    """Produce one counterfactual trajectory via abduction/action/prediction."""
    if noise is None:
        noise = scm_module.recover_noise(model, engine)          # abduction

    factual = scm_module.simulate(model, engine, noise=noise)     # twin-world factual
    counterfactual = scm_module.simulate(                        # action + prediction
        model, engine, noise=noise,
        interventions={variable: value}, intervention_start=start_idx,
    )

    h_fact = readout.transform(factual)
    h_cf = readout.transform(counterfactual)
    tail = slice(start_idx, len(engine))

    return {
        "variable": variable,
        "cf_value": float(value),
        "start_idx": int(start_idx),
        "noise": noise,
        "factual_trajectory": factual,
        "counterfactual_trajectory": counterfactual,
        "health_factual": h_fact,
        "health_counterfactual": h_cf,
        "peak_health_factual": float(h_fact.iloc[tail].max()),
        "peak_health_counterfactual": float(h_cf.iloc[tail].max()),
        "mean_health_factual": float(h_fact.iloc[tail].mean()),
        "mean_health_counterfactual": float(h_cf.iloc[tail].mean()),
    }


def score(result: dict, engine: pd.DataFrame, threshold: float) -> dict:
    """Score a counterfactual with the standard CF-XAI criteria."""
    variable = result["variable"]
    observed = engine[variable].to_numpy(dtype=float)
    factual_level = float(observed[result["start_idx"] :].mean())
    delta = abs(result["cf_value"] - factual_level)
    spread = float(observed.std()) or 1.0

    return {
        "valid": bool(result["peak_health_counterfactual"] < threshold),
        "proximity_abs": delta,
        "proximity_sd": delta / spread,
        "sparsity": 1,                       # single-variable counterfactual
        "plausible": bool(observed.min() <= result["cf_value"] <= observed.max()),
        "health_reduction": float(
            result["peak_health_factual"] - result["peak_health_counterfactual"]
        ),
    }


def minimal_change(
    model: scm_module.StructuralCausalModel,
    engine: pd.DataFrame,
    readout,
    variable: str,
    start_idx: int,
    threshold: float,
    n_steps: int = 9,
    noise: pd.DataFrame | None = None,
) -> dict:
    """Search for the smallest change to `variable` that avoids the failure.

    Minimality matters for actionability: an explanation demanding a physically
    extreme value is not a maintenance instruction. The search sweeps the
    engine's own observed range, so any answer found is achievable by
    construction.
    """
    if noise is None:
        noise = scm_module.recover_noise(model, engine)

    observed = engine[variable].to_numpy(dtype=float)
    factual_level = float(observed[start_idx:].mean())
    grid = np.linspace(np.percentile(observed, 2), np.percentile(observed, 98), n_steps)
    grid = sorted(grid, key=lambda v: abs(v - factual_level))   # nearest-first

    attempts = []
    best = None
    for value in grid:
        result = generate(model, engine, readout, variable, float(value), start_idx, noise=noise)
        metrics = score(result, engine, threshold)
        attempts.append({"do_value": float(value), **{k: v for k, v in metrics.items()}})
        if metrics["valid"] and best is None:
            best = {"result": result, "metrics": metrics}
            break

    return {
        "variable": variable,
        "attempts": pd.DataFrame(attempts),
        "found": best is not None,
        "best": best,
        "factual_level": factual_level,
        "threshold": threshold,
    }


def explain(
    model: scm_module.StructuralCausalModel,
    engine: pd.DataFrame,
    readout,
    variable: str,
    start_idx: int,
    threshold: float,
    unit: int | None = None,
) -> dict:
    """Build a natural-language counterfactual explanation plus its metrics."""
    search = minimal_change(model, engine, readout, variable, start_idx, threshold)
    cycle = int(engine.loc[start_idx, "cycle"]) if "cycle" in engine.columns else start_idx
    engine_label = f"Engine {unit}" if unit is not None else "This engine"

    if search["found"]:
        best = search["best"]
        direction = "below" if best["result"]["cf_value"] < search["factual_level"] else "above"
        text = (
            f"{engine_label} reached a degraded state. Had {variable} been held "
            f"{direction} {best['result']['cf_value']:.2f} from cycle {cycle} onward "
            f"(a shift of {best['metrics']['proximity_sd']:.1f} standard deviations from its "
            f"actual level of {search['factual_level']:.2f}), the health indicator would have "
            f"peaked at {best['result']['peak_health_counterfactual']:.2f} instead of "
            f"{best['result']['peak_health_factual']:.2f}, staying below the "
            f"{threshold:.2f} failure threshold."
        )
    else:
        text = (
            f"{engine_label}: no achievable value of {variable} within its observed operating "
            f"range prevents the degraded state. This variable is a genuine root cause "
            f"indicator but not a sufficient lever on its own - a multi-variable intervention "
            f"or component replacement is required."
        )

    return {**search, "explanation": text, "cycle": cycle}


def run(
    model: scm_module.StructuralCausalModel,
    engine: pd.DataFrame,
    readout,
    variable: str,
    start_idx: int,
    threshold: float,
    unit: int | None = None,
) -> dict:
    """Execute Module 8 end to end."""
    return explain(model, engine, readout, variable, start_idx, threshold, unit=unit)
