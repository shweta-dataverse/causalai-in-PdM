#!/usr/bin/env python3
"""
Runs the whole Actionable RCA pipeline in one go and builds the dashboard.

    python scripts/run_pipeline.py

Want to change something? Edit pipeline_config.py - not this file.

The pipeline has 9 steps. Each one is a separate module in
src/causalpdm/framework/, and each step below just calls it and prints what
happened. The notebooks in notebooks/framework/ walk through the same 9 steps
with much more explanation and plots.
"""

import sys
import time
import warnings
from pathlib import Path

# Make sure Python can find our code and our config file.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore")

import pipeline_config as cfg
from causalpdm.framework import (
    artifacts,
    counterfactual,
    dashboard,
    discovery,
    features,
    intervention,
    preprocessing,
    rca,
    scm,
    validation,
)

START_TIME = time.time()


def say(step, message):
    """Print a progress line with the elapsed time, so slow steps are obvious."""
    elapsed = time.time() - START_TIME
    print(f"[{elapsed:6.1f}s] {step:<18} {message}", flush=True)


def run_or_load(name, compute):
    """
    Run a step, or reuse its saved result if we already computed it before.

    Causal discovery is slow, so we save each step's output to artifacts/.
    On the next run we load it back instead of recomputing.
    Set USE_CACHE = False in the config to always recompute.
    """
    if cfg.USE_CACHE and artifacts.exists(name):
        return artifacts.load(name)
    result = compute()
    artifacts.save(name, result)
    return result


def main():
    print(f"Running the Actionable RCA pipeline on C-MAPSS {cfg.DATASET}\n")

    # ------------------------------------------------------------------
    # STEP 1 - Clean up the raw data
    # ------------------------------------------------------------------
    # Drops sensors that never change, normalises each engine against itself,
    # and works out how many cycles each engine has left (RUL).
    prep = run_or_load(
        "prep",
        lambda: preprocessing.run(cfg.DATASET, detrend=cfg.REMOVE_TREND_FOR_DISCOVERY),
    )
    say("1 preprocessing", f"{prep['n_rows']:,} rows from {prep['n_units']} engines")

    # Decide which sensors we treat as causal variables.
    # The config either lists them explicitly, or says None meaning "use all".
    if cfg.CAUSAL_VARIABLES:
        variables = [v for v in cfg.CAUSAL_VARIABLES if v in prep["feature_cols"]]
    else:
        variables = prep["feature_cols"]
    say("", f"using {len(variables)} sensors as causal variables")

    # ------------------------------------------------------------------
    # STEP 2 - Build features and a health score
    # ------------------------------------------------------------------
    # Adds rolling averages, trends and derivatives, then builds a single
    # "health loss" score from all the sensors. That score is what tells us
    # when an engine has entered a degraded state.
    feat = run_or_load("feat", lambda: features.run(prep["normalised"], variables))
    health_variance = feat["health_indicator_meta"]["explained_variance_ratio"]
    say("2 features", f"health score captures {health_variance:.0%} of sensor variation")
    say("", f"failure threshold set at {feat['failure_threshold']:.2f}")

    # ------------------------------------------------------------------
    # STEP 3 - Learn the causal graph with PCMCI+
    # ------------------------------------------------------------------
    # This is the slow step. We use the detrended data here, because the
    # degradation trend would otherwise create fake links between everything.
    engines = sorted(prep["detrended"]["unit"].unique())[: cfg.ENGINES_FOR_DISCOVERY]
    discovery_data = prep["detrended"][prep["detrended"]["unit"].isin(engines)]

    disc = run_or_load(
        "disc",
        lambda: discovery.run(
            discovery_data,
            variables,
            tau_max=cfg.MAX_LAG,
            pc_alpha=cfg.SIGNIFICANCE,
            max_conds_dim=cfg.MAX_CONDITION_SET_SIZE,
            max_combinations=cfg.MAX_COMBINATIONS,
        ),
    )
    say("3 discovery", f"found {disc['n_directed_edges']} causal links "
                       f"across {disc['n_realisations']} engines")

    # ------------------------------------------------------------------
    # STEP 4 - Check whether the graph can be trusted
    # ------------------------------------------------------------------
    # C-MAPSS has no "correct" graph to compare against, so we cannot check
    # correctness. What we CAN check is robustness: re-run discovery on random
    # subsets of engines and keep only the links that keep showing up.
    if not cfg.RUN_VALIDATION:
        valid = None
        say("4 validation", "skipped (RUN_VALIDATION is False in the config)")
    else:
        valid = run_or_load(
            "valid",
            lambda: validation.run(
                discovery_data,
                variables,
                disc["edges"],
                n_boot=cfg.BOOTSTRAP_RUNS,
                stability_threshold=cfg.STABILITY_THRESHOLD,
                tau_max=cfg.MAX_LAG,
                pc_alpha=cfg.SIGNIFICANCE,
                max_conds_dim=cfg.MAX_CONDITION_SET_SIZE,
                max_combinations=cfg.MAX_COMBINATIONS,
            ),
        )
        say("4 validation", f"{valid['n_validated']} of {len(disc['edges'])} links "
                            f"were stable enough to keep")

    # Use the validated links if we have them, otherwise fall back to all of them.
    if valid is not None and not valid["validated_edges"].empty:
        edges = valid["validated_edges"]
    else:
        edges = disc["edges"]

    # ------------------------------------------------------------------
    # STEP 5 - Fit the structural causal model
    # ------------------------------------------------------------------
    # The graph tells us WHICH sensors affect which. This step learns HOW much,
    # by fitting one equation per sensor. Without this we cannot simulate
    # interventions or counterfactuals at all.
    # Note: we fit on the trended data, because that is the same space the
    # health score lives in.
    #
    # This is deliberately NOT cached. It depends on which edges survived
    # validation, and it only takes a fraction of a second - caching it would
    # risk silently reusing a model built from a different graph.
    scm_result = scm.run(prep["normalised"], variables, edges, tau_max=cfg.MAX_LAG)
    say("5 scm fitting", f"fitted {scm_result['n_mechanisms']} equations, "
                         f"average fit R2 = {scm_result['mean_r2']:.2f}")

    # ------------------------------------------------------------------
    # STEP 6 - Find the root causes
    # ------------------------------------------------------------------
    # At failure almost every sensor looks abnormal. The causal graph is what
    # lets us tell the difference between a cause and a knock-on symptom.
    rca_result = rca.run(
        feat["featured"], variables, edges, unit=cfg.ENGINE_TO_ANALYSE
    )
    say("6 root cause", f"engine {rca_result['unit']} degraded at cycle "
                        f"{rca_result['onset_cycle']} of {rca_result['total_cycles']}")
    say("", f"{len(rca_result['root_causes'])} root causes, "
            f"{len(rca_result['symptoms'])} downstream symptoms")

    # ------------------------------------------------------------------
    # STEP 7 - Test interventions
    # ------------------------------------------------------------------
    # For each root cause we ask: if we had held this sensor in its healthy
    # range, what would have happened to engine health?
    candidates = rca_result["root_causes"]["variable"].tolist()
    if not candidates:
        candidates = rca_result["ranking"]["variable"].tolist()
    candidates = candidates[: cfg.INTERVENTION_CANDIDATES]

    interv = intervention.run(
        scm_result["scm"],
        rca_result["engine"],
        feat["health_readout"],
        candidates,
        rca_result["onset_idx"],
        lead_time=cfg.MAINTENANCE_LEAD_TIME,
    )
    say("7 intervention", f"most useful action: control {interv['best_variable']}")

    # ------------------------------------------------------------------
    # STEP 8 - Generate the counterfactual explanation
    # ------------------------------------------------------------------
    # The "what would have prevented this failure?" answer, for this specific
    # engine. Works by recovering this engine's own noise, applying the change,
    # and re-running the model with that noise held fixed.
    cf = counterfactual.run(
        scm_result["scm"],
        rca_result["engine"],
        feat["health_readout"],
        interv["best_variable"],
        interv["start_idx"],
        feat["failure_threshold"],
        unit=rca_result["unit"],
    )
    if cf["found"]:
        say("8 counterfactual", "found a change that would have avoided failure")
    else:
        say("8 counterfactual", "no single sensor change was enough to avoid failure")

    # ------------------------------------------------------------------
    # STEP 9 - Build the dashboard
    # ------------------------------------------------------------------
    # One self-contained HTML file. No internet needed, opens in any browser.
    output_file = REPO_ROOT / cfg.DASHBOARD_PATH
    dashboard.render(
        prep, feat, disc, valid, scm_result, rca_result, interv, cf,
        out_path=output_file,
        dataset_label=f"NASA C-MAPSS {cfg.DATASET}",
    )
    size_kb = output_file.stat().st_size / 1024
    say("9 dashboard", f"saved to {cfg.DASHBOARD_PATH} ({size_kb:.0f} KB)")

    print(f"\nDone in {time.time() - START_TIME:.1f} seconds.")
    print(f"Open this file to see the result:\n  {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
