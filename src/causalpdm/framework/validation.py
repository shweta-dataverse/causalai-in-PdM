"""Module 4 — Causal graph validation.

On C-MAPSS there is no ground-truth causal graph, so "validation" here means
**robustness and plausibility**, not correctness. That distinction is stated
plainly because conflating the two is the most common overclaim in applied
causal-discovery papers. Four independent checks are run:

1. Stability      - bootstrap over engines; how often does each edge reappear?
2. Sensitivity    - does the graph survive changes to pc_alpha and tau_max?
3. Cross-method   - does an independent estimator (lagged regression with
                    Granger-style selection) agree on the strongest edges?
4. Domain check   - are edges consistent with the turbofan gas-path ordering
                    documented by Saxena et al. (2008)?

When the framework is re-targeted at causRCA, `score_against_ground_truth`
replaces checks 1-4 as the primary evaluation, since a expert-validated DAG
exists there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from . import discovery

# Gas-path station order of the turbofan (Saxena et al. 2008, Table 1).
# Lower rank = further upstream. Used only as a *plausibility prior*: an edge
# from upstream to downstream is physically expected, the reverse needs a
# feedback-control explanation.
GAS_PATH_RANK = {
    "setting_1": 0, "setting_2": 0, "setting_3": 0,   # flight condition (exogenous)
    "s_1": 1,                                          # T2   fan inlet
    "s_5": 1,                                          # P2   fan inlet
    "s_2": 2,                                          # T24  LPC outlet
    "s_6": 2,                                          # P15  bypass duct
    "s_8": 2, "s_13": 2,                               # fan speeds
    "s_3": 3,                                          # T30  HPC outlet
    "s_7": 3, "s_11": 3,                               # P30 / Ps30 HPC outlet
    "s_9": 3, "s_14": 3,                               # core speeds
    "s_12": 4,                                         # phi  fuel flow / Ps30
    "s_16": 4,                                         # farB burner fuel-air
    "s_17": 4,                                         # htBleed
    "s_20": 5,                                         # W31  HPT coolant bleed
    "s_4": 6,                                          # T50  LPT outlet
    "s_21": 6,                                         # W32  LPT coolant bleed
    "s_10": 6, "s_15": 6,                              # epr / BPR (whole-engine)
    "s_18": 0, "s_19": 0,                              # demanded speeds (commands)
}


def bootstrap_stability(
    df: pd.DataFrame,
    var_names: list[str],
    n_boot: int = 8,
    frac_units: float = 0.85,
    random_state: int = 0,
    **pcmci_kwargs,
) -> pd.DataFrame:
    """Re-run PCMCI+ on bootstrap subsamples of engines and count edge recovery.

    An edge that appears in most subsamples is a stable finding; one that
    appears in a minority is a candidate false positive.

    `frac_units` is high (0.85) on purpose. The prototype pools only ~10
    engines, and dropping 30% of such a small fleet changes the statistical
    power enough that even genuine edges disappear - the resulting recovery
    rates measure sample size, not edge reliability. Resampling 85% keeps the
    power roughly constant so the rate reflects what it is supposed to.
    """
    rng = np.random.default_rng(random_state)
    units = df["unit"].unique()
    k = max(2, int(round(frac_units * len(units))))
    tally: dict[tuple[str, str, int], int] = {}

    for _ in range(n_boot):
        chosen = rng.choice(units, size=k, replace=False)
        sub = df[df["unit"].isin(chosen)]
        edges = discovery.edge_list(
            discovery.run_pcmci_plus(sub, var_names, **pcmci_kwargs)
        )
        for _, e in edges.iterrows():
            key = (e["source"], e["target"], int(e["lag"]))
            tally[key] = tally.get(key, 0) + 1

    if not tally:
        return pd.DataFrame(columns=["source", "target", "lag", "recovery_rate"])
    return (
        pd.DataFrame(
            [{"source": s, "target": t, "lag": l, "recovery_rate": c / n_boot}
             for (s, t, l), c in tally.items()]
        )
        .sort_values("recovery_rate", ascending=False, ignore_index=True)
    )


def sensitivity_analysis(
    df: pd.DataFrame,
    var_names: list[str],
    alphas: tuple[float, ...] = (0.002, 0.005, 0.01),
    tau_maxes: tuple[int, ...] = (1, 2),
    baseline_alpha: float = 0.01,
    baseline_tau: int = 2,
    **pcmci_kwargs,
) -> pd.DataFrame:
    """Report graph size and edge overlap across hyperparameter settings.

    Each setting means another full discovery run, so the grid is kept small on
    purpose: the question being answered is "does the graph change shape when
    the knobs move", which a handful of points answers well enough.

    The swept significance levels are all deliberately strict. A loose level
    (0.05, say) keeps far more links, and because discovery cost is driven by
    graph density rather than sample count, one loose setting can take longer
    than the entire rest of the pipeline. Sweeping strict-to-stricter answers
    the stability question without that cost.
    """
    # The caller's own tau_max/pc_alpha are swept here, so drop them if present.
    pcmci_kwargs = {
        k: v for k, v in pcmci_kwargs.items() if k not in ("tau_max", "pc_alpha")
    }

    rows, edge_sets = [], {}
    for alpha in alphas:
        for tau in tau_maxes:
            res = discovery.run_pcmci_plus(
                df, var_names, tau_max=tau, pc_alpha=alpha, **pcmci_kwargs
            )
            edges = discovery.edge_list(res)
            edge_sets[(alpha, tau)] = {
                (r["source"], r["target"], int(r["lag"])) for _, r in edges.iterrows()
            }
            rows.append({"pc_alpha": alpha, "tau_max": tau, "n_edges": len(edges)})

    baseline = edge_sets.get((baseline_alpha, baseline_tau)) or next(iter(edge_sets.values()))
    for row in rows:
        current = edge_sets[(row["pc_alpha"], row["tau_max"])]
        union = baseline | current
        row["jaccard_vs_baseline"] = len(baseline & current) / len(union) if union else 1.0
    return pd.DataFrame(rows)


def cross_method_lagged_regression(
    df: pd.DataFrame, var_names: list[str], tau_max: int = 3, top_k: int = 25
) -> pd.DataFrame:
    """Independent baseline: per-target lagged linear regression importances.

    This is a deliberately different estimator family (score-based rather than
    constraint-based). Agreement on the strongest links is evidence that those
    links are not artefacts of PCMCI+'s particular test.
    """
    rows = []
    for target in var_names:
        designs, labels, ys = [], [], []
        for _, block in df.groupby("unit"):
            values = block[var_names].to_numpy(dtype=float)
            if len(values) <= tau_max:
                continue
            y = values[tau_max:, var_names.index(target)]
            cols, names = [], []
            for lag in range(1, tau_max + 1):
                cols.append(values[tau_max - lag : len(values) - lag])
                names += [(v, lag) for v in var_names]
            designs.append(np.hstack(cols))
            ys.append(y)
            labels = names
        if not designs:
            continue
        X, y = np.vstack(designs), np.concatenate(ys)
        model = LinearRegression().fit(X, y)
        for (src, lag), coef in zip(labels, model.coef_):
            if src == target and lag == 0:
                continue
            rows.append(
                {"source": src, "target": target, "lag": lag, "coef": float(coef),
                 "abs_coef": abs(float(coef))}
            )
    out = pd.DataFrame(rows).sort_values("abs_coef", ascending=False, ignore_index=True)
    return out.head(top_k)


def domain_plausibility(edges: pd.DataFrame) -> dict:
    """Check discovered edges against the turbofan gas-path ordering."""
    if edges.empty:
        return {"n_checked": 0}
    ranked = edges.copy()
    ranked["source_rank"] = ranked["source"].map(GAS_PATH_RANK)
    ranked["target_rank"] = ranked["target"].map(GAS_PATH_RANK)
    known = ranked.dropna(subset=["source_rank", "target_rank"])
    downstream = (known["source_rank"] <= known["target_rank"]).sum()
    return {
        "n_checked": int(len(known)),
        "n_downstream_consistent": int(downstream),
        "share_downstream_consistent": float(downstream / len(known)) if len(known) else np.nan,
        "detail": known,
    }


def score_against_ground_truth(
    discovered: pd.DataFrame, truth_edges: set[tuple[str, str]]
) -> dict:
    """Adjacency precision/recall/F1 and SHD against a ground-truth edge set.

    Unused on C-MAPSS (no ground truth); this is the entry point used when the
    framework is re-targeted at causRCA, whose expert DAG provides truth.
    """
    found = {(r["source"], r["target"]) for _, r in discovered.iterrows()}
    tp = len(found & truth_edges)
    fp = len(found - truth_edges)
    fn = len(truth_edges - found)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "shd": fp + fn}


def run(
    df: pd.DataFrame,
    var_names: list[str],
    edges: pd.DataFrame,
    n_boot: int = 8,
    stability_threshold: float = 0.6,
    **pcmci_kwargs,
) -> dict:
    """Execute Module 4 end to end and return the validated edge set."""
    stability = bootstrap_stability(df, var_names, n_boot=n_boot, **pcmci_kwargs)
    sensitivity = sensitivity_analysis(
        df, var_names,
        baseline_alpha=pcmci_kwargs.get("pc_alpha", 0.01),
        baseline_tau=pcmci_kwargs.get("tau_max", 2),
        **pcmci_kwargs,
    )
    cross = cross_method_lagged_regression(
        df, var_names, tau_max=pcmci_kwargs.get("tau_max", 2)
    )
    domain = domain_plausibility(edges)

    keyed = stability.set_index(["source", "target", "lag"])["recovery_rate"] if not stability.empty else pd.Series(dtype=float)
    annotated = edges.copy()
    annotated["recovery_rate"] = [
        float(keyed.get((r["source"], r["target"], int(r["lag"])), 0.0))
        for _, r in annotated.iterrows()
    ]
    validated = annotated[annotated["recovery_rate"] >= stability_threshold].reset_index(drop=True)

    return {
        "stability": stability,
        "sensitivity": sensitivity,
        "cross_method": cross,
        "domain": domain,
        "annotated_edges": annotated,
        "validated_edges": validated,
        "stability_threshold": stability_threshold,
        "n_validated": int(len(validated)),
    }
