"""Module 5 — SCM fitting (mechanism & noise estimation).

A causal *graph* states which variables matter; a structural causal model states
*how*. Interventions and counterfactuals are impossible without this step:
Pearl's counterfactual procedure needs the exogenous noise terms, which exist
only once a functional mechanism has been fitted and residuals recovered.

For each variable X_i with parent set PA_i (lagged and contemporaneous):

    X_i(t) = f_i( PA_i ) + e_i(t)

`f_i` is linear here. That is a deliberate prototype choice: linear-Gaussian
SCMs are identifiable given the graph, fast to fit, and yield closed-form
interventional means, which keeps the pipeline auditable. The interface accepts
any scikit-learn regressor, so a nonlinear mechanism class can be substituted
without touching Modules 6-8.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class Mechanism:
    """One fitted structural equation."""

    target: str
    parents: list[tuple[str, int]]           # (variable, lag)
    coefficients: np.ndarray
    intercept: float
    noise_std: float
    r2: float

    def predict(self, parent_values: np.ndarray) -> np.ndarray:
        return parent_values @ self.coefficients + self.intercept


@dataclass
class StructuralCausalModel:
    """Fitted SCM: mechanisms plus the evaluation order they must run in."""

    mechanisms: dict[str, Mechanism]
    var_names: list[str]
    tau_max: int
    topological_order: list[str]
    exogenous: list[str] = field(default_factory=list)

    def parents_of(self, var: str) -> list[tuple[str, int]]:
        mech = self.mechanisms.get(var)
        return list(mech.parents) if mech else []

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "variable": name,
                    "n_parents": len(m.parents),
                    "r2": m.r2,
                    "noise_std": m.noise_std,
                    "parents": ", ".join(f"{p}(t-{l})" if l else f"{p}(t)" for p, l in m.parents),
                }
                for name, m in self.mechanisms.items()
            ]
        ).sort_values("r2", ascending=False, ignore_index=True)


def _contemporaneous_order(var_names: list[str], edges: pd.DataFrame) -> list[str]:
    """Topological order of the lag-0 subgraph (must be acyclic to simulate)."""
    g = nx.DiGraph()
    g.add_nodes_from(var_names)
    for _, e in edges[edges["lag"] == 0].iterrows():
        g.add_edge(e["source"], e["target"])
    if not nx.is_directed_acyclic_graph(g):
        # Break cycles by dropping the weakest contemporaneous edge until acyclic.
        contemporaneous = edges[edges["lag"] == 0].sort_values("abs_strength")
        for _, e in contemporaneous.iterrows():
            if nx.is_directed_acyclic_graph(g):
                break
            if g.has_edge(e["source"], e["target"]):
                g.remove_edge(e["source"], e["target"])
    return list(nx.topological_sort(g))


def _design_matrix(
    df: pd.DataFrame, target: str, parents: list[tuple[str, int]], tau_max: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-engine lagged design blocks (no cross-engine rows)."""
    xs, ys = [], []
    for _, block in df.groupby("unit"):
        values = {v: block[v].to_numpy(dtype=float) for v in block.columns if v in df.columns}
        n = len(block)
        if n <= tau_max:
            continue
        cols = [values[p][tau_max - lag : n - lag] for p, lag in parents]
        xs.append(np.column_stack(cols))
        ys.append(values[target][tau_max:n])
    if not xs:
        return np.empty((0, len(parents))), np.empty(0)
    return np.vstack(xs), np.concatenate(ys)


def fit(
    df: pd.DataFrame,
    var_names: list[str],
    edges: pd.DataFrame,
    tau_max: int = 3,
    include_autoregressive: bool = True,
) -> StructuralCausalModel:
    """Fit one structural equation per variable from the validated graph.

    An autoregressive term X_i(t-1) -> X_i(t) is added by default: sensor
    signals are strongly autocorrelated, and omitting self-history would push
    that variance into the parent coefficients and inflate apparent causal
    effects.
    """
    mechanisms: dict[str, Mechanism] = {}
    exogenous: list[str] = []

    for target in var_names:
        incoming = edges[edges["target"] == target]
        parents = [(r["source"], int(r["lag"])) for _, r in incoming.iterrows()]
        if include_autoregressive and (target, 1) not in parents:
            parents.append((target, 1))
        parents = [(p, l) for p, l in parents if not (p == target and l == 0)]

        if not parents:
            exogenous.append(target)
            continue

        X, y = _design_matrix(df, target, parents, tau_max)
        if len(y) == 0:
            exogenous.append(target)
            continue

        model = LinearRegression().fit(X, y)
        residuals = y - model.predict(X)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        mechanisms[target] = Mechanism(
            target=target,
            parents=parents,
            coefficients=model.coef_.astype(float),
            intercept=float(model.intercept_),
            noise_std=float(residuals.std(ddof=1)) if len(residuals) > 1 else 0.0,
            r2=float(1 - (residuals**2).sum() / ss_tot) if ss_tot > 0 else 0.0,
        )

    return StructuralCausalModel(
        mechanisms=mechanisms,
        var_names=list(var_names),
        tau_max=tau_max,
        topological_order=_contemporaneous_order(var_names, edges),
        exogenous=exogenous,
    )


def recover_noise(scm: StructuralCausalModel, trajectory: pd.DataFrame) -> pd.DataFrame:
    """Abduction step: back out the exogenous noise e_i(t) from observed data.

    This is what makes counterfactuals possible - the recovered noise encodes
    everything about this particular engine that the mechanisms do not explain,
    and is held fixed when the counterfactual world is simulated.
    """
    n = len(trajectory)
    noise = pd.DataFrame(0.0, index=trajectory.index, columns=scm.var_names)
    for name, mech in scm.mechanisms.items():
        values = {p: trajectory[p].to_numpy(dtype=float) for p, _ in mech.parents}
        target = trajectory[name].to_numpy(dtype=float)
        for t in range(scm.tau_max, n):
            parent_vec = np.array([values[p][t - lag] for p, lag in mech.parents])
            noise.iloc[t, noise.columns.get_loc(name)] = target[t] - float(
                mech.predict(parent_vec.reshape(1, -1))[0]
            )
    return noise


def simulate(
    scm: StructuralCausalModel,
    trajectory: pd.DataFrame,
    noise: pd.DataFrame | None = None,
    interventions: dict[str, float] | None = None,
    intervention_start: int | None = None,
) -> pd.DataFrame:
    """Roll the SCM forward, optionally under do-interventions.

    Parameters
    ----------
    noise
        Recovered noise to reuse (counterfactual / twin-world simulation). When
        None the mechanisms run noise-free, giving interventional *means*.
    interventions
        {variable: value} held fixed from `intervention_start` onward.
    """
    interventions = interventions or {}
    start = scm.tau_max if intervention_start is None else max(intervention_start, scm.tau_max)
    sim = trajectory[scm.var_names].copy().astype(float).reset_index(drop=True)
    n = len(sim)

    for t in range(scm.tau_max, n):
        for var in scm.topological_order:
            if var in interventions and t >= start:
                sim.at[t, var] = interventions[var]
                continue
            mech = scm.mechanisms.get(var)
            if mech is None:
                continue
            parent_vec = np.array(
                [sim.at[t - lag, p] if lag > 0 else sim.at[t, p] for p, lag in mech.parents]
            )
            value = float(mech.predict(parent_vec.reshape(1, -1))[0])
            if noise is not None:
                value += float(noise.iloc[t][var])
            sim.at[t, var] = value
    return sim


def run(df: pd.DataFrame, var_names: list[str], edges: pd.DataFrame, tau_max: int = 3) -> dict:
    """Execute Module 5 end to end."""
    model = fit(df, var_names, edges, tau_max=tau_max)
    summary = model.summary()
    return {
        "scm": model,
        "summary": summary,
        "n_mechanisms": len(model.mechanisms),
        "n_exogenous": len(model.exogenous),
        "mean_r2": float(summary["r2"].mean()) if not summary.empty else np.nan,
    }
