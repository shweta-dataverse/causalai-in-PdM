"""
Turning a causal graph into a working model of the machine.

A causal graph says which variable affects which. It does not say by how much,
or in which direction, so on its own it cannot answer "what would happen if we
changed this?". For that we need one equation per variable, estimated from the
data. That set of equations is the structural causal model.

Every variable in the coolant subsystem is True/False, so each equation is a
logistic model: given the state of its causes a moment ago, how likely is this
variable to be on?

    P(variable is on) = logistic( weights . its causes )

**Why there is a hidden random term, and why it matters**

Each variable also carries its own randomness - everything about this
particular machine and moment that the equation does not explain. We model it
the standard way for binary outcomes: a hidden number U between 0 and 1, with

    variable is on  exactly when  U < P(variable is on)

Recovering U from a real recording is the abduction step of counterfactual
reasoning, and it is what makes an explanation specific to one fault rather
than to faults in general. We cannot pin U to an exact value from a single
True/False observation, but we can pin it to a range, and we take the middle of
that range. This keeps the important guarantee: replay the model with the same
U and no changes, and you get back exactly what really happened.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Probabilities are never allowed to reach exactly 0 or 1, because that would
# make a variable impossible to flip in a counterfactual no matter what we do.
PROBABILITY_FLOOR = 1e-4


@dataclass
class Mechanism:
    """The fitted equation for one variable."""

    variable: str
    causes: list[tuple[str, int]]      # (cause name, how many steps back)
    model: LogisticRegression
    accuracy: float                    # how often it predicts the variable correctly
    base_rate: float                   # how often the variable is on, overall

    def probability(self, cause_values: np.ndarray) -> float:
        """Chance this variable is on, given the state of its causes."""
        if len(self.causes) == 0:
            return float(self.base_rate)
        raw = self.model.predict_proba(cause_values.reshape(1, -1))[0]
        chance = float(raw[1]) if len(raw) > 1 else float(self.model.classes_[0])
        return float(np.clip(chance, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR))


@dataclass
class StructuralCausalModel:
    """All the equations, plus the order they must be evaluated in."""

    mechanisms: dict[str, Mechanism]
    variables: list[str]
    max_lag: int
    evaluation_order: list[str]
    always_on_or_off: list[str]        # variables that never changed, so no equation

    def causes_of(self, variable: str) -> list[tuple[str, int]]:
        mechanism = self.mechanisms.get(variable)
        return list(mechanism.causes) if mechanism else []

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, mechanism in self.mechanisms.items():
            rows.append({
                "variable": name,
                "number_of_causes": len(mechanism.causes),
                "accuracy": round(mechanism.accuracy, 3),
                "share_of_time_on": round(mechanism.base_rate, 3),
                "causes": ", ".join(
                    f"{cause} ({lag} steps back)" if lag else f"{cause} (same step)"
                    for cause, lag in mechanism.causes) or "none",
            })
        return pd.DataFrame(rows).sort_values("accuracy", ascending=False, ignore_index=True)


def _evaluation_order(variables: list[str], edges: pd.DataFrame) -> list[str]:
    """
    Work out a safe order to compute variables in.

    Links that act within the same time step must be computed in dependency
    order, otherwise we would be using a value we have not calculated yet. If
    those same-step links form a loop we break the weakest one, since a loop
    cannot be simulated forward at all.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(variables)
    same_step = edges[edges["lag"] == 0] if not edges.empty else edges

    for _, edge in same_step.iterrows():
        graph.add_edge(edge["cause"], edge["effect"])

    if not nx.is_directed_acyclic_graph(graph):
        weakest_first = same_step.sort_values("abs_strength")
        for _, edge in weakest_first.iterrows():
            if nx.is_directed_acyclic_graph(graph):
                break
            if graph.has_edge(edge["cause"], edge["effect"]):
                graph.remove_edge(edge["cause"], edge["effect"])

    return list(nx.topological_sort(graph))


def _training_rows(tables: list[pd.DataFrame], variable: str,
                   causes: list[tuple[str, int]], max_lag: int):
    """Line up each variable with the state of its causes a few steps earlier."""
    inputs, outputs = [], []
    for table in tables:
        if len(table) <= max_lag:
            continue
        values = {name: table[name].to_numpy(dtype=float) for name in table.columns}
        length = len(table)
        columns = [values[cause][max_lag - lag: length - lag] for cause, lag in causes]
        inputs.append(np.column_stack(columns))
        outputs.append(values[variable][max_lag:length])
    if not inputs:
        return np.empty((0, len(causes))), np.empty(0)
    return np.vstack(inputs), np.concatenate(outputs)


def fit(tables: list[pd.DataFrame], variables: list[str], edges: pd.DataFrame,
        max_lag: int, include_own_past: bool = True) -> StructuralCausalModel:
    """
    Estimate one equation per variable from the recordings.

    Each variable's own previous value is included as a cause by default. These
    signals hold their state for long stretches, so without it the model would
    wrongly credit the other causes for that persistence.
    """
    mechanisms: dict[str, Mechanism] = {}
    unchanging: list[str] = []

    for variable in variables:
        incoming = edges[edges["effect"] == variable] if not edges.empty else edges
        causes = [(row["cause"], int(row["lag"])) for _, row in incoming.iterrows()]

        if include_own_past and (variable, 1) not in causes and max_lag >= 1:
            causes.append((variable, 1))
        causes = [(name, lag) for name, lag in causes if not (name == variable and lag == 0)]

        inputs, outputs = _training_rows(tables, variable, causes, max_lag)
        if len(outputs) == 0 or len(np.unique(outputs)) < 2:
            unchanging.append(variable)
            continue

        model = LogisticRegression(max_iter=1000, C=1.0)
        model.fit(inputs, outputs)
        mechanisms[variable] = Mechanism(
            variable=variable,
            causes=causes,
            model=model,
            accuracy=float(model.score(inputs, outputs)),
            base_rate=float(outputs.mean()),
        )

    return StructuralCausalModel(
        mechanisms=mechanisms,
        variables=list(variables),
        max_lag=max_lag,
        evaluation_order=_evaluation_order(variables, edges),
        always_on_or_off=unchanging,
    )


def recover_hidden_randomness(model: StructuralCausalModel,
                              recording: pd.DataFrame) -> pd.DataFrame:
    """
    The abduction step: work out this recording's own hidden randomness.

    For each variable at each moment we know what the model expected, and what
    actually happened. That narrows the hidden number U to a range, and we take
    the middle of it. Replaying with these numbers and no intervention
    reproduces the original recording exactly, which is the property that makes
    a counterfactual about *this* fault rather than faults in general.
    """
    hidden = pd.DataFrame(0.5, index=range(len(recording)), columns=model.variables)

    for name, mechanism in model.mechanisms.items():
        observed = recording[name].to_numpy(dtype=float)
        cause_series = {cause: recording[cause].to_numpy(dtype=float)
                        for cause, _ in mechanism.causes}

        for step in range(model.max_lag, len(recording)):
            values = np.array([cause_series[cause][step - lag]
                               for cause, lag in mechanism.causes])
            chance = mechanism.probability(values)
            # Variable was on  -> U was below the chance, so take halfway below.
            # Variable was off -> U was above it, so take halfway above.
            hidden.at[step, name] = chance / 2 if observed[step] > 0.5 else (chance + 1) / 2

    return hidden


def simulate(model: StructuralCausalModel, recording: pd.DataFrame,
             hidden_randomness: pd.DataFrame | None = None,
             hold_fixed: dict[str, float] | None = None,
             from_step: int = 0) -> pd.DataFrame:
    """
    Run the model forward, optionally forcing some variables to a fixed value.

    `hold_fixed` is the do-operation: those variables are pinned to the given
    value from `from_step` onwards, and everything downstream reacts.

    Passing `hidden_randomness` replays this specific recording's own
    randomness, which is what makes the result a counterfactual rather than a
    general prediction.
    """
    hold_fixed = hold_fixed or {}
    start = max(from_step, model.max_lag)
    result = recording[model.variables].copy().astype(float).reset_index(drop=True)

    for step in range(model.max_lag, len(result)):
        for name in model.evaluation_order:
            if name in hold_fixed and step >= start:
                result.at[step, name] = float(hold_fixed[name])
                continue

            mechanism = model.mechanisms.get(name)
            if mechanism is None:
                continue

            values = np.array([
                result.at[step - lag, cause] if lag > 0 else result.at[step, cause]
                for cause, lag in mechanism.causes
            ])
            chance = mechanism.probability(values)

            if hidden_randomness is not None:
                threshold = float(hidden_randomness.at[step, name])
                result.at[step, name] = 1.0 if threshold < chance else 0.0
            else:
                result.at[step, name] = 1.0 if chance > 0.5 else 0.0

    return result
