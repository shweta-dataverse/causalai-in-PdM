"""
The structural causal model has one property that everything else depends on:

    replay a recording with its own recovered randomness and change nothing,
    and you must get back exactly what really happened.

If that fails, every counterfactual built on it is meaningless, because the
difference we report would partly be model error rather than the effect of the
change we made. This is the most important test in the suite.
"""

import numpy as np
import pandas as pd
import pytest

from causalpdm.pipeline import discovery, scm


@pytest.fixture(scope="module")
def small_model(coolant, coolant_tables):
    tables, _, _ = coolant_tables
    variables = coolant.expert_graph.variable_names
    # Use the expert graph itself, so this test checks the model and not the
    # quality of causal discovery.
    edges = pd.DataFrame([
        {"cause": cause, "effect": effect, "lag": 1, "strength": 0.5, "abs_strength": 0.5}
        for cause, effect in coolant.expert_graph.edge_pairs
    ])
    model = scm.fit(tables, variables, edges, max_lag=2)
    return model, tables


def test_an_equation_is_fitted_for_every_variable_that_moves(small_model):
    model, _ = small_model
    assert len(model.mechanisms) > 0
    for mechanism in model.mechanisms.values():
        assert mechanism.causes
        assert 0.0 <= mechanism.accuracy <= 1.0


def test_probabilities_stay_strictly_between_zero_and_one(small_model):
    """
    A probability of exactly 0 or 1 would make a variable impossible to flip,
    which would silently break every counterfactual.
    """
    model, _ = small_model
    for mechanism in model.mechanisms.values():
        values = np.zeros(len(mechanism.causes))
        chance = mechanism.probability(values)
        assert 0.0 < chance < 1.0


def test_replaying_with_recovered_randomness_reproduces_reality(small_model):
    """The consistency property. Without this, nothing downstream is trustworthy."""
    model, tables = small_model
    recording = tables[-1]

    hidden = scm.recover_hidden_randomness(model, recording)
    replayed = scm.simulate(model, recording, hidden_randomness=hidden)

    for name in model.mechanisms:
        original = recording[name].to_numpy(dtype=float)[model.max_lag:]
        returned = replayed[name].to_numpy(dtype=float)[model.max_lag:]
        assert np.array_equal(original, returned), \
            f"replaying {name} did not reproduce the original recording"


def test_holding_a_variable_fixed_actually_holds_it(small_model):
    model, tables = small_model
    recording = tables[-1]
    target = next(iter(model.mechanisms))

    result = scm.simulate(model, recording, hold_fixed={target: 1.0}, from_step=3)
    assert (result[target].to_numpy(dtype=float)[3:] == 1.0).all()


def test_the_evaluation_order_covers_every_variable(small_model):
    model, _ = small_model
    assert set(model.evaluation_order) == set(model.variables)
