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
    # Use the expert graph itself, so this test checks the model rather than
    # the quality of causal discovery. Each link is given at lag 0 and lag 1,
    # because the machine's rules act both immediately and after a delay and we
    # do not want the test to depend on guessing which.
    edges = pd.DataFrame([
        {"cause": cause, "effect": effect, "lag": lag,
         "strength": 0.5, "abs_strength": 0.5}
        for cause, effect in coolant.expert_graph.edge_pairs
        for lag in (0, 1)
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


def test_rare_events_can_still_be_switched_on(small_model):
    """
    Alarms fire rarely, so a model fitted on the raw data learns they are
    almost always off - so firmly that it will not switch one on even when its
    cause has failed. The re-weighted decision model exists to prevent that,
    and this test is what caught the problem originally.
    """
    model, _ = small_model
    alarm = "HP_A_700304"
    mechanism = model.mechanisms.get(alarm)
    if mechanism is None:
        pytest.skip(f"{alarm} has no equation in this graph")

    causes = [name for name, _ in mechanism.causes]
    if "HP_Pump_Ok" not in causes:
        pytest.skip("the pump is not a cause of the alarm in this graph")

    # Pump has failed, alarm not yet firing: the alarm should switch on.
    values = np.array([0.0 for _ in mechanism.causes])
    assert mechanism.will_switch_on(values), \
        "the alarm refuses to fire even though the pump it watches has failed"


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
