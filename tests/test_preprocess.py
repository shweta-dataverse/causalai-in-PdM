"""Turning change logs into tables must not invent or lose information."""

import numpy as np
import pandas as pd

from causalpdm.pipeline import data, preprocess

from conftest import TEST_NORMAL_RECORDINGS


def test_boolean_text_becomes_one_and_zero():
    assert preprocess.parse_value("True") == 1.0
    assert preprocess.parse_value("False") == 0.0
    assert preprocess.parse_value("1") == 1.0
    assert preprocess.parse_value(2.5) == 2.5
    assert np.isnan(preprocess.parse_value("some_text_state"))


def test_a_value_is_held_until_it_changes():
    """Between two recorded changes, a variable keeps its last value."""
    log = pd.DataFrame({
        "time_s": [0.0, 5.0],
        "node": ["pump", "pump"],
        "value": ["True", "False"],
        "type": ["Binary", "Binary"],
    })
    table = preprocess.to_table(log, ["pump"], seconds_per_step=1.0,
                               healthy_values={"pump": 1.0})
    # On from 0 to just before 5, off from 5 onwards.
    assert table.loc[0.0, "pump"] == 1.0
    assert table.loc[4.0, "pump"] == 1.0
    assert table.loc[5.0, "pump"] == 0.0


def test_absent_variable_falls_back_to_its_healthy_value():
    """A variable missing from a recording never changed, so hold it steady."""
    log = pd.DataFrame({
        "time_s": [0.0, 2.0],
        "node": ["pump", "pump"],
        "value": ["True", "False"],
        "type": ["Binary", "Binary"],
    })
    table = preprocess.to_table(log, ["pump", "valve"], seconds_per_step=1.0,
                               healthy_values={"pump": 1.0, "valve": 1.0})
    assert "valve" in table.columns
    assert (table["valve"] == 1.0).all()


def test_every_requested_variable_gets_a_column(coolant, coolant_tables):
    tables, _, _ = coolant_tables
    expected = coolant.expert_graph.variable_names
    for table in tables:
        assert list(table.columns) == expected


def test_tables_contain_no_missing_values(coolant_tables):
    tables, _, _ = coolant_tables
    for table in tables:
        assert not table.isna().any().any()


def test_healthy_values_match_the_machine_documentation(coolant_tables):
    """
    In healthy operation every 'Ok' signal should be on and no alarm should
    fire. If this fails, the healthy baseline is being learned wrongly.
    """
    _, _, healthy = coolant_tables
    assert healthy["HP_Pump_Ok"] == 1.0
    assert healthy["LP_Pump_Ok"] == 1.0
    assert healthy["CLF_Filter_Ok"] == 1.0
    assert healthy["HP_A_700304"] == 0.0     # alarm off
    assert healthy["CLT_A_700310"] == 0.0    # alarm off
    assert healthy["CLT_Level_lt_Min"] == 0.0


def test_both_healthy_and_fault_recordings_are_included(coolant_tables):
    _, labels, _ = coolant_tables
    assert any(label.startswith("normal") for label in labels)
    assert any(label.startswith("fault") for label in labels)


def test_every_variable_varies_once_pooled(coolant, coolant_tables):
    """A variable that never changes cannot take part in causal discovery."""
    tables, _, _ = coolant_tables
    variation = preprocess.describe_tables(tables, coolant.expert_graph.variable_names)
    assert variation["varies"].all(), \
        f"never change: {variation.loc[~variation['varies'], 'variable'].tolist()}"


def test_pressure_healthy_value_comes_from_before_the_fault():
    """
    Hydraulic pressure is never logged in a healthy recording, because it never
    changes there. It must be learned from the pre-fault part of the fault
    recordings - not left at 0 while the real machine sits at 10000.
    """
    hydraulics = data.load("hydraulics", normal_recordings=TEST_NORMAL_RECORDINGS)
    healthy = preprocess.learn_healthy_values(
        hydraulics.normal_recordings, ["Hyd_Pressure", "Hyd_A_700202"], hydraulics.fault_runs)
    assert healthy["Hyd_Pressure"] == 10000.0
    assert healthy["Hyd_A_700202"] == 0.0


def test_a_signal_that_sits_at_one_value_is_cut_around_it():
    """
    95% of pressure rows sit at 10000, so equal-sized bins would collapse to one
    cut. It must become below / at / above healthy instead.
    """
    table = pd.DataFrame({"pressure": [10000.0] * 95 + [6600.0, 4000.0, 300.0, 11248.0, 17273.0]})
    cuts = preprocess.learn_cut_points([table], {"pressure": 10000.0})
    assert preprocess.to_state("pressure", 6600.0, cuts) == 0.0
    assert preprocess.to_state("pressure", 10000.0, cuts) == 1.0
    assert preprocess.to_state("pressure", 11248.0, cuts) == 2.0


def test_cut_points_mean_the_same_in_every_recording():
    """The same value must land in the same state whichever recording it is in."""
    low = pd.DataFrame({"speed": [*np.linspace(0, 100, 49), 75.0]})
    high = pd.DataFrame({"speed": [*np.linspace(50, 150, 49), 75.0]})
    cuts = preprocess.learn_cut_points([low, high], {"speed": 0.0})
    first, second = (preprocess.discretise(t, cuts) for t in (low, high))
    assert first["speed"].iloc[-1] == second["speed"].iloc[-1]


def test_forced_pressure_conditions_resolve_to_what_the_recording_shows():
    """
    "< 6600" is a condition, not a value. It must resolve to a below-healthy
    state; "!= 10000" went upwards in exp_9, so it must not be assumed low.
    """
    hydraulics = data.load("hydraulics", normal_recordings=TEST_NORMAL_RECORDINGS)
    cuts = {"Hyd_Pressure": np.array([10000.0, np.nextafter(10000.0, np.inf)])}
    by_scenario = {run.experiment: run for run in hydraulics.fault_runs}
    assert preprocess.forced_state(by_scenario["exp_15"], "Hyd_Pressure", cuts) == 0.0
    assert preprocess.forced_state(by_scenario["exp_8"], "Hyd_Pressure", cuts) == 0.0
    assert preprocess.forced_state(by_scenario["exp_9"], "Hyd_Pressure", cuts) == 2.0


def test_inverted_temperature_labels_follow_the_recording():
    """
    exp_10 labels the temperature switches as forced True - their healthy state -
    but every recording shows them going False before the temperature alarms.
    Replaying the label would replay a fault that did not happen.
    """
    hydraulics = data.load("hydraulics", normal_recordings=TEST_NORMAL_RECORDINGS)
    run = next(r for r in hydraulics.fault_runs if r.experiment == "exp_10")
    assert run.forced_states["Hyd_Temp_lt_70"] is True
    assert preprocess.forced_state(run, "Hyd_Temp_lt_70", {}) == 0.0
    assert preprocess.forced_state(run, "Hyd_Temp_lt_80", {}) == 0.0


def test_labels_that_match_their_recording_are_kept():
    coolant = data.load("coolant", normal_recordings=TEST_NORMAL_RECORDINGS)
    run = next(r for r in coolant.fault_runs if r.experiment == "exp_27")
    assert preprocess.forced_state(run, "CLT_Level_lt_Min", {}) == 1.0
    assert preprocess.forced_state(run, "HP_Pump_Ok", {}) == 0.0
