"""Turning change logs into tables must not invent or lose information."""

import numpy as np
import pandas as pd

from causalpdm.pipeline import preprocess


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
